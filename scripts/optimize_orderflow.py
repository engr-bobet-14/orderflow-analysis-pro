"""Resumable Optuna search for notebook 02's rule-based orderflow strategy.

Dates are inclusive UTC days. Each symbol/bar type has its own identity-hashed
study. See README for scoring, holdout and parallelism limitations.
"""
from __future__ import annotations

import argparse
import calendar
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone
from enum import Enum
import fcntl
import hashlib
from importlib import metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from orderflow_system.config.settings import INFO_BARS, get_all_configs
from orderflow_system.data.infobar_feed import open_bars
from orderflow_system.data.models import BarType, InfoBar
from orderflow_system.ledger import TradeLedger
from orderflow_system.pipeline import InfoBarPipeline, ReplayDigest


def json_safe(value):
    """Normalise non-finite scalars, with a status beside each affected metric."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result[key] = json_safe(item)
            if isinstance(item, (float, np.floating)) and not math.isfinite(item):
                result[key + "_status"] = "non_finite"
        return result
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def file_lock(path, *, shared=False, nonblocking=False, message="lock already held"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        try:
            fcntl.flock(handle, mode | (fcntl.LOCK_NB if nonblocking else 0))
        except BlockingIOError as exc:
            raise RuntimeError(f"{message}: {path}") from exc
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path, data):
    path = Path(path)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(json_safe(data), handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


# BEGIN DATA PROCESSING -- hashed into cache and study identity.
CACHE_VERSION = 1
MINUTE_BUILDER_VERSION = 1
DAY_MS = 86_400_000
BAR_DTYPE = np.dtype([
    ("timestamp_ms", "<i8"), ("seq", "<i8"),
    *[(name, "<f8") for name in ("open", "high", "low", "close", "volume", "buy_volume", "sell_volume")],
    ("tick_count", "<i8"), ("dollar_value", "<f8"), ("threshold", "<f8"),
    ("duration_ms", "<i8"), ("data_gap", "?"),
])
CHUNK_SIZE = 250_000


def bar_tuple(bar):
    return tuple(-1 if name == "duration_ms" and bar.duration_ms is None else getattr(bar, name)
                 for name in BAR_DTYPE.names)


class CachedBars:
    """Read-only structured-array view; construct InfoBars only as replay needs them."""
    def __init__(self, path, bar_type, start=0, stop=None):
        self.array = np.memmap(path, dtype=BAR_DTYPE, mode="r")
        self.array = self.array[start:stop]
        self.bar_type = None if bar_type == "time_1m" else BarType(bar_type)

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        if isinstance(index, slice):
            obj = object.__new__(CachedBars)
            obj.array, obj.bar_type = self.array[index], self.bar_type
            return obj
        row = self.array[index]
        values = {name: row[name].item() for name in BAR_DTYPE.names}
        if values["duration_ms"] == -1:
            values["duration_ms"] = None
        return InfoBar(**values, bar_type=self.bar_type)


def require_source(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Data root missing (is the volume mounted?): {root}")
    return root


def month_bounds(day):
    start = day.replace(day=1)
    return start, start + timedelta(days=calendar.monthrange(day.year, day.month)[1])


def trade_files(root, symbol, start, stop):
    """Partition dates are only a pruning hint; filter exact trade UTC time later."""
    files = set()
    day = start - timedelta(days=1)
    while day <= stop:
        partition = root / f"date={day.isoformat()}"
        files.update(partition.glob(f"symbol={symbol}/*.parquet"))
        files.update(partition.glob(f"hour=*/symbol={symbol}/*.parquet.ready"))
        day += timedelta(days=1)
    return sorted(files)


def inventory(root, files):
    return [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in files]


def minute_rows(root, symbol, start, stop, cache_dir):
    """One UTC day per DuckDB query; exact duplicates removed before ID validation."""
    import duckdb
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="duckdb-", dir=cache_dir) as spill_dir, duckdb.connect() as con:
        con.execute("SET threads=1")
        con.execute("SET memory_limit='512MB'")
        con.execute("SET temp_directory = ?", [spill_dir])
        day = start
        while day < stop:
            files = trade_files(root, symbol, day, day + timedelta(days=1))
            if files:
                lo = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
                con.execute("""CREATE OR REPLACE TEMP TABLE ticks AS
                    SELECT DISTINCT trade_id, trade_time_ms, price, quantity, quote_quantity, is_buyer_maker
                    FROM read_parquet(?, hive_partitioning=false)
                    WHERE trade_time_ms >= ? AND trade_time_ms < ?""", [list(map(str, files)), lo, lo + DAY_MS])
                duplicate = con.execute("SELECT trade_id FROM ticks GROUP BY trade_id HAVING count(*)>1 LIMIT 1").fetchone()
                if duplicate:
                    raise ValueError(f"Conflicting duplicate trade_id {duplicate[0]} on {day}")
                invalid = con.execute("""SELECT count(*) FROM ticks WHERE
                    trade_id IS NULL OR trade_time_ms IS NULL OR is_buyer_maker IS NULL OR
                    price IS NULL OR quantity IS NULL OR quote_quantity IS NULL OR
                    NOT isfinite(price) OR NOT isfinite(quantity) OR NOT isfinite(quote_quantity) OR
                    price <= 0 OR quantity <= 0 OR quote_quantity <= 0""").fetchone()[0]
                if invalid:
                    raise ValueError(f"Invalid price/volume/trade on {day}: {invalid} rows")
                rows = con.execute("""SELECT (trade_time_ms // 60000 + 1)*60000 AS close_ms,
                    max(trade_id), first(price ORDER BY trade_time_ms, trade_id), max(price), min(price),
                    last(price ORDER BY trade_time_ms, trade_id), sum(quantity),
                    sum(CASE WHEN NOT is_buyer_maker THEN quantity ELSE 0 END),
                    sum(CASE WHEN is_buyer_maker THEN quantity ELSE 0 END), count(*), sum(quote_quantity)
                    FROM ticks GROUP BY trade_time_ms // 60000 ORDER BY close_ms""").fetchall()
                for row in rows:
                    ts, seq, o, h, l, c, vol, buy, sell, ticks, dollars = row
                    yield InfoBar(timestamp_ms=ts, seq=seq, open=o, high=h, low=l, close=c,
                                  volume=vol, buy_volume=buy, sell_volume=sell,
                                  tick_count=ticks, dollar_value=dollars)
            day += timedelta(days=1)


def write_bar_stream(path, bars):
    """Stream to a temporary binary file and atomically publish; never buffer history."""
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    count, days, first, previous = 0, set(), None, None
    chunk = []
    try:
        with os.fdopen(fd, "wb") as handle:
            for bar in bars:
                if previous and (bar.timestamp_ms < previous[0] or bar.seq <= previous[1]):
                    raise ValueError("Bars must have chronological timestamps and strictly increasing unique seq")
                if first is None:
                    first = bar.timestamp_ms
                previous = bar.timestamp_ms, bar.seq
                days.add(bar.timestamp_ms // DAY_MS)
                chunk.append(bar_tuple(bar))
                count += 1
                if len(chunk) == CHUNK_SIZE:
                    np.asarray(chunk, dtype=BAR_DTYPE).tofile(handle)
                    chunk.clear()
            if chunk:
                np.asarray(chunk, dtype=BAR_DTYPE).tofile(handle)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    return {"n_bars": count, "first_ts_ms": first,
            "last_ts_ms": previous[0] if previous else None, "observed_days": len(days)}


def minute_stages(root, symbol, start, stop, cache_dir):
    stages = []
    month = start.replace(day=1)
    while month < stop:
        low, high = month_bounds(month)
        files = trade_files(root, symbol, low, high)
        identity = {"symbol": symbol, "month": str(month), "source_inventory": inventory(root, files),
                    "builder_version": MINUTE_BUILDER_VERSION, "data_proc_fp": DATA_PROC_FP}
        key = fingerprint(identity)
        path = cache_dir / f"minute-{key}.bin"
        manifest = path.with_suffix(".json")
        with file_lock(str(path) + ".lock"):
            if not (manifest.exists() and path.exists()):
                coverage = write_bar_stream(path, minute_rows(root, symbol, low, high, cache_dir))
                atomic_json(manifest, {**identity, **coverage, "fingerprint": key})
        stages.append((path, key, identity))
        month = high
    return stages


def assemble_minutes(stages, start_ms, end_ms):
    previous = None
    for path, _, _ in stages:
        if path.stat().st_size == 0:
            continue
        for bar in CachedBars(path, "time_1m"):
            # Duration/gaps use a predecessor even when it falls outside the requested range.
            bar.duration_ms = None if previous is None else bar.timestamp_ms - previous
            bar.data_gap = bar.duration_ms is not None and bar.duration_ms > 3_600_000
            previous = bar.timestamp_ms
            if start_ms <= bar.timestamp_ms < end_ms:
                yield bar


def build_cache(args, symbol, bar_type):
    start = args.start_date
    stop = args.end_date + timedelta(days=1)
    start_ms, end_ms = (int(pd.Timestamp(d, tz="UTC").timestamp() * 1000) for d in (start, stop))
    root = require_source(args.trades_root if bar_type == "time_1m" else args.data_root)
    cache_dir = Path(args.cache_dir)
    # Root guard precedes all writes. Sources are never opened writable.
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_metadata = {}
    if bar_type == "time_1m":
        # Full months make staging independent of the first requested sub-range.
        # Include the previous month to supply the exact midnight-close/predecessor bar.
        stage_start = (start.replace(day=1) - timedelta(days=1)).replace(day=1)
        stages = minute_stages(root, symbol, stage_start, stop, cache_dir)
        source = [{"fingerprint": key, **identity} for _, key, identity in stages]
        bars = lambda: assemble_minutes(stages, start_ms, end_ms)
    else:
        reader = open_bars(root, symbol, bar_type, start.isoformat(), stop.isoformat(), strict=True)
        source = reader.info.dataset_id
        source_metadata = {k: reader.info.sidecar.get(k) for k in ("hyperparams", "config", "created_utc", "git_sha")}
        bars = lambda: iter(reader)
    identity = {"cache_version": CACHE_VERSION, "data_proc_fp": DATA_PROC_FP, "source": source,
                "symbol": symbol, "bar_type": bar_type, "start": str(start), "end": str(args.end_date)}
    key = fingerprint(identity)
    path = cache_dir / f"bars-{key}.bin"
    manifest = path.with_suffix(".json")
    with file_lock(str(path) + ".lock"):
        if manifest.exists() and path.exists():
            metadata = json.loads(manifest.read_text())
        else:
            coverage = write_bar_stream(path, bars())
            metadata = {**identity, **coverage, "fingerprint": key, "source_metadata": source_metadata}
            atomic_json(manifest, metadata)
    if not metadata["n_bars"]:
        raise ValueError(f"No bars observed for {symbol}/{bar_type} in {start}..{args.end_date}")
    return path, metadata
# END DATA PROCESSING


# BEGIN STRATEGY -- notebook 02 cells 15-20; accounting and objective identity.
INITIAL_CAPITAL_USD = 5000.0
RISK_PER_TRADE_PCT = 0.01
MIN_STOP_LOSS_PCT = 0.0005
ANNUALIZATION_PERIODS = 365
PENALTY_SHARPE = -5.0
SEARCH_SPACE_BOUNDS = {
    "price_proximity_pct": (0.0005, 0.02, "float_log"),
    "min_composite_score": (10.0, 70.0, "float"),
    "signal_cooldown_seconds": (5.0, 120.0, "float"),
    "volume_decline_pct": (0.05, 0.6, "float"),
    "lookback_bars": (5, 30, "int"),
    "delta_failure_pct": (0.5, 0.95, "float"),
}


def instrument_config(symbol):
    for cfg in get_all_configs():
        if cfg.instrument.value == symbol:
            return cfg
    raise ValueError(f"No registered InstrumentConfig for {symbol}; enabled: "
                     + ", ".join(c.instrument.value for c in get_all_configs()))


def build_instrument_config(params, symbol="BTCUSDT"):
    base = instrument_config(symbol)
    return replace(base,
        risk=replace(base.risk, min_composite_score=params["min_composite_score"],
                     signal_cooldown_seconds=params["signal_cooldown_seconds"]),
        exhaustion=replace(base.exhaustion, volume_decline_pct=params["volume_decline_pct"]),
        divergence=replace(base.divergence, lookback_bars=params["lookback_bars"],
                           delta_failure_pct=params["delta_failure_pct"]))


def suggest_params(trial, space=None):
    params = {}
    for name, (lo, hi, kind) in (space or SEARCH_SPACE_BOUNDS).items():
        if kind == "int":
            params[name] = trial.suggest_int(name, lo, hi)
        else:
            params[name] = trial.suggest_float(name, lo, hi, log=kind == "float_log")
    return params


def run_backtest(bars, instrument_config, price_proximity_pct, bar_type=None, prefix_ends=()):
    replay = InfoBarPipeline(instrument_config, bar_type)
    replay.aggregator.price_proximity_pct = price_proximity_pct
    ledger, digest = TradeLedger(), ReplayDigest()
    prefix_digests = {}
    for idx, bar in enumerate(bars):
        result = replay.process_bar(bar)
        ledger.process_result(result)
        digest.update(result)
        if idx in prefix_ends:
            prefix_digests[idx] = digest.hexdigest()
    return {"ledger": ledger, "replay_digest": digest.hexdigest(), "prefix_digests": prefix_digests}


def closed_execution_rows(shared_ledger):
    """Notebook 02 cell 15: pair immutable snapshots; unfinished positions stay open."""
    entries = {record.execution.trade_id: record.execution for record in shared_ledger.entries}
    rows = []
    for record in shared_ledger.closes:
        closed = record.execution
        opened = entries[closed.trade_id]
        rows.append({
            "trade_id": closed.trade_id,
            "entry_seq": opened.bar_seq, "exit_seq": closed.bar_seq,
            "entry_time": pd.Timestamp(opened.timestamp_ms, unit="ms", tz="UTC"),
            "exit_time": pd.Timestamp(closed.timestamp_ms, unit="ms", tz="UTC"),
            "side": opened.direction.value,
            "entry_price": opened.fill_price, "exit_price": closed.fill_price,
            "stop_loss": opened.stop_loss, "take_profit": opened.take_profit,
            "exit_reason": closed.exit_reason, "pnl_per_unit": closed.pnl,
        })
    return pd.DataFrame(rows, columns=[
        "trade_id", "entry_seq", "exit_seq", "entry_time", "exit_time", "side",
        "entry_price", "exit_price", "stop_loss", "take_profit", "exit_reason", "pnl_per_unit",
    ])


def simulate_trades_risk_based(shared_ledger, initial_capital=INITIAL_CAPITAL_USD,
                               fee_bps=5.0, slippage_bps=1.0):
    """Notebook 02 cell 16, fixed stop-loss sizing and CLI costs."""
    trades = closed_execution_rows(shared_ledger)
    equity, rows = initial_capital, []
    for trade in trades.to_dict("records"):
        entry, exit_price = trade["entry_price"], trade["exit_price"]
        stop_pct = max(abs(entry - trade["stop_loss"]) / entry, MIN_STOP_LOSS_PCT)
        notional = min(equity * RISK_PER_TRADE_PCT / stop_pct, equity) if equity > 0 else 0.0
        quantity = notional / entry
        direction = 1.0 if trade["side"] == "buy" else -1.0
        slip = slippage_bps * 1e-4
        entry_cost_price = entry * (1 + direction * slip)
        exit_cost_price = exit_price * (1 - direction * slip)
        fee = fee_bps * 1e-4 * (entry_cost_price + exit_cost_price) * quantity
        slippage = (abs(entry_cost_price - entry) + abs(exit_cost_price - exit_price)) * quantity
        net_pnl = direction * (exit_cost_price - entry_cost_price) * quantity - fee
        rows.append({**trade, "entry_cost_price": entry_cost_price, "exit_cost_price": exit_cost_price,
                     "fee": fee, "slippage": slippage, "net_pnl": net_pnl,
                     "return_pct": net_pnl / notional if notional else 0.0,
                     "holding_seconds": (trade["exit_time"] - trade["entry_time"]).total_seconds(),
                     "stop_loss_pct": stop_pct, "position_notional": notional,
                     "capital_used": notional, "dollar_risk": notional * stop_pct,
                     "quantity": quantity, "equity_before": equity, "equity_after": equity + net_pnl})
        equity += net_pnl
    extra = ["entry_cost_price", "exit_cost_price", "fee", "slippage", "net_pnl", "return_pct",
             "holding_seconds", "stop_loss_pct", "position_notional", "capital_used", "dollar_risk",
             "quantity", "equity_before", "equity_after"]
    return pd.DataFrame(rows, columns=[*trades.columns, *extra])


def daily_returns(trades, start_ts, end_ts, initial_capital=INITIAL_CAPITAL_USD):
    """Keep day one's PnL and all trailing flat days, including the zero anchor day."""
    start, end = pd.Timestamp(start_ts).floor("D"), pd.Timestamp(end_ts).floor("D")
    anchor = start - pd.Timedelta(days=1)
    ordered = trades.sort_values("exit_time", kind="stable")
    equity = pd.Series([initial_capital, *list(initial_capital + ordered["net_pnl"].cumsum())],
                       index=pd.DatetimeIndex([anchor, *list(ordered["exit_time"])]))
    daily = equity.resample("1D").last().reindex(pd.date_range(anchor, end, freq="1D")).ffill()
    return daily.pct_change().fillna(0.0)


def compute_financial_metrics(trades, initial_capital=INITIAL_CAPITAL_USD, start_ts=None, end_ts=None):
    """Notebook 02 cell 17 formulas, corrected scored-window bounds (R1)."""
    import quantstats as qs
    if start_ts is None or end_ts is None:
        raise ValueError("Both scored-window bounds are required for daily metrics")
    returns = daily_returns(trades, start_ts, end_ts, initial_capital)
    if trades.empty:
        sharpe = sortino = drawdown = 0.0
    else:
        sharpe = float(qs.stats.sharpe(returns, periods=ANNUALIZATION_PERIODS))
        sortino = float(qs.stats.sortino(returns, periods=ANNUALIZATION_PERIODS))
        drawdown = float(qs.stats.max_drawdown((1 + returns).cumprod())) * 100.0
    return {"n_trades": len(trades), "total_pnl": float(trades["net_pnl"].sum()),
            "win_rate_pct": float((trades["net_pnl"] > 0).mean() * 100) if len(trades) else 0.0,
            "sharpe_ratio": sharpe, "sortino_ratio": sortino, "max_drawdown_pct": drawdown}


def compute_profit_factor(trades):
    if trades.empty:
        return 0.0
    gains = trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum()
    losses = -trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum()
    return float(gains / losses) if losses else None


def build_cv_folds(bars, n_splits=5, embargo=0.02):
    from RiskLabAI.backtest.validation.purged_kfold import PurgedKFold
    sequence = pd.Index(bars.array["seq"] if isinstance(bars, CachedBars) else [bar.seq for bar in bars])
    if not sequence.is_unique or not sequence.is_monotonic_increasing:
        raise ValueError("seq must be unique and strictly increasing")
    if len(bars) < n_splits:
        raise ValueError(f"Search segment N={len(bars)} is smaller than n_splits={n_splits}")
    times = pd.Series(sequence.to_numpy(), index=sequence)
    folds = []
    splitter = PurgedKFold(n_splits=n_splits, times=times, embargo=embargo)
    for train, test in splitter.split(pd.DataFrame(index=sequence)):
        start = int(test.min()) + int(len(bars) * embargo)
        if start > int(test.max()):
            raise ValueError("Embargo consumed the fold; use a longer research window or lower embargo")
        folds.append({"context_idx": train[train < test.min()], "test_idx": test, "score_start_idx": start})
    return folds


def split_timeline(bars, args):
    n = len(bars)
    holdout, blackout = int(n * args.holdout_fraction), int(n * args.embargo)
    if args.holdout_fraction > 0 and holdout <= blackout:
        raise ValueError(f"N={n}: holdout bars={holdout} must exceed holdout blackout bars={blackout}")
    n_search = n - holdout
    try:
        folds = build_cv_folds(bars[:n_search], args.n_splits, args.embargo)
    except ValueError as exc:
        raise ValueError(f"N={n}, holdout bars={holdout}, blackout bars={blackout}: {exc}") from exc
    return n_search, folds


def score_window(full, bars, start, end, min_trades):
    trades = full.loc[full["exit_seq"].between(bars[start].seq, bars[end].seq)].reset_index(drop=True)
    first, last = [pd.Timestamp(bars[i].timestamp_ms, unit="ms", tz="UTC") for i in (start, end)]
    metrics = compute_financial_metrics(trades, start_ts=first, end_ts=last)
    metrics["profit_factor"] = compute_profit_factor(trades)
    if metrics["profit_factor"] is None:
        metrics["profit_factor_status"] = "no_losses"
    blown = bool((trades["equity_after"] <= 0).any())
    score = metrics["sharpe_ratio"]
    penalized = len(trades) < min_trades or blown or not math.isfinite(score)
    return {**metrics, "account_blown": blown, "penalized": penalized,
            "fold_sharpe": PENALTY_SHARPE if penalized else score,
            "n_scored_bars": end - start + 1}, trades


def evaluate_fold(fold, bars, params, args, full=None):
    start, end = fold["score_start_idx"], int(fold["test_idx"].max())
    if full is None:
        result = run_backtest(bars[:end + 1], build_instrument_config(params, args.symbol),
                              params["price_proximity_pct"], bars[0].bar_type)
        full = simulate_trades_risk_based(result["ledger"], fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    metrics, _ = score_window(full, bars, start, end, args.min_trades_per_fold)
    return {**metrics, "n_context_bars": len(fold["context_idx"]), "n_test_bars": len(fold["test_idx"])}


def objective(trial, bars, folds, args):
    params = suggest_params(trial, args.space)
    result = run_backtest(bars, build_instrument_config(params, args.symbol),
                          params["price_proximity_pct"], bars[0].bar_type)
    full = simulate_trades_risk_based(result["ledger"], fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    results = [evaluate_fold(fold, bars, params, args, full) for fold in folds]
    scores = [r["fold_sharpe"] for r in results]
    raw = [r["sharpe_ratio"] / np.sqrt(ANNUALIZATION_PERIODS) for r in results
           if not r["penalized"] and math.isfinite(r["sharpe_ratio"])]
    attrs = {"params": params, "fold_results": results, "cv_sharpe_mean": float(np.mean(scores)),
             "cv_sharpe_std": float(np.std(scores)), "total_trades": sum(r["n_trades"] for r in results),
             "n_scored_bars": sum(r["n_scored_bars"] for r in results), "n_bars_search": len(bars),
             "n_penalized_folds": sum(r["penalized"] for r in results),
             "raw_daily_sharpe": float(np.mean(raw)) if raw else None, "replay_digest": result["replay_digest"]}
    for key, value in json_safe(attrs).items():
        trial.set_user_attr(key, value)
    return attrs["cv_sharpe_mean"]
# END STRATEGY


def source_fingerprints():
    source = Path(__file__).read_text()
    data_section = source.split("# BEGIN DATA PROCESSING", 1)[1].split("# END DATA PROCESSING", 1)[0]
    strategy_section = source.split("# BEGIN STRATEGY", 1)[1].split("# END STRATEGY", 1)[0]
    reader = (ROOT / "orderflow_system/data/infobar_feed.py").read_bytes()
    data_fp = hashlib.sha256(reader + data_section.encode()).hexdigest()
    names = ["pipeline.py", "ledger.py", "config/settings.py", "data/models.py", "utils/clock.py", "data/infobar_feed.py"]
    paths = [ROOT / "orderflow_system" / name for name in names]
    for directory in ("analytics", "patterns", "signals"):
        paths.extend(sorted((ROOT / "orderflow_system" / directory).glob("*.py")))
    h = hashlib.sha256(strategy_section.encode())
    for path in sorted(paths):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return data_fp, h.hexdigest()


DATA_PROC_FP, STRATEGY_FP = source_fingerprints()


def library_versions(bar_type=None):
    names = ("optuna", "RiskLabAI", "quantstats", "pandas", "numpy")
    if bar_type == "time_1m":
        names += ("duckdb",)
    return {name: metadata.version(name) for name in names}


def study_identity(args, metadata):
    identity = {"search_space": args.space, "strategy_fp": STRATEGY_FP,
                "instrument_defaults": asdict(instrument_config(args.symbol)),
                "initial_capital_usd": INITIAL_CAPITAL_USD, "risk_per_trade_pct": RISK_PER_TRADE_PCT,
                "min_stop_loss_pct": MIN_STOP_LOSS_PCT, "annualization_periods": ANNUALIZATION_PERIODS,
                "penalty_sharpe": PENALTY_SHARPE, "fee_bps": args.fee_bps, "slippage_bps": args.slippage_bps,
                "n_splits": args.n_splits, "embargo": args.embargo,
                "holdout_fraction": args.holdout_fraction, "min_trades_per_fold": args.min_trades_per_fold,
                "seed": args.seed}
    space_fp = fingerprint(identity)
    name = (f"{args.symbol}__{args.bar_type}__{args.start_date}_{args.end_date}"
            f"__d{metadata['fingerprint'][:8]}__s{space_fp[:8]}")
    return name, {"data_fp": metadata["fingerprint"], "space_fp": space_fp, **json_safe(identity)}


def study_lock_path(db, name):
    return Path(str(db) + "." + hashlib.sha256(name.encode()).hexdigest()[:16] + ".lock")


def open_study(args, name, identity, worker_index=0):
    import optuna
    args.study_db.parent.mkdir(parents=True, exist_ok=True)
    # Database-wide schema/Alembic/create-study serialisation, before study admission.
    with file_lock(str(args.study_db) + ".init.lock"):
        storage = optuna.storages.RDBStorage(f"sqlite:///{args.study_db.resolve()}",
                                           engine_kwargs={"connect_args": {"timeout": 60}})
        study = optuna.create_study(storage=storage, study_name=name, direction="maximize",
                                   sampler=optuna.samplers.TPESampler(seed=args.seed),
                                   pruner=optuna.pruners.NopPruner(), load_if_exists=True)
        stored = study.user_attrs
        versions = library_versions(args.bar_type)
        if "identity" in stored and stored["identity"] != identity:
            raise ValueError(f"Study identity mismatch: {name}")
        for package in ("RiskLabAI", "quantstats", "optuna"):
            if "versions" in stored and stored["versions"][package].split(".")[:2] != versions[package].split(".")[:2]:
                raise ValueError(f"{package} major.minor changed; resume refused; start a new study/database")
        if "identity" not in stored:
            study.set_user_attr("identity", identity)
            study.set_user_attr("versions", versions)
        study.sampler = optuna.samplers.TPESampler(seed=(args.seed + worker_index * 1_000_003 + len(study.trials)) % 2**32,
                                                  constant_liar=args.n_jobs > 1)
    return study


def complete_trials(study):
    from optuna.trial import TrialState
    return study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))


def optimize_loop(study, bars, folds, args, parent_pid=None):
    import optuna
    while parent_pid is None or os.getppid() == parent_pid:
        if len(complete_trials(study)) >= args.n_trials:
            break
        # Per-trial seeding makes sequential resumes reproduce uninterrupted proposals.
        study.sampler = optuna.samplers.TPESampler(seed=(args.seed + getattr(args, "worker_index", 0) * 1_000_003 + len(study.trials)) % 2**32,
                                                  constant_liar=args.n_jobs > 1)
        study.optimize(lambda trial: objective(trial, bars, folds, args), n_trials=1)


def worker_main(args, name, identity, cache_path, n_search, parent_pid, index):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    lock = study_lock_path(args.study_db, name)
    with file_lock(str(lock) + ".workers", shared=True):
        if os.getppid() != parent_pid:
            return
        args.worker_index = index
        study = open_study(args, name, identity, index)
        bars = CachedBars(cache_path, args.bar_type, stop=n_search)
        folds = build_cv_folds(bars, args.n_splits, args.embargo)
        optimize_loop(study, bars, folds, args, parent_pid)


def run_workers(study, args, name, identity, cache_path, n_search):
    count = min(args.n_jobs, max(0, args.n_trials - len(complete_trials(study))))
    workers = []
    try:
        for index in range(count):
            worker = mp.get_context("spawn").Process(target=worker_main,
                args=(args, name, identity, cache_path, n_search, os.getpid(), index))
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()
        failures = [(w.pid, w.exitcode) for w in workers if w.exitcode != 0]
        if failures:
            raise RuntimeError(f"Optimizer worker failure(s) (pid, exit code): {failures}; failed trials retained")
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join()


def deflated_gate(study, returns, *, ineligible_reason=None):
    from RiskLabAI.backtest.validation.leakage_aware_hpo import deflated_sharpe_gate
    trials = complete_trials(study)
    raw = [t.user_attrs.get("raw_daily_sharpe") for t in trials]
    raw = [s for s in raw if s is not None and math.isfinite(s)]
    std = float(np.std(raw, ddof=1)) if len(raw) >= 2 else None
    counts = {"n_trials": len(trials), "n_dispersion_trials": len(raw), "trial_sharpe_std": std}
    if ineligible_reason:
        return json_safe({**counts, "passes": None, "status": "indeterminate", "reason": ineligible_reason})
    if std is None or not math.isfinite(std) or std <= 0:
        return json_safe({**counts, "passes": None, "status": "indeterminate",
                          "reason": "Need at least two finite raw trial Sharpes with positive sample dispersion"})
    values = np.asarray(returns, dtype=float)
    if len(values) < 3 or not np.isfinite(values).all() or np.std(values) <= 0:
        return {**counts, "passes": None, "status": "indeterminate", "reason": "Insufficient varying finite daily returns"}
    gate = deflated_sharpe_gate(values, n_trials=len(trials), trial_sharpe_std=std)
    if not math.isfinite(gate["deflated_sharpe"]):
        gate.update(passes=None, status="indeterminate", reason="RiskLabAI returned a non-finite deflated Sharpe")
    else:
        gate["status"] = "evaluated"
    return json_safe({**gate, **counts})


def evaluate_holdout(study, bars, n_search, args):
    prior_evaluations = sum(key.startswith("holdout::") for key in study.user_attrs)
    if prior_evaluations > 0:
        print("Warning: repeated looks erode holdout independence "
              f"({prior_evaluations} prior evaluations)")
    trials = complete_trials(study)
    candidates = [t for t in trials
                  if t.user_attrs.get("n_penalized_folds", args.n_splits) < args.n_splits]
    independent = args.bar_type == "time_1m"
    base = {"prior_evaluations": prior_evaluations, "trials_at_evaluation": len(trials),
            "bar_construction_independent": independent,
            "caveat": ("Time bars have no calibrated construction parameters" if independent else
                       "Conditional on bars calibrated on the full history; not independent end-to-end validation")}
    if not candidates:
        return {**base, "status": "not_evaluated", "reason": "No non-all-penalized trial"}
    best = max(candidates, key=lambda t: t.value)
    replay = run_backtest(bars, build_instrument_config(best.params, args.symbol),
                           best.params["price_proximity_pct"], bars[0].bar_type)
    full = simulate_trades_risk_based(replay["ledger"], fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    start = n_search + int(len(bars) * args.embargo)
    metrics, trades = score_window(full, bars, start, len(bars) - 1, args.min_trades_per_fold)
    returns = daily_returns(trades, pd.Timestamp(bars[start].timestamp_ms, unit="ms", tz="UTC"),
                             pd.Timestamp(bars[-1].timestamp_ms, unit="ms", tz="UTC"))
    reasons = []
    if metrics["n_trades"] < args.min_trades_per_fold:
        reasons.append("Holdout trade count is below min_trades_per_fold")
    if metrics["account_blown"]:
        reasons.append("Holdout account blown")
    if not math.isfinite(metrics["sharpe_ratio"]):
        reasons.append("Holdout Sharpe is non-finite")
    return json_safe({**base, **metrics, "status": "evaluated", "trial_number": best.number,
                      "n_bars": len(bars) - n_search, "replay_digest": replay["replay_digest"],
                      "daily_returns": [{"day": str(day.date()), "return": float(r)} for day, r in returns.items()],
                      "deflated_sharpe_gate": deflated_gate(study, returns, ineligible_reason="; ".join(reasons))})


def git_state():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    try:
        return {"sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (subprocess.CalledProcessError, OSError) as exc:
        return {"sha": None, "dirty": None, "error": str(exc)}


def study_result(study, args, metadata, identity):
    trials = complete_trials(study)
    best = max(trials, key=lambda t: t.value) if trials else None
    best_degenerate = best is not None and best.user_attrs.get("n_penalized_folds") == args.n_splits
    if best_degenerate:
        print(f"Warning: {study.study_name}: reported best trial has all folds penalised")
    return json_safe({"study_name": study.study_name, "symbol": args.symbol, "bar_type": args.bar_type,
        "best_params": best.params if best else None, "best_objective": best.value if best else None,
        "best_trial_number": best.number if best else None, "best_degenerate": best_degenerate,
        "n_scored_bars": best.user_attrs["n_scored_bars"] if best else 0,
        "total_trades": best.user_attrs["total_trades"] if best else 0,
        "best_trial_attrs": best.user_attrs if best else {},
        "trial_counts": {state: sum(t.state.name == state for t in study.trials)
                         for state in ("COMPLETE", "FAIL", "RUNNING", "PRUNED", "WAITING")},
        "identity": identity, "dataset": metadata, "seed": args.seed, "versions": study.user_attrs["versions"],
        "git": git_state(), "holdout_history": {key: value for key, value in study.user_attrs.items()
                                                 if key.startswith("holdout::")},
        "limitations": ["point-in-time purge is a no-op; no fitted training set",
                         "embargo is a scoring blackout; outcomes attributed by exit bar; no forced liquidation",
                         "cold fold 0; compounded prefix sizing; metrics anchored at $5000 per scored window",
                         "tuned CV is optimistically biased; adjacent folds are weakly dependent",
                         "DSR multiplicity counts only this study; repeated holdout evaluation erodes independence",
                         "small trade counts give noisy daily Sharpe",
                         "parallel proposals depend on completion order; target overshoot <= n_jobs-1"]})


def save_results(path, entry):
    with file_lock(str(path) + ".lock"):
        data = json.loads(path.read_text()) if path.exists() else {}
        if not isinstance(data, dict):
            raise ValueError(f"Results must contain a JSON object: {path}")
        studies = data.setdefault("symbols", {}).setdefault(entry["symbol"], {}).setdefault(entry["bar_type"], {})
        previous = studies.get(entry["study_name"], {})
        entry = {**entry, "holdout_history": {**previous.get("holdout_history", {}), **entry["holdout_history"]}}
        studies[entry["study_name"]] = entry
        atomic_json(path, data)
    return data


def optimize_study(args, cache_path, metadata):
    name, identity = study_identity(args, metadata)
    study = open_study(args, name, identity)
    lock = study_lock_path(args.study_db, name)
    bars = CachedBars(cache_path, args.bar_type)
    n_search, folds = split_timeline(bars, args)
    with file_lock(lock, nonblocking=True, message="Same study already running"):
        # Orphan workers retain their own shared descriptors even after parent death.
        with file_lock(str(lock) + ".workers", nonblocking=True,
                       message="orphaned workers still running; wait or kill pid(s)"):
            pass
        if args.n_jobs == 1:
            optimize_loop(study, bars[:n_search], folds, args)
        else:
            run_workers(study, args, name, identity, cache_path, n_search)
        with file_lock(str(lock) + ".workers", nonblocking=True,
                       message="orphaned workers still running; wait or kill pid(s)"):
            if args.evaluate_holdout:
                evaluation = evaluate_holdout(study, bars, n_search, args)
                key = "holdout::" + datetime.now(timezone.utc).isoformat() + "::" + uuid.uuid4().hex[:8]
                study.set_user_attr(key, evaluation)
                print(f"Holdout: {evaluation['caveat']}")
            entry = study_result(study, args, metadata, identity)
            save_results(args.output, entry)
    print(f"{args.symbol}/{args.bar_type}: best={entry['best_objective']}, "
          f"n_scored_bars={entry['n_scored_bars']}, total_trades={entry['total_trades']}, "
          f"trials={entry['trial_counts']} [{name}]")
    return entry


def positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def nonnegative(value):
    n = float(value)
    if not math.isfinite(n) or n < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return n


def fraction(value):
    n = float(value)
    if not math.isfinite(n) or not 0 <= n < 1:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return n


def symbols(value):
    values = value.upper().split(",")
    if any(not re.fullmatch(r"[A-Z0-9]{2,20}", s) for s in values):
        raise argparse.ArgumentTypeError("invalid comma-separated symbols")
    return list(dict.fromkeys(values))


def bar_types(value):
    info = [b.value for b in BarType]
    values = [bar for token in value.split(",") for bar in (info if token == "all" else [token])]
    if any(b not in info + ["time_1m"] for b in values):
        raise argparse.ArgumentTypeError("unknown bar type; use all, time_1m or the six fixed info-bar types")
    return list(dict.fromkeys(values))


def load_space(path):
    space = dict(SEARCH_SPACE_BOUNDS)
    if path is None:
        return space
    overrides = json.loads(Path(path).read_text())
    if not isinstance(overrides, dict):
        raise ValueError("Search-space JSON must be an object")
    for name, bounds in overrides.items():
        if name not in space:
            raise ValueError(f"Unknown search dimension: {name}")
        if not isinstance(bounds, dict) or set(bounds) != {"low", "high"}:
            raise ValueError(f"{name}: expected {{low, high}} bounds")
        lo, hi, kind = bounds["low"], bounds["high"], space[name][2]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (lo, hi)):
            raise ValueError(f"{name}: bounds must be finite numbers")
        if lo > hi or lo <= 0 or (kind == "int" and any(int(v) != v for v in (lo, hi))):
            raise ValueError(f"{name}: invalid ordered positive bounds")
        if name in ("volume_decline_pct", "delta_failure_pct") and hi > 1:
            raise ValueError(f"{name}: fractions must not exceed 1")
        space[name] = (int(lo), int(hi), kind) if kind == "int" else (lo, hi, kind)
    return space


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", type=symbols, default="BTCUSDT")
    parser.add_argument("--bar-type", type=bar_types, default="all",
                        help="comma-separated bar types; all expands to six info types; add time_1m explicitly (all,time_1m)")
    parser.add_argument("--start-date", type=date.fromisoformat, default=os.getenv("START_DATE", "2019-09-10"))
    parser.add_argument("--end-date", type=date.fromisoformat, default=os.getenv("END_DATE", "2023-12-31"), help="inclusive UTC day")
    for flag, default in (("n-trials", 50), ("n-jobs", 1), ("n-splits", 5), ("min-trades-per-fold", 2)):
        parser.add_argument("--" + flag, type=positive_int, default=default)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embargo", type=fraction, default=0.02)
    parser.add_argument("--holdout-fraction", type=fraction, default=0.2)
    parser.add_argument("--evaluate-holdout", action="store_true")
    parser.add_argument("--fee-bps", type=nonnegative, default=5.0)
    parser.add_argument("--slippage-bps", type=nonnegative, default=1.0)
    parser.add_argument("--search-space", type=Path)
    parser.add_argument("--data-root", type=Path, default=INFO_BARS.root)
    parser.add_argument("--trades-root", type=Path, default=os.getenv("ORDERFLOW_TRADES_ROOT", "/Volumes/Market-Data/orderbook-data_binance/futures_trades"))
    output = ROOT / "output/optimize"
    parser.add_argument("--study-db", type=Path, default=output / "studies.db")
    parser.add_argument("--cache-dir", type=Path, default=output / "cache")
    parser.add_argument("--output", type=Path, default=output / "results.json")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        for name in ("study_db", "output", "cache_dir"):
            setattr(args, name, getattr(args, name).expanduser().resolve())
        if args.start_date > args.end_date:
            raise ValueError("start-date must not be after end-date")
        if args.n_splits < 2:
            raise ValueError("n-splits must be >= 2")
        if not 0 <= args.seed <= 2**32 - 1:
            raise ValueError("seed must be in [0, 2**32-1]")
        if args.slippage_bps >= 10_000:
            raise ValueError("slippage-bps must be < 10000 to keep cost prices positive")
        if args.evaluate_holdout and args.holdout_fraction == 0:
            raise ValueError("evaluate-holdout requires positive holdout-fraction")
        for symbol in args.symbol:
            instrument_config(symbol)
        args.space = load_space(args.search_space)
        # Validate all mount roots before doing any work.
        if any(b != "time_1m" for b in args.bar_type):
            require_source(args.data_root)
        if "time_1m" in args.bar_type:
            require_source(args.trades_root)
        results = {"symbols": {}}
        for symbol in args.symbol:
            results["symbols"][symbol] = {}
            for bar_type in args.bar_type:
                study_args = argparse.Namespace(**vars(args))
                study_args.symbol, study_args.bar_type = symbol, bar_type
                path, metadata = build_cache(study_args, symbol, bar_type)
                # Reject unusable holdout/CV geometry before creating or running a study.
                split_timeline(CachedBars(path, bar_type), study_args)
                entry = optimize_study(study_args, path, metadata)
                results["symbols"][symbol][bar_type] = {entry["study_name"]: entry}
        return results
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

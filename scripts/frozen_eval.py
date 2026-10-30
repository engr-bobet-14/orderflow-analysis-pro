"""Frozen-parameter evaluation over a dated window, shared by validation and dev-window checks.

The caller supplies the window guard (dataset policy); it runs before any data access. Trades count
if their ENTRY bar is inside the window (earlier entries are warm-up and excluded even if they exit
inside); a position still open at the window end is excluded and reported.
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orderflow_system.data.infobar_feed import find_dataset
from scripts import diagnose_trades as dx
from scripts import optimize_orderflow as opt
from scripts.dataset_policy import window_slice


def window_ledger(ledger, first_seq):
    """Ledger restricted to entries at or after first_seq, plus the count still open at the end."""
    entries = [e for e in ledger.entries if e.execution.bar_seq >= first_seq]
    ids = {e.execution.trade_id for e in entries}
    closes = [c for c in ledger.closes if c.execution.trade_id in ids]
    return SimpleNamespace(entries=entries, closes=closes), len(entries) - len(closes)


CONSTRUCTION_KEYS = ("hyperparams", "config", "created_utc", "git_sha")


def construction_problems(expected, source, source_metadata, data_proc_fp):
    """Differences between the development study's bar construction and what would be read now."""
    problems = []
    if isinstance(expected.get("source"), str) and source != expected["source"]:
        problems.append(f"source {source!r} != {expected['source']!r}")
    if {k: (source_metadata or {}).get(k) for k in CONSTRUCTION_KEYS} != \
            {k: (expected.get("source_metadata") or {}).get(k) for k in CONSTRUCTION_KEYS}:
        problems.append("source_metadata (calibration hyperparameters, config, creation time or git SHA) differs")
    if data_proc_fp is not None and data_proc_fp != expected.get("data_proc_fp"):
        problems.append(f"data_proc_fp {data_proc_fp!r} != {expected.get('data_proc_fp')!r}")
    return problems


def evaluate_frozen(params, *, symbol, bar_type, start, end, settings, data_root, cache_dir, trades_root=None,
                    warmup_days=90, guard, min_trades=1, expected_dataset):
    """`expected_dataset` is the development cache's {source, source_metadata, data_proc_fp}; a different bar
    construction is refused before any bars are read."""
    guard(start, end)
    if bar_type == "time_1m" and trades_root is None:
        raise ValueError("time_1m validation needs trades_root (the tick archive); pass --trades-root")
    if bar_type != "time_1m":                 # newest-stamp discovery could silently pick a recalibrated artifact
        info = find_dataset(data_root, symbol, bar_type)
        problems = construction_problems(expected_dataset, info.dataset_id, info.sidecar, None)
        if problems:
            raise ValueError("bar construction differs from the development study: " + "; ".join(problems))
    cache_args = SimpleNamespace(start_date=start - timedelta(days=warmup_days), end_date=end,
                                 data_root=data_root, trades_root=trades_root, cache_dir=cache_dir)
    path, built = opt.build_cache(cache_args, symbol, bar_type)
    problems = construction_problems(expected_dataset, built.get("source") if bar_type != "time_1m" else None,
                                     built.get("source_metadata"), built.get("data_proc_fp"))
    if problems:
        raise ValueError("bar construction differs from the development study: " + "; ".join(problems))
    bars = opt.CachedBars(path, bar_type)
    first, stop = window_slice(bars.array["timestamp_ms"], start, end)
    if first >= stop:
        raise ValueError("no bars in the window")
    replay = bars[:stop]
    result = opt.run_backtest(replay, opt.build_instrument_config(params, symbol),
                              params["price_proximity_pct"], replay[0].bar_type)
    ledger, n_open = window_ledger(result["ledger"], int(bars.array["seq"][first]))
    full = opt.simulate_trades_risk_based(ledger, fee_bps=settings["fee_bps"], slippage_bps=settings["slippage_bps"])
    metrics, trades = opt.score_window(full, bars, first, stop - 1, min_trades)
    trades = dx.add_gross(trades)
    first_ts, last_ts = (opt.pd.Timestamp(bars[i].timestamp_ms, unit="ms", tz="UTC") for i in (first, stop - 1))
    risk = trades["dollar_risk"].where(trades["dollar_risk"] > 0) if len(trades) else trades.get("dollar_risk")
    return opt.json_safe({
        "window": {"start": str(start), "end": str(end)}, "metrics": metrics,
        "decomposition": dx.decompose(trades),
        "trade_r": [] if not len(trades) else (trades["net_pnl"] / risk).dropna().tolist(),
        "daily_returns": opt.daily_returns(trades, first_ts, last_ts).tolist(),
        "dataset": {"source": built.get("source"), "fingerprint": built.get("fingerprint"),
                    "data_proc_fp": built.get("data_proc_fp"), "source_metadata": built.get("source_metadata")},
        "n_warmup_bars": first, "n_bars": stop - first, "n_open_at_end": n_open,
        "replay_digest": result["replay_digest"]})

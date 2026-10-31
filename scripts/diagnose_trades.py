"""Read-only trade diagnostics for a stored Optuna trial (2019-2023 development window only).

Replays one trial's frozen parameters over the cached search window and splits P&L into
gross edge, fees and slippage. Never opens the source parquet and never runs trials.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import date
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts import optimize_orderflow as opt
from scripts.dataset_policy import ensure_dev_window

DEFAULT_STOP_PCT = 0.003


def _trial_rows(study_db, study_name):
    """{number: (value, user_attrs)} for COMPLETE trials, read-only."""
    con = sqlite3.connect(f"file:{Path(study_db)}?mode=ro", uri=True)
    try:
        study = con.execute("select study_id from studies where study_name = ?", (study_name,)).fetchone()
        if study is None:
            raise ValueError(f"study not found: {study_name}")
        rows = con.execute("""select t.trial_id, t.number, v.value from trials t
            left join trial_values v on v.trial_id = t.trial_id
            where t.study_id = ? and t.state = 'COMPLETE'""", study).fetchall()
        out = {}
        for trial_id, number, value in rows:
            attrs = {key: json.loads(raw) for key, raw in con.execute(
                "select key, value_json from trial_user_attributes where trial_id = ?", (trial_id,))}
            out[number] = (value, attrs)
        return out
    finally:
        con.close()


def load_trial(study_db, study_name, trial):
    """(params, trial number): the explicit trial, else the best feasible one."""
    trials = _trial_rows(study_db, study_name)
    if trial is None:
        if not any("feasible" in attrs for _, attrs in trials.values()):
            raise ValueError("legacy study has no feasibility flag; pass --trial")
        feasible = {n: v for n, (v, attrs) in trials.items() if attrs.get("feasible") and v is not None}
        if not feasible:
            raise ValueError("no feasible trial")
        trial = max(feasible, key=feasible.get)
    if trial not in trials:
        raise ValueError(f"trial {trial} not found in {study_name}")
    return trials[trial][1]["params"], trial


def stored_digest(study_db, study_name, trial):
    return _trial_rows(study_db, study_name)[trial][1].get("replay_digest")


REPLAY_KEYS = ("n_splits", "embargo", "holdout_fraction", "fee_bps", "slippage_bps")


def load_study_attr(study_db, study_name, key):
    """A study-level user attribute (read-only); None when absent."""
    con = sqlite3.connect(f"file:{Path(study_db)}?mode=ro", uri=True)
    try:
        row = con.execute("""select a.value_json from study_user_attributes a
            join studies s on s.study_id = a.study_id where s.study_name = ? and a.key = ?""",
                          (study_name, key)).fetchone()
    finally:
        con.close()
    return None if row is None else json.loads(row[0])


def load_identity(study_db, study_name):
    """The identity dict the study was created with (read-only)."""
    identity = load_study_attr(study_db, study_name, "identity")
    if identity is None:
        raise ValueError(f"study {study_name} has no stored identity")
    return identity


def bind_replay_settings(identity):
    """Replay settings are the study's own, never CLI choices; accounting constants must be unchanged."""
    missing = [key for key in (*REPLAY_KEYS, "data_fp", "initial_capital_usd", "risk_per_trade_pct",
                               "min_stop_loss_pct", "annualization_periods") if key not in identity]
    if missing:
        raise ValueError(f"stored study identity lacks {', '.join(missing)}")
    current = {"initial_capital_usd": opt.INITIAL_CAPITAL_USD, "risk_per_trade_pct": opt.RISK_PER_TRADE_PCT,
               "min_stop_loss_pct": opt.MIN_STOP_LOSS_PCT, "annualization_periods": opt.ANNUALIZATION_PERIODS}
    changed = [key for key, value in current.items() if identity[key] != value]
    if changed:
        raise ValueError(f"accounting constants changed since the study was created: {', '.join(changed)}")
    return {key: identity[key] for key in REPLAY_KEYS}


def verify_replay(actual, expected):
    """The replay must reproduce the study's digest; an absent digest cannot be verified."""
    if expected is None:
        raise RuntimeError("stored replay digest missing; trial is unverifiable")
    if actual != expected:
        raise RuntimeError(f"replay digest mismatch: replayed {actual} != stored {expected}; "
                           "strategy, data or cache differs from the study")


def add_gross(trades):
    """Add gross_pnl = direction * (exit - entry) * quantity (before slippage and fees)."""
    out = trades.copy()
    direction = np.where(out["side"] == "buy", 1.0, -1.0)
    out["gross_pnl"] = direction * (out["exit_price"] - out["entry_price"]) * out["quantity"]
    return out


def replay_trades(bars, params, args):
    """Closed trades from replaying the search segment only (never the pseudo-holdout)."""
    n_search, _ = opt.split_timeline(bars, args)
    search = bars[:n_search]
    result = opt.run_backtest(search, opt.build_instrument_config(params, args.symbol),
                              params["price_proximity_pct"], search[0].bar_type)
    trades = add_gross(opt.simulate_trades_risk_based(
        result["ledger"], fee_bps=args.fee_bps, slippage_bps=args.slippage_bps))
    trades.attrs["replay_digest"] = result["replay_digest"]
    return trades


def scored_only(trades, bars, folds):
    """Trades the objective actually scored: exits inside any fold's [score_start, test_end] window."""
    mask = pd.Series(False, index=trades.index)
    for fold in folds:
        low, high = bars[fold["score_start_idx"]].seq, bars[int(fold["test_idx"].max())].seq
        mask |= trades["exit_seq"].between(low, high)
    return trades[mask].reset_index(drop=True)


def decompose(trades):
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "gross_pnl": 0.0, "fee_total": 0.0, "slippage_total": 0.0, "net_pnl": 0.0,
                "gross_expectancy_r": None, "net_expectancy_r": None, "std_r": None, "se_r": None,
                "cost_r_mean": None, "win_rate_pct": None, "exit_reason_counts": {},
                "median_holding_s": None, "median_stop_pct": None, "share_default_stop": None}
    risk = trades["dollar_risk"].where(trades["dollar_risk"] > 0)
    net_r = trades["net_pnl"] / risk
    std = float(net_r.std(ddof=1)) if n > 1 else None
    return {"n_trades": n,
            "gross_pnl": float(trades["gross_pnl"].sum()), "fee_total": float(trades["fee"].sum()),
            "slippage_total": float(trades["slippage"].sum()), "net_pnl": float(trades["net_pnl"].sum()),
            "gross_expectancy_r": float((trades["gross_pnl"] / risk).mean()),
            "net_expectancy_r": float(net_r.mean()), "std_r": std,
            "se_r": std / math.sqrt(n) if std is not None else None,
            "cost_r_mean": float(((trades["fee"] + trades["slippage"]) / risk).mean()),
            "win_rate_pct": float((trades["net_pnl"] > 0).mean() * 100),
            "exit_reason_counts": {str(k): int(v) for k, v in trades["exit_reason"].value_counts().items()},
            "median_holding_s": float(trades["holding_seconds"].median()),
            "median_stop_pct": float(trades["stop_loss_pct"].median()),
            "share_default_stop": float(((trades["stop_loss_pct"] - DEFAULT_STOP_PCT).abs() < 1e-9).mean())}


def trades_needed(delta_r, std_r, alpha=0.05, power=0.8):
    """One-sided test of mean R > 0: trades to detect an edge of delta_r."""
    z = NormalDist().inv_cdf(1 - alpha) + NormalDist().inv_cdf(power)
    return math.ceil((z * std_r / delta_r) ** 2)


def floors(decomposition, n_splits, delta_multiplier=1.0):
    cost, std = decomposition.get("cost_r_mean"), decomposition.get("std_r")
    if not cost or not std or cost <= 0 or std <= 0:
        raise ValueError("floors need a measured positive cost_r_mean and std_r")
    delta = cost * delta_multiplier
    total = trades_needed(delta, std)
    return {"delta_r": delta, "std_r": std, "min_total_trades": total,
            "min_trades_per_fold": math.ceil(0.5 * total / n_splits)}


def reprice_costs(trades, fee_bps, slippage_bps):
    """Same entries, exits and quantities under other costs (compounded sizing is not re-derived)."""
    out = trades.copy()
    direction = np.where(out["side"] == "buy", 1.0, -1.0)
    slip = slippage_bps * 1e-4
    entry_cost = out["entry_price"] * (1 + direction * slip)
    exit_cost = out["exit_price"] * (1 - direction * slip)
    out["fee"] = fee_bps * 1e-4 * (entry_cost + exit_cost) * out["quantity"]
    out["slippage"] = ((entry_cost - out["entry_price"]).abs() + (exit_cost - out["exit_price"]).abs()) * out["quantity"]
    out["net_pnl"] = direction * (exit_cost - entry_cost) * out["quantity"] - out["fee"]
    return out


def cost_grid(trades, fees=(0.0, 2.0, 5.0), slips=(0.0, 1.0, 3.0, 5.0)):
    risk = trades["dollar_risk"].where(trades["dollar_risk"] > 0)
    rows = []
    for fee in fees:
        for slip in slips:
            net = reprice_costs(trades, fee, slip)["net_pnl"]
            rows.append({"fee_bps": fee, "slippage_bps": slip, "net_pnl": float(net.sum()),
                         "net_expectancy_r": float((net / risk).mean()) if len(net) else None})
    return pd.DataFrame(rows)


def next_bar_entry(trades, bars, fee_bps=5.0, slippage_bps=1.0):
    """Latency stress: enter at the open of the first bar after the signal bar, exits unchanged.

    A trade is dropped when there is no next bar, the next bar is a data gap, or the new entry
    is already at or beyond its stop or target.
    """
    seqs, opens, gaps = bars.array["seq"], bars.array["open"], bars.array["data_gap"]
    keep, entries = [], []
    for position, trade in enumerate(trades.itertuples(index=False)):
        i = int(np.searchsorted(seqs, trade.entry_seq, side="right"))
        if i >= len(seqs) or gaps[i]:
            continue
        entry, long = float(opens[i]), trade.side == "buy"
        inside = trade.stop_loss < entry < trade.take_profit if long else trade.take_profit < entry < trade.stop_loss
        if inside:
            keep.append(position)
            entries.append(entry)
    out = trades.iloc[keep].copy()
    out["entry_price"] = entries
    out = add_gross(reprice_costs(out, fee_bps, slippage_bps)).reset_index(drop=True)
    return out, len(trades) - len(out)


def stress_summary(trades, bars, fee_bps, slippage_bps):
    delayed, dropped = next_bar_entry(trades, bars, fee_bps, slippage_bps)
    return {"cost_grid": cost_grid(trades).to_dict("records"),
            "next_bar_entry": {"dropped": dropped, "n_remaining": len(delayed),
                               "decomposition": decompose(delayed)}}


def find_cached_bars(cache_dir, data_fp):
    """Locate the cache whose manifest fingerprint is the study's data_fp (never opens the source parquet)."""
    for manifest in sorted(Path(cache_dir).glob("bars-*.json")):
        meta = json.loads(manifest.read_text())
        if meta.get("fingerprint") == data_fp:
            return manifest.with_suffix(".bin"), meta
    raise ValueError(f"no cached bars with fingerprint {data_fp} in {cache_dir}; "
                     "run the optimizer once to build the cache")


def replay_and_verify(path, meta, params, settings, expected):
    """Replay the cached development window with the study's own settings and require its stored digest."""
    if expected is None:
        verify_replay(None, None)
    bars = opt.CachedBars(path, meta["bar_type"])
    replay_args = argparse.Namespace(symbol=meta["symbol"], **settings)
    n_search, folds = opt.split_timeline(bars, replay_args)
    trades = replay_trades(bars, params, replay_args)
    verify_replay(trades.attrs["replay_digest"], expected)
    return bars, trades, n_search, folds


def build_parser():
    out = Path("output")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-db", type=Path, default=out / "optimize/studies.db")
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--trial", type=int)
    parser.add_argument("--cache-dir", type=Path, default=out / "optimize/cache")
    parser.add_argument("--output-dir", type=Path, default=out / "diagnose")
    parser.add_argument("--floors", action="store_true")
    parser.add_argument("--stress", action="store_true", help="cost grid and next-bar-entry latency stress")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    identity = load_identity(args.study_db, args.study_name)
    settings = bind_replay_settings(identity)
    path, meta = find_cached_bars(args.cache_dir, identity["data_fp"])
    start, end = date.fromisoformat(meta["start"]), date.fromisoformat(meta["end"])
    ensure_dev_window(start, end)
    params, number = load_trial(args.study_db, args.study_name, args.trial)
    baseline = opt.baseline_extra_params(meta["symbol"])
    filled = sorted(set(baseline) - set(params))       # older studies lack the later dimensions
    params = {**baseline, **params}
    expected = stored_digest(args.study_db, args.study_name, number)
    bars, trades, n_search, folds = replay_and_verify(path, meta, params, settings, expected)
    scored = scored_only(trades, bars, folds)
    report = {"study": args.study_name, "trial": number, "params": params,
              "replay_digest": trades.attrs["replay_digest"], "stored_digest": expected, "digest_matches": True,
              "baseline_filled_dims": filled,
              "replay_settings": {**settings, "data_fp": identity["data_fp"], "start": meta["start"],
                                  "end": meta["end"], "n_search": n_search},
              "decomposition_all_closed": decompose(trades), "decomposition": decompose(scored)}
    if args.stress:
        report["stress"] = stress_summary(scored, bars, settings["fee_bps"], settings["slippage_bps"])
    if args.floors:
        report["floors"] = {str(m): floors(report["decomposition"], settings["n_splits"], m)
                            for m in (0.5, 1.0, 2.0)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    opt.atomic_json(args.output_dir / f"{args.study_name}__trial{number}.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(opt.json_safe(main()), indent=2))

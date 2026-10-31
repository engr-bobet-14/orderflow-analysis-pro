"""Selection stability from stored Optuna trials: an expanding-window walk-forward with no replay.

For each k, the best trial on folds < k (by day-weighted Sharpe, among trials feasible in all of
them) is scored on fold k. Reads the SQLite study only; never opens market data.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts import optimize_orderflow as opt

NOTE = ("selection stability only: TPE proposed trials after seeing all folds, so the candidate pool is "
        "not independent of later folds; not an out-of-sample estimate")


def load_trial_folds(study_db, study_name):
    con = sqlite3.connect(f"file:{Path(study_db)}?mode=ro", uri=True)
    try:
        rows = con.execute("""select t.number, a.value_json from trials t
            join studies s on s.study_id = t.study_id
            join trial_user_attributes a on a.trial_id = t.trial_id and a.key = 'fold_results'
            where s.study_name = ? and t.state = 'COMPLETE' order by t.number""", (study_name,)).fetchall()
    finally:
        con.close()
    out = []
    for number, raw in rows:
        folds = json.loads(raw)
        out.append({"number": number,
                    "sharpe": [math.nan if f.get("sharpe_ratio") is None else float(f["sharpe_ratio"]) for f in folds],
                    "n_trades": [int(f["n_trades"]) for f in folds],
                    "blown": [bool(f["account_blown"]) for f in folds],
                    "weight": [f.get("n_days", f.get("n_scored_bars", 1)) for f in folds]})
    return out


def _feasible(trial, fold, min_trades):
    return (trial["n_trades"][fold] >= min_trades and not trial["blown"][fold]
            and math.isfinite(trial["sharpe"][fold]))


def _finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def selection_walk_forward(trials, min_trades):
    n_splits = max((len(t["sharpe"]) for t in trials), default=0)
    rows = []
    for k in range(1, n_splits):
        candidates = [t for t in trials if all(_feasible(t, f, min_trades) for f in range(k))]
        if not candidates:
            rows.append({"test_fold": k + 1, "skipped": "no eligible trial"})
            continue
        scores = {t["number"]: float(np.average(t["sharpe"][:k], weights=t["weight"][:k])) for t in candidates}
        picked = max(candidates, key=lambda t: (scores[t["number"]], -t["number"]))
        feasible_now = [t["sharpe"][k] for t in trials if _feasible(t, k, min_trades)]
        test = picked["sharpe"][k]
        rows.append({"test_fold": k + 1, "selected_trial": picked["number"],
                     "train_score": scores[picked["number"]], "test_sharpe": _finite(test),
                     "test_n_trades": picked["n_trades"][k], "test_feasible": _feasible(picked, k, min_trades),
                     "n_candidates": len(candidates),
                     "test_percentile": (sum(s < test for s in feasible_now) / len(feasible_now)
                                         if _feasible(picked, k, min_trades) else None),
                     "median_test_sharpe": float(np.median(feasible_now)) if feasible_now else None})
    return rows


def rank_stability(trials):
    n_splits = max((len(t["sharpe"]) for t in trials), default=0)
    out = {}
    for i in range(n_splits):
        for j in range(i + 1, n_splits):
            pairs = [(t["sharpe"][i], t["sharpe"][j]) for t in trials
                     if math.isfinite(t["sharpe"][i]) and math.isfinite(t["sharpe"][j])]
            if len(pairs) < 3:
                out[f"{i + 1}-{j + 1}"] = None
                continue
            first, second = (pd.Series(v).rank() for v in zip(*pairs))
            with np.errstate(invalid="ignore", divide="ignore"):         # a constant fold has no rank correlation
                out[f"{i + 1}-{j + 1}"] = _finite(np.corrcoef(first, second)[0, 1])
    return out


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-db", type=Path, default=Path("output/optimize/studies.db"))
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--min-trades", type=int, default=2, help="per-fold trade floor for eligibility")
    parser.add_argument("--output-dir", type=Path, default=Path("output/diagnose"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    trials = load_trial_folds(args.study_db, args.study_name)
    if not trials:
        raise ValueError(f"study {args.study_name} has no completed trials with fold results")
    report = {"study": args.study_name, "min_trades": args.min_trades, "n_trials": len(trials), "note": NOTE,
              "rows": selection_walk_forward(trials, args.min_trades), "rank_stability": rank_stability(trials)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    opt.atomic_json(args.output_dir / f"walk_forward_{args.study_name}.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(opt.json_safe(main()), indent=2))

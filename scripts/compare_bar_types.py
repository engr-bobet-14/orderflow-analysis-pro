"""Rank optimizer studies by bar type. Eligibility is recomputed from the stored folds, never trusted."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

COLUMNS = ["bar_type", "study_name", "best_objective", "total_trades", "cv_sharpe_std",
           "n_penalized_folds", "best_degenerate", "eligible"]


def folds_eligible(folds, n_splits, min_trades_per_fold):
    return (len(folds) == n_splits
            and all(f["n_trades"] >= min_trades_per_fold and not f["account_blown"]
                    and f.get("sharpe_ratio") is not None and "sharpe_ratio_status" not in f for f in folds))


def rank(results, n_splits, min_trades_per_fold, min_total_trades):
    rows = []
    for studies in results["symbols"].values():
        for bar_type, by_name in studies.items():
            for name, entry in by_name.items():
                attrs = entry.get("best_trial_attrs") or {}
                folds = attrs.get("fold_results") or []
                total = entry.get("total_trades", 0)
                rows.append({"bar_type": bar_type, "study_name": name,
                             "best_objective": entry.get("best_objective"), "total_trades": total,
                             "cv_sharpe_std": attrs.get("cv_sharpe_std"),
                             "n_penalized_folds": attrs.get("n_penalized_folds"),
                             "best_degenerate": bool(entry.get("best_degenerate")),
                             "eligible": bool(folds) and folds_eligible(folds, n_splits, min_trades_per_fold)
                             and total >= min_total_trades})
    table = pd.DataFrame(rows, columns=COLUMNS)
    return table.sort_values("best_objective", ascending=False, na_position="last", kind="stable").reset_index(drop=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("output/optimize/results.json"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-trades-per-fold", type=int, required=True)
    parser.add_argument("--min-total-trades", type=int, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    table = rank(json.loads(args.results.read_text()), args.n_splits, args.min_trades_per_fold, args.min_total_trades)
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    main()

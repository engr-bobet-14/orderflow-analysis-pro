"""Ranking of studies by bar type with fold-level eligibility."""
import json

import pandas as pd
import pytest

from scripts import compare_bar_types as cmp


def fold(n_trades, sharpe=0.5, blown=False, status=None):
    row = {"n_trades": n_trades, "account_blown": blown, "sharpe_ratio": sharpe}
    if status:
        row.update(sharpe_ratio=None, sharpe_ratio_status=status)
    return row


def entry(folds, objective=-1.0, name="study"):
    attrs = {"fold_results": folds, "cv_sharpe_std": 0.2, "n_penalized_folds": 0} if folds is not None else {}
    return {"study_name": name, "best_objective": objective if folds is not None else None,
            "best_trial_attrs": attrs, "total_trades": sum(f["n_trades"] for f in folds) if folds else 0,
            "best_degenerate": folds is None}


def results_of(**entries):
    return {"symbols": {"BTCUSDT": {bar: {e["study_name"]: e} for bar, e in entries.items()}}}


CASES = {
    "empty_fold": ([fold(0)] + [fold(38)] * 4, False),             # 152 pooled trades, one empty fold
    "bankrupt_fold": ([fold(38, blown=True)] + [fold(38)] * 4, False),
    "non_finite_fold": ([fold(38, status="non_finite")] + [fold(38)] * 4, False),
    "valid": ([fold(30)] * 5, True),
    "missing_fold": ([fold(40)] * 4, False),
}


@pytest.mark.parametrize("name", list(CASES))
def test_fold_level_eligibility(name):
    folds, expected = CASES[name]
    table = cmp.rank(results_of(fixed_tick_imbalance=entry(folds, name=name)), n_splits=5,
                     min_trades_per_fold=30, min_total_trades=150)
    assert bool(table.iloc[0]["eligible"]) is expected


def test_pooled_floor_and_per_fold_floor_both_apply():
    short = entry([fold(30)] * 5)                                      # 150 pooled, ok
    table = cmp.rank(results_of(a=short), 5, 30, 151)
    assert bool(table.iloc[0]["eligible"]) is False
    assert bool(cmp.rank(results_of(a=short), 5, 31, 100).iloc[0]["eligible"]) is False


def test_degenerate_study_is_ineligible_and_sorted_last():
    results = results_of(a=entry([fold(40)] * 5, objective=-2.0, name="a"), b=entry(None, name="b"),
                         c=entry([fold(40)] * 5, objective=-1.0, name="c"))
    table = cmp.rank(results, 5, 30, 150)
    assert list(table.study_name) == ["c", "a", "b"]
    assert list(table.eligible) == [True, True, False] and bool(table.iloc[2]["best_degenerate"])


def test_stored_feasible_flag_is_not_trusted():
    bad = entry([fold(0)] + [fold(40)] * 4)
    bad["best_trial_attrs"]["feasible"] = True                         # stale or wrong flag
    assert bool(cmp.rank(results_of(a=bad), 5, 30, 150).iloc[0]["eligible"]) is False


def test_main_prints_table_from_results_json(tmp_path, capsys):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results_of(a=entry([fold(40)] * 5, name="a"))))
    table = cmp.main(["--results", str(path), "--n-splits", "5", "--min-trades-per-fold", "30",
                      "--min-total-trades", "150"])
    assert isinstance(table, pd.DataFrame) and "a" in capsys.readouterr().out


def test_floors_must_be_explicit():
    with pytest.raises(SystemExit):
        cmp.build_parser().parse_args(["--results", "x.json"])

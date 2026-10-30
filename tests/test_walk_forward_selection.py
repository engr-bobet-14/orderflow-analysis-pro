"""Selection stability from stored trials (no replay, no market data)."""
import json
import math

import numpy as np
import optuna
import pytest

from scripts import optimize_orderflow as opt
from scripts import walk_forward_selection as wf


def trial(number, sharpe, n_trades=None, blown=None, weight=None):
    n = len(sharpe)
    return {"number": number, "sharpe": list(sharpe), "n_trades": n_trades or [10] * n,
            "blown": blown or [False] * n, "weight": weight or [100] * n}


def test_selection_uses_only_earlier_folds():
    a = trial(0, [1.0, 1.0, -5.0, 0, 0])
    b = trial(1, [0.5, 0.5, 3.0, 0, 0])
    rows = wf.selection_walk_forward([a, b], min_trades=2)
    third = next(r for r in rows if r["test_fold"] == 3)
    assert third["selected_trial"] == 0 and third["test_sharpe"] == -5.0
    assert third["train_score"] == pytest.approx(1.0) and third["n_candidates"] == 2
    a["sharpe"][2], b["sharpe"][2] = 9.0, -9.0                      # change the test fold only
    again = next(r for r in wf.selection_walk_forward([a, b], min_trades=2) if r["test_fold"] == 3)
    assert again["selected_trial"] == 0 and again["test_sharpe"] == 9.0


def test_train_score_is_weight_averaged():
    a = trial(0, [1.0, 3.0, 0, 0, 0], weight=[1, 3, 1, 1, 1])
    row = next(r for r in wf.selection_walk_forward([a], min_trades=2) if r["test_fold"] == 3)
    assert row["train_score"] == pytest.approx((1.0 * 1 + 3.0 * 3) / 4)


def test_percentile_and_median():
    trials = [trial(i, [float(i), float(i + 1), 0, 0, 0]) for i in range(1, 5)]   # fold 2 Sharpes 2,3,4,5
    row = next(r for r in wf.selection_walk_forward(trials, min_trades=2) if r["test_fold"] == 2)
    assert row["selected_trial"] == 4 and row["test_sharpe"] == 5.0
    assert row["test_percentile"] == pytest.approx(3 / 4)                         # 2, 3, 4 are below 5
    assert row["median_test_sharpe"] == pytest.approx(3.5)


def test_skips_when_no_eligible_trial():
    starved = [trial(0, [1.0] * 5, n_trades=[1, 10, 10, 10, 10]),
               trial(1, [1.0] * 5, blown=[True, False, False, False, False]),
               trial(2, [float("nan")] + [1.0] * 4)]
    rows = wf.selection_walk_forward(starved, min_trades=2)
    assert all(r.get("skipped") == "no eligible trial" for r in rows) and len(rows) == 4


def test_test_fold_infeasibility_gives_no_percentile():
    t = trial(0, [1.0, 1.0, 1.0, 1.0, 1.0], n_trades=[10, 10, 1, 10, 10])
    row = next(r for r in wf.selection_walk_forward([t], min_trades=2) if r["test_fold"] == 3)
    assert row["test_percentile"] is None and row["test_n_trades"] == 1 and row["test_feasible"] is False
    ok = trial(0, [1.0] * 5)
    assert next(r for r in wf.selection_walk_forward([ok], min_trades=2) if r["test_fold"] == 3)["test_feasible"] is True


def test_rank_stability_perfect_and_inverse():
    base = [trial(i, [float(i), float(i), float(10 - i), 0, 0]) for i in range(6)]
    stability = wf.rank_stability(base)
    assert stability["1-2"] == pytest.approx(1.0) and stability["1-3"] == pytest.approx(-1.0)


def make_study(tmp_path, folds_per_trial):
    db = tmp_path / "s.db"
    study = optuna.create_study(study_name="s", storage=f"sqlite:///{db}", direction="maximize")
    for folds in folds_per_trial:
        study.add_trial(optuna.trial.create_trial(value=-1.0, user_attrs={"fold_results": folds}))
    return db


def fold_row(n_trades, sharpe, blown=False, **extra):
    row = {"n_trades": n_trades, "account_blown": blown, **extra}
    if sharpe is None:
        row.update(sharpe_ratio=None, sharpe_ratio_status="non_finite")
    else:
        row["sharpe_ratio"] = sharpe
    return row


def test_load_trial_folds_weights_and_non_finite(tmp_path):
    db = make_study(tmp_path, [[fold_row(5, 1.0, n_days=30), fold_row(6, None, n_scored_bars=900),
                                fold_row(7, 2.0, blown=True, n_days=10)]])
    (loaded,) = wf.load_trial_folds(db, "s")
    assert loaded["n_trades"] == [5, 6, 7] and loaded["blown"] == [False, False, True]
    assert loaded["weight"] == [30, 900, 10]                       # n_days, else the legacy n_scored_bars
    assert loaded["sharpe"][0] == 1.0 and math.isnan(loaded["sharpe"][1])


def test_main_refuses_empty_study(tmp_path):
    db = make_study(tmp_path, [])
    with pytest.raises(ValueError, match="no completed trials"):
        wf.main(["--study-db", str(db), "--study-name", "s", "--output-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_reads_no_market_data(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("market data accessed by the stored-trial walk-forward")
    monkeypatch.setattr(opt, "build_cache", forbidden)
    monkeypatch.setattr(opt, "open_bars", forbidden)
    monkeypatch.setattr(opt, "CachedBars", forbidden)
    folds = [fold_row(10, s, n_days=100) for s in (1.0, 0.5, 0.2, 0.1, 0.0)]
    db = make_study(tmp_path, [folds, [fold_row(10, s + 1, n_days=100) for s in (1.0, 0.5, 0.2, 0.1, 0.0)]])
    report = wf.main(["--study-db", str(db), "--study-name", "s", "--min-trades", "2",
                      "--output-dir", str(tmp_path / "out")])
    assert report["rows"] and "not an out-of-sample estimate" in report["note"]
    written = json.loads(next((tmp_path / "out").glob("walk_forward_*.json")).read_text())
    assert written["study"] == "s"


def test_rank_stability_with_a_constant_column_is_quiet_and_none():
    import warnings
    trials = [trial(i, [float(i), 0.0, 1.0, 0, 0]) for i in range(6)]            # folds 2-5 are constant
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stability = wf.rank_stability(trials)
    assert stability["1-2"] is None and stability["2-3"] is None

"""Search-space boundary audit and read-only trial reader (no replay, no market data)."""
import hashlib
import json
from pathlib import Path

import optuna
import pytest

from scripts import optimize_orderflow as opt
from scripts import search_space_audit as audit

ROOT = Path(__file__).resolve().parents[1]


def test_boundary_hits_flags_top_and_tail_pressure():
    space = {"a": (0.0, 1.0, "float"), "b": (0.001, 0.1, "float_log"), "c": (1, 30, "int")}
    trials = [{"number": i, "state": "COMPLETE", "value": float(i), "feasible": True,
               "params": {"a": 0.99, "b": 0.0011, "c": 15}} for i in range(60)]
    hits = audit.boundary_hits(trials, space)
    assert hits["a"]["widen_high"] and not hits["a"]["widen_low"]
    assert hits["b"]["widen_low"] and not hits["b"]["widen_high"]      # log-scale position
    assert not hits["c"]["widen_low"] and not hits["c"]["widen_high"]


def test_fixed_dimensions_are_reported_without_raising():
    # load_space accepts low == high; such a dimension has no edge to press against
    space = {"a": (0.5, 0.5, "float"), "b": (0.01, 0.01, "float_log"), "c": (5, 5, "int")}
    trials = [{"number": i, "state": "COMPLETE", "value": float(i), "feasible": True,
               "params": {"a": 0.5, "b": 0.01, "c": 5}} for i in range(60)]
    zero = {"top_low": 0, "top_high": 0, "tail_low": 0.0, "tail_high": 0.0, "widen_low": False, "widen_high": False}
    hits = audit.boundary_hits(trials, space)
    assert hits == {name: zero for name in space}
    assert "-" in audit.format_table(hits, space)


def test_boundary_hits_ignores_infeasible_for_top_but_not_tail():
    # infeasible trials with huge values never count toward top_k; tail counts every COMPLETE trial
    space = {"a": (0.0, 1.0, "float")}
    feasible = [{"number": i, "state": "COMPLETE", "value": float(i), "feasible": True,
                 "params": {"a": 0.5}} for i in range(10)]
    infeasible = [{"number": 10 + i, "state": "COMPLETE", "value": 1e9 + i, "feasible": False,
                   "params": {"a": 0.99}} for i in range(40)]
    hits = audit.boundary_hits(feasible + infeasible, space)["a"]
    assert hits["top_high"] == 0 and hits["top_low"] == 0
    assert hits["tail_high"] == pytest.approx(40 / 50)
    assert hits["widen_high"] and not hits["widen_low"]
    # tail window is the last 50 COMPLETE trials by number; non-COMPLETE trials are skipped
    extra = [{"number": 99, "state": "FAIL", "value": None, "feasible": False, "params": {"a": 0.0}}]
    assert audit.boundary_hits(feasible + infeasible + extra, space)["a"]["tail_low"] == hits["tail_low"]


def _make_study(path):
    url = f"sqlite:///{path}"
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(study_name="s1", storage=url, direction="maximize")
    study.set_user_attr("identity", {"search_space": {"lookback_bars": [5, 30, "int"]}})
    for i, feasible in enumerate([True, False, True]):
        t = optuna.trial.create_trial(
            params={"lookback_bars": 10 + i, "stop": 0.002 * (i + 1), "score": 40.0 + i},
            distributions={"lookback_bars": optuna.distributions.IntDistribution(5, 30),
                           "stop": optuna.distributions.FloatDistribution(0.001, 0.01, log=True),
                           "score": optuna.distributions.FloatDistribution(10.0, 70.0)},
            value=float(i), user_attrs={"feasible": feasible})
        study.add_trial(t)
    return url


def test_read_study_trials_roundtrip_and_readonly(tmp_path):
    db = tmp_path / "s.db"
    _make_study(db)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    trials = opt.read_study_trials(db, "s1")
    assert [t["number"] for t in trials] == [0, 1, 2]
    assert [t["feasible"] for t in trials] == [True, False, True]
    assert [t["state"] for t in trials] == ["COMPLETE"] * 3
    assert isinstance(trials[0]["params"]["lookback_bars"], int)
    assert trials[1]["params"]["stop"] == pytest.approx(0.004)
    assert trials[2]["value"] == 2.0
    assert trials[0]["attrs"]["feasible"] is True
    assert opt.read_study_attrs(db, "s1")["identity"]["search_space"]["lookback_bars"] == [5, 30, "int"]
    assert opt.list_study_names(db) == ["s1"]
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_read_study_trials_unknown_study_raises(tmp_path):
    db = tmp_path / "s.db"
    _make_study(db)
    with pytest.raises(ValueError, match="study not found: nope"):
        opt.read_study_trials(db, "nope")
    with pytest.raises(ValueError, match="study not found: nope"):
        opt.read_study_attrs(db, "nope")


SPACES = ROOT / "scripts/search_spaces"
EXPECTED_WIDENED = {
    "fixed_tick_imbalance": {"default_stop_pct", "price_proximity_pct", "signal_cooldown_seconds",
                             "delta_failure_pct", "target_rr"},
    "fixed_volume_imbalance": {"default_stop_pct", "price_proximity_pct",
                               "absorption_min_delta_pct", "initiative_min_delta_pct"},
}


@pytest.mark.parametrize("bar_type", EXPECTED_WIDENED)
def test_search_space_file_widens_only_justified_dimensions_and_is_a_superset(bar_type):
    space = opt.load_space(SPACES / f"{bar_type}.json")
    changed = {n for n, b in space.items() if b != opt.SEARCH_SPACE_BOUNDS[n]}
    assert changed == EXPECTED_WIDENED[bar_type]
    for n, (lo, hi, kind) in space.items():
        dlo, dhi, dkind = opt.SEARCH_SPACE_BOUNDS[n]
        assert lo <= dlo and hi >= dhi and kind == dkind


@pytest.mark.parametrize("dim", ["absorption_min_delta_pct", "initiative_min_delta_pct"])
def test_load_space_rejects_delta_fraction_above_one(tmp_path, dim):
    path = tmp_path / "s.json"; path.write_text(json.dumps({dim: {"low": 0.2, "high": 1.5}}))
    with pytest.raises(ValueError, match="fraction"):
        opt.load_space(path)

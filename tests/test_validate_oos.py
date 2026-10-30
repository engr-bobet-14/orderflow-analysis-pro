"""Frozen-parameter evaluation and 2024-2025 validation. Synthetic data only: no market data is read."""
import json
from datetime import date
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import pytest

from orderflow_system.data.models import Side
from scripts import dataset_policy as policy
from scripts import frozen_eval as fe
from scripts import optimize_orderflow as opt
from scripts import validate_oos as vo


def ms(text):
    return int(pd.Timestamp(text, tz="UTC").timestamp() * 1000)


def write_cache(tmp_path, stamps):
    """A tiny CachedBars file: bar i has seq 10*(i+1), the given close time and a flat price."""
    array = np.zeros(len(stamps), dtype=opt.BAR_DTYPE)
    array["timestamp_ms"] = [ms(t) for t in stamps]
    array["seq"] = [10 * (i + 1) for i in range(len(stamps))]
    for column in ("open", "high", "low", "close"):
        array[column] = 100.0
    array["duration_ms"] = -1
    path = tmp_path / "bars-synthetic.bin"
    array.tofile(path)
    return path


def execution_pair(i, entry_seq, exit_seq, entry=100.0, exit_price=101.0, stop=99.5):
    opened = SimpleNamespace(trade_id=str(i), bar_seq=entry_seq, timestamp_ms=1_700_000_000_000 + entry_seq,
                             direction=Side.BUY, fill_price=entry, stop_loss=stop, take_profit=105.0)
    closed = None if exit_seq is None else SimpleNamespace(
        trade_id=str(i), bar_seq=exit_seq, timestamp_ms=1_700_000_000_000 + exit_seq,
        fill_price=exit_price, exit_reason="target", pnl=exit_price - entry)
    return opened, closed


def fake_ledger(specs):
    entries, closes = [], []
    for i, (entry_seq, exit_seq) in enumerate(specs):
        opened, closed = execution_pair(i, entry_seq, exit_seq)
        entries.append(SimpleNamespace(execution=opened))
        if closed is not None:
            closes.append(SimpleNamespace(execution=closed))
    return SimpleNamespace(entries=entries, closes=closes)


STAMPS = ([f"2023-12-{day:02d}T12:00:00" for day in range(20, 30)]            # 10 warm-up bars, seq 10..100
          + [f"2024-01-{day:02d}T12:00:00" for day in range(2, 12)])          # 10 window bars,  seq 110..200


@pytest.fixture
def window(tmp_path, monkeypatch):
    path = write_cache(tmp_path, STAMPS)
    monkeypatch.setattr(policy, "VALIDATION_START", date(2024, 1, 1))
    monkeypatch.setattr(policy, "VALIDATION_END", date(2024, 1, 31))
    monkeypatch.setattr(opt, "build_cache", lambda args, symbol, bar_type: (path, {"fingerprint": "x", **DATASET}))
    monkeypatch.setattr(fe, "find_dataset", lambda root, symbol, bar_type: SimpleNamespace(
        dataset_id=DATASET["source"], sidecar={**DATASET["source_metadata"]}))
    kwargs = dict(symbol="BTCUSDT", bar_type="fixed_tick_imbalance", start=date(2024, 1, 1), end=date(2024, 1, 31),
                  settings={"fee_bps": 5.0, "slippage_bps": 1.0}, data_root=tmp_path, cache_dir=tmp_path,
                  guard=policy.ensure_validation_window, expected_dataset=dict(DATASET))
    return kwargs


PARAMS = {name: low for name, (low, _, _) in opt.SEARCH_SPACE_BOUNDS.items()}
DATASET = {"source": "src-A", "data_proc_fp": "dp-1",
           "source_metadata": {"hyperparams": {"e0_mult": 0.33}, "config": {"w": 441}, "created_utc": "2026-10-26T20:48:20+00:00",
                               "git_sha": "abc"}}


@pytest.mark.parametrize("start, end", [(date(2023, 12, 31), date(2024, 6, 1)), (date(2024, 1, 1), date(2026, 1, 1)),
                                        (date(2026, 1, 1), date(2026, 6, 1))])
def test_evaluate_frozen_refuses_outside_policy_before_data(start, end, monkeypatch, tmp_path):
    def forbidden(*a, **k):
        raise AssertionError("data accessed before the validation-window guard")
    monkeypatch.setattr(opt, "build_cache", forbidden)
    with pytest.raises(ValueError, match="validation window must lie within 2024-01-01..2025-12-31"):
        fe.evaluate_frozen(PARAMS, symbol="BTCUSDT", bar_type="fixed_tick_imbalance", start=start, end=end,
                           settings={"fee_bps": 5.0, "slippage_bps": 1.0}, data_root=tmp_path, cache_dir=tmp_path,
                           guard=policy.ensure_validation_window, expected_dataset=dict(DATASET))


def test_warmup_entries_excluded_and_open_position_reported(window, monkeypatch):
    # T0 enters in warm-up (seq 50) and exits inside the window (seq 120): excluded.
    # T1 enters and exits inside the window: scored. T2 enters on the last bar and never closes.
    specs = [(50, 120), (130, 150), (200, None)]
    monkeypatch.setattr(opt, "run_backtest", lambda bars, *a, **k: {"ledger": fake_ledger(specs), "replay_digest": "d"})
    result = fe.evaluate_frozen(PARAMS, **window)
    assert result["metrics"]["n_trades"] == 1 and result["n_open_at_end"] == 1 and result["n_warmup_bars"] == 10
    alone = opt.simulate_trades_risk_based(fake_ledger([(130, 150)]))
    assert result["metrics"]["total_pnl"] == pytest.approx(float(alone["net_pnl"].sum()))   # sizing anchored at $5000
    assert result["window"] == {"start": "2024-01-01", "end": "2024-01-31"}


def test_replay_stops_at_the_window_end(window, monkeypatch):
    seen = {}
    monkeypatch.setattr(opt, "run_backtest", lambda bars, *a, **k: seen.setdefault("n", len(bars)) and
                        {"ledger": fake_ledger([]), "replay_digest": "d"})
    path_bars = len(STAMPS)
    window["end"] = date(2024, 1, 5)                                  # closes up to 2024-01-05T12:00 -> 4 window bars
    fe.evaluate_frozen(PARAMS, **window)
    assert seen["n"] == 14 < path_bars


def test_empty_window_is_refused(window, monkeypatch):
    window["start"], window["end"] = date(2024, 2, 1), date(2024, 2, 5)
    monkeypatch.setattr(policy, "VALIDATION_END", date(2024, 2, 28))
    with pytest.raises(ValueError, match="no bars in the window"):
        fe.evaluate_frozen(PARAMS, **window)


# ---- D10: confirmation ----------------------------------------------------------------------------

def candidate(expectancy, se, n=100, trade_r=(), blown=False):
    return {"decomposition": {"net_expectancy_r": expectancy, "se_r": se, "n_trades": n}, "trade_r": list(trade_r),
            "metrics": {"account_blown": blown}}


def test_confirmation_bonferroni():
    pair = vo.confirmation([candidate(0.30, 0.10), candidate(0.18, 0.10)])
    z2 = NormalDist().inv_cdf(1 - 0.05 / 2)
    assert pair[0]["z"] == pytest.approx(z2) and pair[0]["lower_bound"] == pytest.approx(0.30 - z2 * 0.10)
    assert [c["confirmed"] for c in pair] == [True, False]
    alone = vo.confirmation([candidate(0.18, 0.10)])
    assert alone[0]["z"] == pytest.approx(NormalDist().inv_cdf(0.95)) and alone[0]["confirmed"] is True


def test_confirmation_inflates_se_for_positive_serial_correlation():
    clustered = [1.0] * 6 + [-1.0] * 6 + [1.0] * 6 + [-1.0] * 6        # strong positive lag-1 autocorrelation
    plain = candidate(0.30, 0.16, trade_r=[0.3, -0.1] * 12)             # negative autocorrelation: no inflation
    wide = candidate(0.30, 0.16, trade_r=clustered)
    a, b = vo.confirmation([plain])[0], vo.confirmation([wide])[0]
    assert a["rho1"] < 0 and a["se_r_inflated"] == pytest.approx(0.16)
    assert b["rho1"] > 0.5 and b["se_r_inflated"] > 0.16 * 1.5
    assert a["confirmed"] is True and b["confirmed"] is False           # 0.30 - 1.645 * se: passes plain, fails inflated


def test_confirmation_needs_a_standard_error():
    assert vo.confirmation([candidate(0.5, None, n=1)])[0]["confirmed"] is False
    assert vo.confirmation([candidate(None, None, n=0)])[0]["confirmed"] is False


# ---- orchestration ---------------------------------------------------------------------------------

CURRENT_DEFAULTS = opt.json_safe(__import__("dataclasses").asdict(opt.instrument_config("BTCUSDT")))
IDENTITY = {"strategy_fp": opt.STRATEGY_FP, "instrument_defaults": CURRENT_DEFAULTS, "data_fp": "fp-1", "n_splits": 5, "embargo": 0.02, "holdout_fraction": 0.2, "fee_bps": 5.0,
            "slippage_bps": 1.0, "initial_capital_usd": opt.INITIAL_CAPITAL_USD,
            "risk_per_trade_pct": opt.RISK_PER_TRADE_PCT, "min_stop_loss_pct": opt.MIN_STOP_LOSS_PCT,
            "annualization_periods": opt.ANNUALIZATION_PERIODS}


def make_study(tmp_path, *, feasible=True, identity=None, versions=None, digest="D"):
    db = tmp_path / "s.db"
    study = optuna.create_study(study_name="s", storage=f"sqlite:///{db}", direction="maximize")
    study.set_user_attr("identity", identity or IDENTITY)
    study.set_user_attr("versions", versions or opt.library_versions())
    study.add_trial(optuna.trial.create_trial(value=-1.0, user_attrs={
        "params": {"min_composite_score": 50.0}, "feasible": feasible,
        **({"replay_digest": digest} if digest else {})}))
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    (cache / "bars-x.json").write_text(json.dumps({"fingerprint": "fp-1", "start": "2019-09-10", "end": "2023-12-31",
                                                    "symbol": "BTCUSDT", "bar_type": "fixed_tick_imbalance", **DATASET}))
    return db, cache


def stub_dev_replay(monkeypatch, digest="D"):
    """The development-window replay: stubbed here, verified for real in the diagnose tests."""
    from scripts import diagnose_trades as dx
    table = pd.DataFrame()
    table.attrs["replay_digest"] = digest
    monkeypatch.setattr(opt, "CachedBars", lambda path, bar_type: SimpleNamespace(array=None))
    monkeypatch.setattr(opt, "split_timeline", lambda bars, args: (10, []))
    monkeypatch.setattr(dx, "replay_trades", lambda bars, params, args: table)


def fake_result(expectancy, se=0.05):
    return {"decomposition": {"net_expectancy_r": expectancy, "se_r": se, "n_trades": 100}, "trade_r": [],
            "metrics": {"n_trades": 100, "account_blown": False}, "daily_returns": [0.0, 0.01, -0.01, 0.02]}


def test_validate_refuses_study_without_feasible_trial(tmp_path):
    db, cache = make_study(tmp_path, feasible=False)
    with pytest.raises(ValueError, match="no feasible trial"):
        vo.validate(db, "s", None, cache_dir=cache, data_root=tmp_path)


def prepared(name="s", trial=0):
    return {"study": name, "trial": trial, "params": {"min_composite_score": 50.0, "x": trial}}


def run_main(tmp_path, monkeypatch, candidates, results, *, extra=()):
    """Drive main() with stubbed preflight/evaluation; returns (summary, evaluated candidates)."""
    db, cache = make_study(tmp_path)
    out = tmp_path / "out"
    evaluated = []

    def fake_preflight(study_db, name, trial, **kw):
        if name == "missing":
            raise ValueError("study not found: missing")
        return prepared(name, 0 if trial is None else trial)

    def fake_evaluate(prep, **kw):
        manifests = list(out.glob("campaign-*.json"))
        assert manifests, "validation data read before the campaign manifest was frozen"
        evaluated.append(prep["study"])
        value = results[prep["study"]]
        return {**(value if isinstance(value, dict) else fake_result(value)), "study": prep["study"], "trial": prep["trial"]}
    monkeypatch.setattr(vo, "preflight_candidate", fake_preflight)
    monkeypatch.setattr(vo, "evaluate_candidate", fake_evaluate)
    argv = ["--study-db", str(db), "--candidates", candidates, "--cache-dir", str(cache),
            "--output-dir", str(out), *extra]
    return argv, evaluated, out


def test_no_confirmed_candidate_selects_nothing(tmp_path, monkeypatch, capsys):
    argv, evaluated, _ = run_main(tmp_path, monkeypatch, "a:0", {"a": 0.02})
    summary = vo.main(argv)
    assert summary["selected"] is None and "no candidate confirmed" in capsys.readouterr().out


def test_highest_expectancy_among_confirmed_is_selected(tmp_path, monkeypatch):
    results = {"a": 0.30, "b": 0.45, "c": fake_result(0.90, se=0.9)}                    # c is unconfirmed
    argv, evaluated, _ = run_main(tmp_path, monkeypatch, "a:0,b:0,c:0", results)
    summary = vo.main(argv)
    assert summary["selected"]["study"] == "b" and evaluated == ["a", "b", "c"]


def test_manifest_is_frozen_before_any_validation_data_is_read(tmp_path, monkeypatch):
    argv, evaluated, out = run_main(tmp_path, monkeypatch, "a:0,b:0", {"a": 0.4, "b": 0.5})
    vo.main(argv)                                                                       # fake_evaluate asserts the manifest exists
    (manifest_path,) = out.glob("campaign-*.json")
    manifest = json.loads(manifest_path.read_text())
    assert [(c["study"], c["trial"], c["status"]) for c in manifest["candidates"]] == [("a", 0, "done"), ("b", 0, "done")]
    assert manifest["alpha"] == 0.05 and manifest["window"] == {"start": "2024-01-01", "end": "2025-12-31"}
    assert all(c["params_fp"] for c in manifest["candidates"]) and manifest["summary"]["selected"]["study"] == "b"


def test_preflight_failure_reads_no_data_and_writes_no_manifest(tmp_path, monkeypatch):
    argv, evaluated, out = run_main(tmp_path, monkeypatch, "a:0,missing:0", {"a": 0.4})
    with pytest.raises(ValueError, match="study not found"):
        vo.main(argv)
    assert evaluated == [] and not list(out.glob("campaign-*.json"))                      # nothing was exposed


def test_unresolved_trial_numbers_are_frozen_in_the_manifest(tmp_path, monkeypatch):
    argv, _, out = run_main(tmp_path, monkeypatch, "a", {"a": 0.4})                       # no explicit trial
    vo.main(argv)
    (manifest_path,) = out.glob("campaign-*.json")
    assert json.loads(manifest_path.read_text())["candidates"][0]["trial"] == 0


def test_identical_rerun_resumes_without_reevaluating(tmp_path, monkeypatch):
    argv, evaluated, out = run_main(tmp_path, monkeypatch, "a:0,b:0", {"a": 0.4, "b": 0.5})
    first = vo.main(argv)
    second = vo.main(argv)
    assert evaluated == ["a", "b"] and second["selected"] == first["selected"]            # no second read of the window
    assert len(list(out.glob("campaign-*.json"))) == 1


def test_failure_midway_resumes_only_the_pending_candidates(tmp_path, monkeypatch):
    argv, evaluated, out = run_main(tmp_path, monkeypatch, "a:0,b:0", {"a": 0.4, "b": 0.5})
    good = vo.evaluate_candidate
    state = {"fail": True}

    def flaky(prep, **kw):
        if prep["study"] == "b" and state["fail"]:
            raise RuntimeError("volume unmounted")
        return good(prep, **kw)
    monkeypatch.setattr(vo, "evaluate_candidate", flaky)
    with pytest.raises(RuntimeError, match="unmounted"):
        vo.main(argv)
    (manifest_path,) = out.glob("campaign-*.json")
    statuses = {c["study"]: c["status"] for c in json.loads(manifest_path.read_text())["candidates"]}
    assert statuses == {"a": "done", "b": "pending"}
    state["fail"] = False
    vo.main(argv)
    assert evaluated == ["a", "b"]                                                          # a was not read again


def test_a_different_candidate_set_is_refused_after_exposure(tmp_path, monkeypatch):
    argv, evaluated, out = run_main(tmp_path, monkeypatch, "a:0", {"a": 0.4, "b": 0.5})
    vo.main(argv)
    changed = argv[:3] + ["a:0,b:0"] + argv[4:]
    with pytest.raises(ValueError, match="already exposed.*campaign-"):
        vo.main(changed)
    assert evaluated == ["a"]
    summary = vo.main(changed + ["--acknowledge-prior-exposure"])
    (new,) = [m for m in map(lambda p: json.loads(p.read_text()), out.glob("campaign-*.json")) if len(m["candidates"]) == 2]
    assert new["prior_campaigns"] and summary["selected"] is not None


def test_bonferroni_counts_the_frozen_campaign_not_the_invocation(tmp_path, monkeypatch):
    argv, evaluated, out = run_main(tmp_path, monkeypatch, "a:0,b:0", {"a": 0.4, "b": 0.5})
    summary = vo.main(argv)
    assert summary["confirmation"][0]["z"] == pytest.approx(NormalDist().inv_cdf(1 - 0.05 / 2))


def test_validate_attaches_diagnostic_deflated_sharpe(tmp_path, monkeypatch):
    db, cache = make_study(tmp_path)
    stub_dev_replay(monkeypatch)
    monkeypatch.setattr(fe, "evaluate_frozen", lambda params, **kw: {**fake_result(0.3), "params_seen": params})
    seen = {}
    monkeypatch.setattr(opt, "deflated_gate", lambda study, returns, campaign=None, **kw:
                        seen.update(campaign=campaign) or {"diagnostic": True, "status": "evaluated"})
    result = vo.validate(db, "s", None, cache_dir=cache, data_root=tmp_path)
    assert result["deflated_sharpe_gate"]["diagnostic"] is True and result["trial"] == 0
    assert result["study"] == "s" and "absorption_min_delta_pct" in result["params"]       # legacy dims completed
    assert seen["campaign"]["n_symbol_all_studies"] == 0     # fixture study "s" has no BTCUSDT__ prefix


def test_validate_refuses_window_outside_policy_before_lookup(tmp_path, monkeypatch):
    db, cache = make_study(tmp_path)

    def forbidden(*a, **k):
        raise AssertionError("study lookup before the window guard")
    monkeypatch.setattr(dx_module(), "load_trial", forbidden)
    with pytest.raises(ValueError, match="validation window"):
        vo.validate(db, "s", None, cache_dir=cache, data_root=tmp_path, start=date(2026, 1, 1), end=date(2026, 3, 1))


def dx_module():
    from scripts import diagnose_trades
    return diagnose_trades


# ---- Codex review fixes: high severity ---------------------------------------------------------------

def bankrupt_but_positive_expectancy_result():
    """1,000 small long wins, then one short that gaps against the position by 101% (zero costs)."""
    from scripts import diagnose_trades as dx
    cases = [(100.0, 99.9, 100.2, Side.BUY)] * 1000 + [(100.0, 101.0, 201.0, Side.SELL)]
    entries, closes = [], []
    for i, (entry, stop, exit_price, side) in enumerate(cases):
        opened = SimpleNamespace(trade_id=str(i), bar_seq=2 * i, timestamp_ms=1_672_531_200_000 + i * 86_400_000,
                                 direction=side, fill_price=entry, stop_loss=stop, take_profit=300.0)
        closed = SimpleNamespace(trade_id=str(i), bar_seq=2 * i + 1, timestamp_ms=opened.timestamp_ms + 60_000,
                                 fill_price=exit_price, exit_reason="exit", pnl=exit_price - entry)
        entries.append(SimpleNamespace(execution=opened))
        closes.append(SimpleNamespace(execution=closed))
    trades = dx.add_gross(opt.simulate_trades_risk_based(SimpleNamespace(entries=entries, closes=closes),
                                                         fee_bps=0.0, slippage_bps=0.0))
    risk = trades["dollar_risk"]
    return {"decomposition": dx.decompose(trades), "trade_r": (trades["net_pnl"] / risk).tolist(),
            "metrics": {"account_blown": bool((trades["equity_after"] <= 0).any())}}, trades


def test_confirmation_rejects_a_bankrupt_strategy_with_positive_expectancy():
    result, trades = bankrupt_but_positive_expectancy_result()
    assert trades["equity_after"].iloc[-1] < 0                                  # insolvent
    assert result["decomposition"]["net_expectancy_r"] > 1.0                    # yet the mean R looks excellent
    row = vo.confirmation([result])[0]
    assert row["confirmed"] is False and row["account_blown"] is True


def test_confirmation_fails_closed_without_solvency_evidence():
    missing = candidate(0.5, 0.05)
    del missing["metrics"]
    row = vo.confirmation([missing])[0]
    assert row["confirmed"] is False and row["account_blown"] is None
    assert vo.confirmation([candidate(0.5, 0.05, blown=True)])[0]["confirmed"] is False
    assert vo.confirmation([candidate(0.5, 0.05)])[0]["confirmed"] is True


def forbid_validation_data(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("validation data accessed before the frozen-strategy preflight passed")
    monkeypatch.setattr(fe, "evaluate_frozen", forbidden)
    monkeypatch.setattr(opt, "build_cache", forbidden)


def test_validate_refuses_changed_strategy_before_data(tmp_path, monkeypatch):
    stale = {**IDENTITY, "strategy_fp": "other", "instrument_defaults": {
        **CURRENT_DEFAULTS, "absorption": {**CURRENT_DEFAULTS["absorption"], "min_delta_pct": 0.99}}}
    db, cache = make_study(tmp_path, identity=stale)
    stub_dev_replay(monkeypatch)
    forbid_validation_data(monkeypatch)
    with pytest.raises(ValueError, match="instrument default.*absorption.*min_delta_pct"):
        vo.validate(db, "s", None, cache_dir=cache, data_root=tmp_path)


def test_validate_refuses_incompatible_library_versions_before_data(tmp_path, monkeypatch):
    db, cache = make_study(tmp_path, versions={**opt.library_versions(), "optuna": "0.1.0"})
    stub_dev_replay(monkeypatch)
    forbid_validation_data(monkeypatch)
    with pytest.raises(ValueError, match="library.*optuna"):
        vo.validate(db, "s", None, cache_dir=cache, data_root=tmp_path)


def test_validate_refuses_unreproducible_development_replay_before_data(tmp_path, monkeypatch):
    db, cache = make_study(tmp_path)
    stub_dev_replay(monkeypatch, digest="OTHER")
    forbid_validation_data(monkeypatch)
    with pytest.raises(RuntimeError, match="mismatch"):
        vo.validate(db, "s", None, cache_dir=cache, data_root=tmp_path)


def test_validate_refuses_trial_without_stored_digest_before_data(tmp_path, monkeypatch):
    db, cache = make_study(tmp_path, digest=None)
    stub_dev_replay(monkeypatch)
    forbid_validation_data(monkeypatch)
    with pytest.raises(RuntimeError, match="unverifiable"):
        vo.validate(db, "s", None, cache_dir=cache, data_root=tmp_path)


def legacy_identity():
    legacy_defaults = {k: v for k, v in CURRENT_DEFAULTS.items() if k != "risk"}
    legacy_defaults["risk"] = {k: v for k, v in CURRENT_DEFAULTS["risk"].items()
                               if k not in ("default_stop_pct", "default_target_pct")}
    return {**IDENTITY, "strategy_fp": "legacy", "instrument_defaults": legacy_defaults}


def test_changed_strategy_fingerprint_is_refused_even_when_the_digest_matches(tmp_path, monkeypatch):
    """Replay equality on the development inputs cannot prove equivalence on validation inputs."""
    db, cache = make_study(tmp_path, identity=legacy_identity())
    stub_dev_replay(monkeypatch, digest="D")                       # the stored digest reproduces
    forbid_validation_data(monkeypatch)
    with pytest.raises(ValueError, match="strategy fingerprint.*legacy"):
        vo.validate(db, "s", None, cache_dir=cache, data_root=tmp_path)


def test_a_reviewed_compatibility_mapping_is_accepted_and_still_digest_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(vo, "COMPATIBLE_LEGACY_STRATEGY_FPS", frozenset({"legacy"}))
    db, cache = make_study(tmp_path, identity=legacy_identity())
    stub_dev_replay(monkeypatch, digest="OTHER")
    forbid_validation_data(monkeypatch)
    with pytest.raises(RuntimeError, match="mismatch"):            # the digest is an additional check, not skipped
        vo.validate(db, "s", None, cache_dir=cache, data_root=tmp_path)
    stub_dev_replay(monkeypatch, digest="D")
    monkeypatch.setattr(fe, "evaluate_frozen", lambda params, **kw: fake_result(0.3))
    monkeypatch.setattr(opt, "deflated_gate", lambda *a, **k: {"diagnostic": True})
    result = vo.validate(db, "s", None, cache_dir=cache, data_root=tmp_path)
    assert result["strategy_fp_matches"] is False and result["development_digest_verified"] is True


def test_the_default_compatibility_mapping_is_empty():
    assert vo.COMPATIBLE_LEGACY_STRATEGY_FPS == frozenset()


# ---- Codex review fixes: medium severity ---------------------------------------------------------------

def forbid_build(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("validation data read before the bar-construction check")
    monkeypatch.setattr(opt, "build_cache", forbidden)


def test_evaluate_frozen_refuses_a_newer_artifact_before_reading_data(window, monkeypatch):
    forbid_build(monkeypatch)
    monkeypatch.setattr(fe, "find_dataset", lambda *a: SimpleNamespace(dataset_id="src-NEW", sidecar={**DATASET["source_metadata"]}))
    with pytest.raises(ValueError, match="bar construction.*source"):
        fe.evaluate_frozen(PARAMS, **window)


def test_evaluate_frozen_refuses_changed_calibration_metadata(window, monkeypatch):
    forbid_build(monkeypatch)
    recalibrated = {**DATASET["source_metadata"], "hyperparams": {"e0_mult": 0.9}}
    monkeypatch.setattr(fe, "find_dataset", lambda *a: SimpleNamespace(dataset_id="src-A", sidecar=recalibrated))
    with pytest.raises(ValueError, match="bar construction.*source_metadata"):
        fe.evaluate_frozen(PARAMS, **window)


def test_evaluate_frozen_records_dataset_provenance(window, monkeypatch):
    monkeypatch.setattr(opt, "run_backtest", lambda bars, *a, **k: {"ledger": fake_ledger([]), "replay_digest": "d"})
    result = fe.evaluate_frozen(PARAMS, **window)
    assert result["dataset"] == {"source": "src-A", "fingerprint": "x", "data_proc_fp": "dp-1",
                                 "source_metadata": DATASET["source_metadata"]}


def test_cache_with_a_different_data_processing_fingerprint_is_refused(window, monkeypatch):
    path = write_cache(window["data_root"], STAMPS)
    monkeypatch.setattr(opt, "build_cache", lambda *a: (path, {"fingerprint": "x", **{**DATASET, "data_proc_fp": "dp-2"}}))
    with pytest.raises(ValueError, match="bar construction.*data_proc_fp"):
        fe.evaluate_frozen(PARAMS, **window)


def test_time_bars_skip_the_artifact_lookup_but_still_check_processing(window, monkeypatch):
    def forbidden(*a):
        raise AssertionError("time bars have no calibrated artifact to look up")
    monkeypatch.setattr(fe, "find_dataset", forbidden)
    window["bar_type"] = "time_1m"
    window["trades_root"] = window["data_root"]
    window["expected_dataset"] = {"source": ["stage-1"], "source_metadata": {}, "data_proc_fp": "dp-1"}
    path = write_cache(window["data_root"], STAMPS)
    monkeypatch.setattr(opt, "build_cache", lambda *a: (path, {"fingerprint": "x", "source": ["stage-2"],
                                                               "source_metadata": {}, "data_proc_fp": "dp-1"}))
    monkeypatch.setattr(opt, "run_backtest", lambda bars, *a, **k: {"ledger": fake_ledger([]), "replay_digest": "d"})
    assert fe.evaluate_frozen(PARAMS, **window)["dataset"]["source"] == ["stage-2"]     # differing month stages are normal


def test_validate_binds_the_validation_cache_to_the_development_dataset(tmp_path, monkeypatch):
    db, cache = make_study(tmp_path)
    stub_dev_replay(monkeypatch)
    seen = {}
    monkeypatch.setattr(fe, "evaluate_frozen", lambda params, **kw: seen.update(kw) or fake_result(0.3))
    monkeypatch.setattr(opt, "deflated_gate", lambda *a, **k: {"diagnostic": True})
    vo.validate(db, "s", None, cache_dir=cache, data_root=tmp_path)
    assert seen["expected_dataset"] == DATASET


# ---- time-bar (time_1m) validation -------------------------------------------------------------------

def time_bar_kwargs(tmp_path, monkeypatch, **overrides):
    """Synthetic tick archive + the real build_cache path; the policy window is moved to the synthetic day."""
    from test_optimize_orderflow import sample_ticks, write_ticks
    ticks = tmp_path / "ticks"
    write_ticks(ticks, "2023-06-10", sample_ticks("2023-06-10"))
    monkeypatch.setattr(policy, "VALIDATION_START", date(2023, 6, 10))
    monkeypatch.setattr(policy, "VALIDATION_END", date(2023, 6, 10))
    kwargs = dict(symbol="BTCUSDT", bar_type="time_1m", start=date(2023, 6, 10), end=date(2023, 6, 10),
                  settings={"fee_bps": 5.0, "slippage_bps": 1.0}, data_root=tmp_path / "unused",
                  cache_dir=tmp_path / "cache", trades_root=ticks, guard=policy.ensure_validation_window,
                  expected_dataset={"source": None, "source_metadata": {}, "data_proc_fp": opt.DATA_PROC_FP})
    return {**kwargs, **overrides}


def test_time_bar_validation_runs_through_the_real_cache_builder(tmp_path, monkeypatch):
    result = fe.evaluate_frozen(PARAMS, **time_bar_kwargs(tmp_path, monkeypatch))
    assert result["n_bars"] > 0 and result["window"] == {"start": "2023-06-10", "end": "2023-06-10"}
    assert result["dataset"]["data_proc_fp"] == opt.DATA_PROC_FP


def test_time_bar_validation_without_a_tick_archive_is_a_clear_error(tmp_path, monkeypatch):
    kwargs = time_bar_kwargs(tmp_path, monkeypatch, trades_root=None)
    with pytest.raises(ValueError, match="time_1m.*trades_root"):
        fe.evaluate_frozen(PARAMS, **kwargs)


def test_cli_exposes_trades_root_with_the_optimizer_default(monkeypatch):
    monkeypatch.setenv("ORDERFLOW_TRADES_ROOT", "/x/ticks")
    assert vo.build_parser().parse_args(["--candidates", "s"]).trades_root == Path("/x/ticks")
    monkeypatch.delenv("ORDERFLOW_TRADES_ROOT")
    assert vo.build_parser().parse_args(["--candidates", "s"]).trades_root == \
        Path(opt.build_parser().get_default("trades_root"))
    assert vo.build_parser().parse_args(["--candidates", "s", "--trades-root", "/y"]).trades_root == Path("/y")


def test_main_forwards_trades_root_to_the_evaluation(tmp_path, monkeypatch):
    argv, _, _ = run_main(tmp_path, monkeypatch, "a:0", {"a": 0.4}, extra=["--trades-root", str(tmp_path / "ticks")])
    seen = {}
    good = vo.evaluate_candidate
    monkeypatch.setattr(vo, "evaluate_candidate", lambda prep, **kw: seen.update(kw) or good(prep, **kw))
    vo.main(argv)
    assert seen["trades_root"] == tmp_path / "ticks"


# ---- campaign integrity: attribution, duplicates, concurrency -----------------------------------------

def test_reordered_partial_resume_keeps_attribution(tmp_path, monkeypatch):
    argv, evaluated, out = run_main(tmp_path, monkeypatch, "a:0,b:0", {"a": 0.4, "b": 0.5})
    good = vo.evaluate_candidate
    state = {"fail": True}

    def flaky(prep, **kw):
        if prep["study"] == "b" and state["fail"]:
            raise RuntimeError("interrupted")
        return good(prep, **kw)
    monkeypatch.setattr(vo, "evaluate_candidate", flaky)
    with pytest.raises(RuntimeError, match="interrupted"):
        vo.main(argv)
    state["fail"] = False
    reordered = argv[:3] + ["b:0,a:0"] + argv[4:]
    summary = vo.main(reordered)
    assert evaluated == ["a", "b"]                                       # a is not read again, b is evaluated once
    (manifest_path,) = out.glob("campaign-*.json")
    rows = {c["study"]: c["result"] for c in json.loads(manifest_path.read_text())["candidates"]}
    assert rows["a"]["study"] == "a" and rows["b"]["study"] == "b"
    assert rows["a"]["decomposition"]["net_expectancy_r"] == 0.4 and rows["b"]["decomposition"]["net_expectancy_r"] == 0.5
    assert summary["selected"]["study"] == "b"


def test_duplicate_candidates_are_refused(tmp_path, monkeypatch):
    argv, evaluated, out = run_main(tmp_path, monkeypatch, "a:0,a:0", {"a": 0.4})
    with pytest.raises(ValueError, match="duplicate candidate"):
        vo.main(argv)
    assert evaluated == [] and not list(out.glob("campaign-*.json"))


def test_record_result_refuses_a_result_for_another_candidate(tmp_path):
    path = tmp_path / "campaign-x.json"
    vo.write_manifest(path, {"candidates": [{"study": "a", "trial": 0, "params_fp": "f", "status": "pending"}]})
    with pytest.raises(ValueError, match="does not belong in the slot of a trial 0"):
        vo.record_result(path, 0, {"study": "b", "trial": 0})
    assert json.loads(path.read_text())["candidates"][0]["status"] == "pending"


def test_prior_exposure_admission_is_serialized_across_concurrent_invocations(tmp_path, monkeypatch):
    import threading
    import time
    out = tmp_path / "out"
    original = vo.write_manifest

    def slow_write(path, manifest):                                       # widen the race window deterministically
        time.sleep(0.3)
        original(path, manifest)
    monkeypatch.setattr(vo, "write_manifest", slow_write)
    start, end = date(2024, 1, 1), date(2025, 12, 31)
    outcomes, barrier = {}, threading.Barrier(2)

    def admit(name):
        barrier.wait()
        try:
            vo.admit_campaign(out, [prepared(name)], start, end, acknowledge=False)
            outcomes[name] = "admitted"
        except ValueError as exc:
            outcomes[name] = str(exc)
    threads = [threading.Thread(target=admit, args=(n,)) for n in ("a", "b")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(v == "admitted" for v in outcomes.values()) == [False, True]       # exactly one set is admitted
    loser = next(v for v in outcomes.values() if v != "admitted")
    assert "already exposed" in loser and len(list(out.glob("campaign-*.json"))) == 1


def test_an_identical_campaign_cannot_be_executed_twice_concurrently(tmp_path):
    import threading
    out = tmp_path / "out"
    path = vo.admit_campaign(out, [prepared("a")], date(2024, 1, 1), date(2025, 12, 31), acknowledge=False)
    started, release, evaluated = threading.Event(), threading.Event(), []

    def slow(prep):
        evaluated.append(prep["study"])
        started.set()
        release.wait(5)
        return {**fake_result(0.4), "study": prep["study"], "trial": prep["trial"]}
    first = threading.Thread(target=lambda: vo.run_campaign(path, [prepared("a")], slow))
    first.start()
    assert started.wait(5)
    with pytest.raises(RuntimeError, match="campaign already running"):
        vo.run_campaign(path, [prepared("a")], slow)
    release.set()
    first.join()
    assert evaluated == ["a"]

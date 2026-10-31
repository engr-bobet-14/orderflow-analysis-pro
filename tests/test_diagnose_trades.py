"""Gross/net decomposition, trade-count floors and dataset-policy refusal for scripts/diagnose_trades.py."""
import json
import math
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import pytest

from orderflow_system.data.models import Side
from scripts import diagnose_trades as dx
from scripts import optimize_orderflow as opt


def fake_ledger(cases=None, open_only=False):
    """Immutable snapshots like test_sizing_and_costs_match_notebook."""
    cases = cases or [(100., 95., 110., Side.BUY), (100., 99.999, 90., Side.BUY),
                      (100., 105., 90., Side.SELL), (100., 100., 110., Side.SELL)]
    entries, closes = [], []
    for i, (entry, stop, exit_price, side) in enumerate(cases):
        opened = SimpleNamespace(trade_id=str(i), bar_seq=2 * i, timestamp_ms=1672531200000 + i * 86400000,
                                 direction=side, fill_price=entry, stop_loss=stop, take_profit=120.)
        closed = SimpleNamespace(trade_id=str(i), bar_seq=2 * i + 1, timestamp_ms=opened.timestamp_ms + 60000,
                                 fill_price=exit_price, exit_reason="target", pnl=exit_price - entry)
        entries.append(SimpleNamespace(execution=opened))
        if not open_only:
            closes.append(SimpleNamespace(execution=closed))
    return SimpleNamespace(entries=entries, closes=closes)


def trades_table(**kw):
    table = dx.add_gross(opt.simulate_trades_risk_based(fake_ledger(**kw)))
    return table


def test_gross_net_identity():
    t = trades_table()
    assert (t.gross_pnl != 0).all()
    np.testing.assert_allclose(t.net_pnl, t.gross_pnl - t.slippage - t.fee, rtol=1e-9)


def test_decompose_default_stop_share_and_r_stats():
    t = trades_table()
    t["stop_loss_pct"] = [0.003, 0.003, 0.01, 0.0005]
    d = dx.decompose(t)
    assert d["n_trades"] == 4 and d["share_default_stop"] == 0.5
    assert d["gross_expectancy_r"] == pytest.approx((t.gross_pnl / t.dollar_risk).mean())
    assert d["std_r"] == pytest.approx(np.std(t.net_pnl / t.dollar_risk, ddof=1))
    assert d["se_r"] == pytest.approx(d["std_r"] / 2)
    assert d["cost_r_mean"] == pytest.approx(((t.fee + t.slippage) / t.dollar_risk).mean())
    assert d["gross_pnl"] - d["fee_total"] - d["slippage_total"] == pytest.approx(d["net_pnl"])


def test_open_position_excluded():
    t = trades_table(open_only=True)
    assert t.empty
    d = dx.decompose(t)
    assert d["n_trades"] == 0 and d["net_expectancy_r"] is None


def test_trades_needed_known_values():
    assert dx.trades_needed(0.2, 1.0) == 155
    assert dx.trades_needed(0.3, 1.0) == 69


def test_floors_derivation():
    base = {"cost_r_mean": 0.2, "std_r": 1.0}
    f = dx.floors(base, n_splits=5)
    assert (f["min_total_trades"], f["min_trades_per_fold"]) == (155, 16)
    assert dx.floors(base, n_splits=5, delta_multiplier=2.0)["min_total_trades"] == 39


def test_floors_refuse_unmeasured_inputs():
    with pytest.raises(ValueError):
        dx.floors({"cost_r_mean": None, "std_r": None}, n_splits=5)


def add_study_trial(study, value, feasible, params=None):
    attrs = {"params": params or {"min_composite_score": 50.0}}
    if feasible is not None:
        attrs["feasible"] = feasible
    study.add_trial(optuna.trial.create_trial(value=value, user_attrs=attrs))


def make_study(tmp_path, name="s"):
    db = tmp_path / "s.db"
    study = optuna.create_study(study_name=name, storage=f"sqlite:///{db}", direction="maximize")
    return db, study


def test_load_trial_picks_best_feasible(tmp_path):
    db, study = make_study(tmp_path)
    add_study_trial(study, -90.0, False)
    add_study_trial(study, -3.0, True, {"min_composite_score": 41.0})
    add_study_trial(study, -1.0, True, {"min_composite_score": 52.0})
    params, number = dx.load_trial(db, "s", None)
    assert number == 2 and params["min_composite_score"] == 52.0
    assert dx.load_trial(db, "s", 1)[1] == 1


def test_load_trial_no_feasible_or_legacy_requires_choice(tmp_path):
    db, study = make_study(tmp_path, "none")
    add_study_trial(study, -90.0, False)
    with pytest.raises(ValueError, match="no feasible trial"):
        dx.load_trial(db, "none", None)
    db2 = tmp_path / "legacy.db"
    legacy = optuna.create_study(study_name="legacy", storage=f"sqlite:///{db2}", direction="maximize")
    add_study_trial(legacy, -0.4, None)
    with pytest.raises(ValueError, match="pass --trial"):
        dx.load_trial(db2, "legacy", None)
    assert dx.load_trial(db2, "legacy", 0)[1] == 0


def test_scored_only_keeps_exits_inside_scored_windows():
    bars = [SimpleNamespace(seq=10 * i) for i in range(20)]
    folds = [{"score_start_idx": 2, "test_idx": np.arange(0, 6)},     # scored bars 2..5 -> seq 20..50
             {"score_start_idx": 12, "test_idx": np.arange(10, 16)}]  # scored bars 12..15 -> seq 120..150
    trades = pd.DataFrame({"exit_seq": [5, 20, 50, 51, 119, 120, 150, 151, 190]})
    kept = dx.scored_only(trades, bars, folds)
    assert list(kept["exit_seq"]) == [20, 50, 120, 150]


def test_reprice_reproduces_original_costs_and_zero_cost_is_gross():
    t = trades_table()
    same = dx.reprice_costs(t, fee_bps=5.0, slippage_bps=1.0)
    np.testing.assert_allclose(same.net_pnl, t.net_pnl, rtol=1e-9)
    free = dx.reprice_costs(t, fee_bps=0.0, slippage_bps=0.0)
    np.testing.assert_allclose(free.net_pnl, t.gross_pnl, rtol=1e-9)


def test_cost_grid_zero_costs_equal_gross_and_monotone():
    t = trades_table()
    grid = dx.cost_grid(t, fees=(0.0, 2.0, 5.0), slips=(0.0, 1.0, 3.0, 5.0))
    zero = grid[(grid.fee_bps == 0) & (grid.slippage_bps == 0)].iloc[0]
    assert zero.net_pnl == pytest.approx(t.gross_pnl.sum())
    for _, rows in grid.groupby("fee_bps"):
        assert rows.sort_values("slippage_bps").net_pnl.is_monotonic_decreasing
    for _, rows in grid.groupby("slippage_bps"):
        assert rows.sort_values("fee_bps").net_pnl.is_monotonic_decreasing


def stress_bars():
    array = np.zeros(4, dtype=opt.BAR_DTYPE)
    array["seq"] = [10, 20, 30, 40]
    array["open"] = [100., 101., 94., 100.]
    array["data_gap"] = [False, False, False, True]
    return SimpleNamespace(array=array)


def test_next_bar_entry_drops_unfillable():
    rows = [  # entry_seq, side, stop, target
        (10, "buy", 95., 120.),    # A: next open 101 is valid
        (40, "buy", 95., 120.),    # B: entry on the last bar, no next bar
        (20, "buy", 95., 120.),    # C: next open 94 is already below the long stop
        (30, "buy", 95., 120.),    # D: next bar is a data gap
    ]
    t = pd.DataFrame({"entry_seq": [r[0] for r in rows], "exit_seq": [r[0] + 5 for r in rows],
                      "side": [r[1] for r in rows], "stop_loss": [r[2] for r in rows],
                      "take_profit": [r[3] for r in rows], "entry_price": 100.0, "exit_price": 110.0,
                      "quantity": 1.0, "dollar_risk": 50.0})
    out, dropped = dx.next_bar_entry(t, stress_bars())
    assert dropped == 3 and len(out) == 1
    assert out.iloc[0].entry_price == 101.0 and out.iloc[0].entry_seq == 10
    assert out.iloc[0].gross_pnl == pytest.approx((110.0 - 101.0) * 1.0)


def test_stress_summary_reports_grid_and_latency():
    t = trades_table()
    t["entry_seq"] = [10, 10, 20, 20]
    summary = dx.stress_summary(t, stress_bars(), fee_bps=5.0, slippage_bps=1.0)
    assert set(summary) == {"cost_grid", "next_bar_entry"}
    assert len(summary["cost_grid"]) == 12
    latency = summary["next_bar_entry"]
    assert latency["dropped"] + latency["n_remaining"] == 4


# ---- replay guards (Codex review fixes) -------------------------------------------------------

def stored_identity(**override):
    base = {"data_fp": "fp-1", "n_splits": 5, "embargo": 0.02, "holdout_fraction": 0.2,
            "fee_bps": 5.0, "slippage_bps": 1.0, "initial_capital_usd": opt.INITIAL_CAPITAL_USD,
            "risk_per_trade_pct": opt.RISK_PER_TRADE_PCT, "min_stop_loss_pct": opt.MIN_STOP_LOSS_PCT,
            "annualization_periods": opt.ANNUALIZATION_PERIODS}
    return {**base, **override}


def test_replay_settings_come_from_stored_identity():
    settings = dx.bind_replay_settings(stored_identity(n_splits=4, embargo=0.03, holdout_fraction=0.1,
                                                       fee_bps=2.0, slippage_bps=0.5))
    assert settings == {"n_splits": 4, "embargo": 0.03, "holdout_fraction": 0.1,
                        "fee_bps": 2.0, "slippage_bps": 0.5}


@pytest.mark.parametrize("override", [{"initial_capital_usd": 1000.0}, {"risk_per_trade_pct": 0.02},
                                      {"min_stop_loss_pct": 0.001}, {"annualization_periods": 252}])
def test_changed_accounting_constants_refused(override):
    with pytest.raises(ValueError, match="accounting"):
        dx.bind_replay_settings(stored_identity(**override))


def test_identity_missing_keys_refused():
    identity = stored_identity()
    del identity["holdout_fraction"]
    with pytest.raises(ValueError, match="holdout_fraction"):
        dx.bind_replay_settings(identity)


@pytest.mark.parametrize("flag, value", [
    ("--holdout-fraction", "0"), ("--n-splits", "3"), ("--embargo", "0"), ("--fee-bps", "1"),
    ("--slippage-bps", "1"), ("--start-date", "2019-09-10"), ("--end-date", "2023-12-31"),
    ("--bar-type", "fixed_tick_imbalance"), ("--symbol", "BTCUSDT")])
def test_cli_cannot_override_stored_study_settings(flag, value):
    with pytest.raises(SystemExit):
        dx.build_parser().parse_args(["--study-name", "s", flag, value])


def test_cache_selected_by_stored_fingerprint(tmp_path):
    for name, fp in (("bars-a", "fp-0"), ("bars-b", "fp-1")):
        (tmp_path / f"{name}.json").write_text(json.dumps({"fingerprint": fp, "start": "2019-09-10",
                                                            "end": "2023-12-31", "symbol": "BTCUSDT",
                                                            "bar_type": "fixed_tick_imbalance"}))
    path, meta = dx.find_cached_bars(tmp_path, "fp-1")
    assert path == tmp_path / "bars-b.bin" and meta["fingerprint"] == "fp-1"
    with pytest.raises(ValueError, match="no cached bars"):
        dx.find_cached_bars(tmp_path, "fp-unknown")


def test_replay_never_exposes_the_holdout_tail(monkeypatch):
    from test_pipeline import synth_bars
    bars = synth_bars(100)
    seen = {}

    def capture(search, config, proximity, bar_type=None, **kw):
        seen["n"] = len(search)
        return {"ledger": fake_ledger(open_only=True), "replay_digest": "d"}
    monkeypatch.setattr(opt, "run_backtest", capture)
    args = SimpleNamespace(symbol="BTCUSDT", n_splits=3, embargo=0.02, holdout_fraction=0.2,
                           fee_bps=5.0, slippage_bps=1.0)
    params = {name: low for name, (low, _, _) in opt.SEARCH_SPACE_BOUNDS.items()}
    dx.replay_trades(bars, params, args)
    assert seen["n"] == 80 < len(bars)


def test_verify_replay_requires_matching_stored_digest():
    dx.verify_replay("abc", "abc")
    with pytest.raises(RuntimeError, match="mismatch"):
        dx.verify_replay("abc", "xyz")
    with pytest.raises(RuntimeError, match="unverifiable"):
        dx.verify_replay("abc", None)


def build_main_fixture(tmp_path, monkeypatch, *, stored_digest="D1", replay_digest="D1", end="2023-12-31"):
    db = tmp_path / "s.db"
    study = optuna.create_study(study_name="s", storage=f"sqlite:///{db}", direction="maximize")
    study.set_user_attr("identity", stored_identity())
    attrs = {"params": {"min_composite_score": 50.0}, "feasible": True}
    if stored_digest is not None:
        attrs["replay_digest"] = stored_digest
    study.add_trial(optuna.trial.create_trial(value=-1.0, user_attrs=attrs))
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "bars-x.json").write_text(json.dumps({"fingerprint": "fp-1", "start": "2019-09-10", "end": end,
                                                    "symbol": "BTCUSDT", "bar_type": "fixed_tick_imbalance"}))
    table = trades_table()
    table.attrs["replay_digest"] = replay_digest
    monkeypatch.setattr(opt, "CachedBars", lambda path, bar_type: SimpleNamespace(array=None, path=path))
    monkeypatch.setattr(opt, "split_timeline", lambda bars, args: (10, []))
    monkeypatch.setattr(dx, "replay_trades", lambda bars, params, args: table)
    return ["--study-db", str(db), "--study-name", "s", "--trial", "0", "--cache-dir", str(cache),
            "--output-dir", str(tmp_path / "out")]


def test_digest_mismatch_stops_before_any_decision_output(tmp_path, monkeypatch):
    argv = build_main_fixture(tmp_path, monkeypatch, replay_digest="OTHER")

    def forbidden(*a, **k):
        raise AssertionError("decision output computed before the digest was verified")
    monkeypatch.setattr(dx, "stress_summary", forbidden)
    monkeypatch.setattr(dx, "floors", forbidden)
    with pytest.raises(RuntimeError, match="mismatch"):
        dx.main(argv + ["--stress", "--floors"])
    assert not (tmp_path / "out").exists()


def test_missing_stored_digest_stops_before_replay(tmp_path, monkeypatch):
    argv = build_main_fixture(tmp_path, monkeypatch, stored_digest=None)

    def forbidden(*a, **k):
        raise AssertionError("replayed an unverifiable trial")
    monkeypatch.setattr(dx, "replay_trades", forbidden)
    with pytest.raises(RuntimeError, match="unverifiable"):
        dx.main(argv)
    assert not (tmp_path / "out").exists()


def test_post_dev_cache_refused_before_bars_are_opened(tmp_path, monkeypatch):
    argv = build_main_fixture(tmp_path, monkeypatch, end="2024-02-01")

    def forbidden(*a, **k):
        raise AssertionError("bars opened before the dataset-policy check")
    monkeypatch.setattr(opt, "CachedBars", forbidden)
    with pytest.raises(ValueError, match="dataset policy"):
        dx.main(argv)


def test_verified_run_persists_effective_replay_settings(tmp_path, monkeypatch):
    argv = build_main_fixture(tmp_path, monkeypatch)
    report = dx.main(argv)
    assert report["digest_matches"] is True
    assert report["replay_settings"] == {"n_splits": 5, "embargo": 0.02, "holdout_fraction": 0.2,
                                         "fee_bps": 5.0, "slippage_bps": 1.0, "data_fp": "fp-1",
                                         "start": "2019-09-10", "end": "2023-12-31", "n_search": 10}
    assert any((tmp_path / "out").glob("*__trial0.json"))


def test_legacy_params_are_completed_with_explicit_baseline(tmp_path, monkeypatch):
    argv = build_main_fixture(tmp_path, monkeypatch)
    seen = {}

    def capture(bars, params, args):
        seen["params"] = params
        table = trades_table()
        table.attrs["replay_digest"] = "D1"
        return table
    monkeypatch.setattr(dx, "replay_trades", capture)
    # stored study: replay digest is "D1" in build_main_fixture's default
    report = dx.main(argv)
    assert seen["params"]["min_composite_score"] == 50.0               # the stored value wins
    assert set(opt.baseline_extra_params()) <= set(seen["params"])
    assert set(report["baseline_filled_dims"]) == set(opt.baseline_extra_params())

"""Warm-start from prior best configurations: resolve-before-select, persisted seeds, idempotent reconcile."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sqlite3
import threading

import optuna
import pytest

from scripts import optimize_orderflow as opt
from scripts.dataset_policy import DEV_END
from test_optimize_orderflow import args_for, cli_args, entry_of, run_cli

SPACE = opt.SEARCH_SPACE_BOUNDS
META = {"fingerprint": "a" * 64}
ORIGINAL_DIMS = ("price_proximity_pct", "min_composite_score", "signal_cooldown_seconds",
                 "volume_decline_pct", "lookback_bars", "delta_failure_pct")
COLD_KEYS = {"data_fp", "space_fp", "search_space", "strategy_fp", "instrument_defaults", "initial_capital_usd",
             "risk_per_trade_pct", "min_stop_loss_pct", "annualization_periods", "objective", "accounting",
             "infeasible_base", "feasible_floor", "min_total_trades", "fee_bps", "slippage_bps", "n_splits",
             "embargo", "holdout_fraction", "min_trades_per_fold", "seed"}


def src(tag, start="2023-01-01", end="2023-02-01", symbol="BTCUSDT", bar_type="fixed_dollar_imbalance",
        fp=META["fingerprint"]):
    """A source study name; defaults to the cli_args destination window and the META data fingerprint."""
    return f"{symbol}__{bar_type}__{start}_{end}__d{fp[:8]}__s{tag}"


def destination(tmp_path):
    """argv, parsed args and the real bar-cache metadata of the cli_args run. Call once per tmp_path:
    cli_args rewrites the dataset, which changes the data fingerprint."""
    argv = cli_args(tmp_path)
    args = opt.build_parser().parse_args(argv)
    args.symbol, args.bar_type, args.space = "BTCUSDT", "fixed_dollar_imbalance", dict(SPACE)
    return argv, args, opt.build_cache(args, args.symbol, args.bar_type)[1]


def valid_params(**overrides):
    params = {n: (lo + hi) // 2 if kind == "int" else round((lo + hi) / 2, 6) for n, (lo, hi, kind) in SPACE.items()}
    return {**params, **overrides}


def dist(name, value):
    lo, hi, kind = SPACE.get(name, (0.0, 1.0, "float"))
    lo, hi = min(lo, value), max(hi, value)
    if kind == "int" and isinstance(value, int):
        return optuna.distributions.IntDistribution(int(lo), int(hi))
    return optuna.distributions.FloatDistribution(float(lo), float(hi), log=kind == "float_log")


def source_trial(value, *, params=None, feasible=True, total_trades=200, n_trades=40, attrs=None):
    params = valid_params() if params is None else params
    return optuna.trial.create_trial(
        params=params, distributions={k: dist(k, v) for k, v in params.items()}, value=value,
        user_attrs={"feasible": feasible, "total_trades": total_trades,
                    "fold_results": [{"n_trades": n_trades}] * 5, **(attrs or {})})


def make_source_study(db, name, trials, *, study_attrs=None, holdout_fraction=0.2):
    """holdout_fraction=None stores no identity attr at all."""
    study = optuna.create_study(study_name=name, storage=f"sqlite:///{db}", direction="maximize")
    for spec in trials:
        study.add_trial(source_trial(**spec))
    identity = {} if holdout_fraction is None else {"identity": {"holdout_fraction": holdout_fraction}}
    for key, value in {**identity, **(study_attrs or {})}.items():
        study.set_user_attr(key, value)
    return study


def load(db, name):
    return optuna.load_study(study_name=name, storage=f"sqlite:///{db}")


def keys(seeds):
    return sorted(f'{s["study"]}#{s["trial"]}' for s in seeds)


def seed_sources(db, name):
    return sorted(t["attrs"]["warm_start_source"] for t in opt.read_study_trials(db, name)
                  if "warm_start_source" in t["attrs"])


def feasible_folds(monkeypatch):
    """Every tiny-run trial feasible with value 1.0, so the run's own best is an eligible warm-start seed."""
    base = {"account_blown": False, "n_scored_bars": 10, "n_days": 10, "penalized": False,
            "n_trades": 3, "sharpe_ratio": 1.0}
    monkeypatch.setattr(opt, "evaluate_fold", lambda *a, **kw: dict(base))


def warm_setup(tmp_path, n=3):
    tmp_path.mkdir(exist_ok=True)
    db = tmp_path / "s.db"
    for i in range(n):
        make_source_study(db, src(f"0{i}"), [{"value": -0.1 * (i + 1),
                                              "params": valid_params(min_composite_score=20.0 + i)}])
    args = args_for(tmp_path, study_db=db)
    resolved = opt.resolve_study(args, META)
    assert resolved["mode"] == "new_warm" and len(resolved["seeds"]) == n
    return args, resolved


def test_selects_best_feasible_per_study_ordered_and_capped(tmp_path):
    db = tmp_path / "src.db"
    distinct = {tag: valid_params(min_composite_score=20.0 + i) for i, tag in enumerate(("00", "01", "02", "03"))}
    make_source_study(db, src("01"), [{"value": -0.5, "params": distinct["01"]}])
    make_source_study(db, src("02"), [{"value": 1.0, "feasible": False}, {"value": -0.2, "params": distinct["02"]}])
    make_source_study(db, src("03"), [{"value": -0.9, "params": distinct["03"]}])
    make_source_study(db, src("00"), [{"value": -0.5, "params": distinct["00"]}])
    make_source_study(db, src("e", symbol="ETHUSDT"), [{"value": 5.0}])            # other symbol: ignored
    make_source_study(db, src("t", bar_type="fixed_tick_imbalance"), [{"value": 5.0}])  # other bar type
    seeds, skipped = opt.select_warm_start(args_for(tmp_path, study_db=db, warm_start_max=2, no_warm_start=False),
                                           META)
    assert [s["value"] for s in seeds] == [-0.2, -0.5]            # value desc, capped, infeasible ignored
    assert all(set(s["params"]) == set(SPACE) for s in seeds)
    assert skipped == []
    seeds, _ = opt.select_warm_start(args_for(tmp_path, study_db=db, warm_start_max=3), META)
    assert [(s["study"], s["trial"], s["value"]) for s in seeds] == [
        (src("02"), 1, -0.2), (src("00"), 0, -0.5), (src("01"), 0, -0.5)]   # ties: study name, trial number


STUDY_LEVEL = {"end_2024": {"name": src("x", end="2024-06-30")},
               "start_2018": {"name": src("x", start="2018-12-31")},
               "has_warm_start_done_attr": {"study_attrs": {"warm_start_done": True}},
               "has_warm_start_skipped_attr": {"study_attrs": {"warm_start_skipped": []}},
               "has_holdout_attr": {"study_attrs": {"holdout::2026-01-01T00:00:00+00:00::ab": {"status": "evaluated"}}}}
TRIAL_LEVEL = {"trades_100": {"total_trades": 100}, "fold_trades_10": {"n_trades": 10},
               "value_at_floor": {"value": opt.FEASIBLE_FLOOR},
               "extra_param": {"params": valid_params(legacy_dim=0.5)},
               "stop_above_hi": {"params": valid_params(default_stop_pct=0.02)},
               "lookback_12_5": {"params": valid_params(lookback_bars=12.5)}}


@pytest.mark.parametrize("case,reason", [
    ("end_2024", "not_dev_window"), ("start_2018", "not_dev_window"),
    ("has_warm_start_done_attr", "derived_study"), ("has_warm_start_skipped_attr", "derived_study"),
    ("has_holdout_attr", "holdout_seen"),
    ("trades_100", "infeasible_under_floors"), ("fold_trades_10", "infeasible_under_floors"),
    ("value_at_floor", "degenerate_value"),
    ("extra_param", "unknown_dimension"), ("stop_above_hi", "out_of_bounds"), ("lookback_12_5", "non_integral_int")])
def test_skips_carry_reasons_and_fall_through_to_next_best(tmp_path, case, reason):
    db = tmp_path / "src.db"
    study = STUDY_LEVEL.get(case, {})
    name = study.get("name", src("x"))
    best = {"value": -0.1, **TRIAL_LEVEL.get(case, {})}
    # A feasible trial scores >= FEASIBLE_FLOOR, so a degenerate best has no worse feasible trial behind it.
    worse = {"value": -0.3, "params": valid_params(lookback_bars=12.0)} if case == "lookback_12_5" else {"value": -0.3}
    trials = [best] if case == "value_at_floor" else [best, worse]
    make_source_study(db, name, trials, study_attrs=study.get("study_attrs"))
    seeds, skipped = opt.select_warm_start(args_for(tmp_path, study_db=db, min_total_trades=105,
                                                    min_trades_per_fold=11), META)
    if case in STUDY_LEVEL:
        assert seeds == [] and skipped == [{"study": name, "trial": None, "reason": reason}]
    elif case == "value_at_floor":
        assert seeds == [] and skipped == [{"study": name, "trial": 0, "reason": reason},
                                           {"study": name, "trial": None, "reason": "no_eligible_trial"}]
    else:
        assert [s["trial"] for s in seeds] == [1]
        assert skipped == [{"study": name, "trial": 0, "reason": reason}]


def test_missing_dimensions_filled_from_baseline(tmp_path):
    db = tmp_path / "src.db"
    old = {k: v for k, v in valid_params().items() if k in ORIGINAL_DIMS}
    make_source_study(db, src("01"), [{"value": -0.2, "params": old}])
    seed, = opt.select_warm_start(args_for(tmp_path, study_db=db), META)[0]
    extra = opt.baseline_extra_params()
    assert set(seed["params"]) == set(SPACE)
    assert {k: seed["params"][k] for k in old} == old
    assert {k: seed["params"][k] for k in extra} == extra


def test_never_writes_sources(tmp_path):
    argv, args, meta = destination(tmp_path)
    db, name = args.study_db, src("01", fp=meta["fingerprint"])
    make_source_study(db, name, [{"value": -0.2}])

    def snapshot():
        return hashlib.sha256(db.read_bytes()).hexdigest(), db.stat().st_mtime_ns
    before = snapshot()
    assert opt.select_warm_start(args, meta)[0]
    assert opt.resolve_study(args, meta)["mode"] == "new_warm"
    assert snapshot() == before
    content = opt.read_study_attrs(db, name), opt.read_study_trials(db, name)
    entry = entry_of(opt.main(argv))
    assert [s["study"] for s in entry["warm_start"]["seeds"]] == [name]   # the run did read the source
    assert (opt.read_study_attrs(db, name), opt.read_study_trials(db, name)) == content


def test_cold_identity_unchanged_and_warm_name_excludes_seeds(tmp_path):
    args = args_for(tmp_path)
    cold_name, cold = opt.study_identity(args, META)
    assert "warm_start" not in cold and set(cold) == COLD_KEYS
    assert (cold_name, cold) == opt.study_identity(args, META, warm_start=False)
    assert cold["space_fp"] == opt.fingerprint({k: v for k, v in cold.items() if k not in ("data_fp", "space_fp")})
    warm_name, warm = opt.study_identity(args, META, warm_start=True)
    assert warm["warm_start"] == {"max": args.warm_start_max} and warm_name != cold_name
    assert ({k: v for k, v in warm.items() if k not in ("warm_start", "space_fp")}
            == {k: v for k, v in cold.items() if k != "space_fp"})
    assert opt.study_identity(args_for(tmp_path, warm_start_max=5), META, warm_start=True)[0] != warm_name


def test_no_warm_start_flag_returns_empty(tmp_path):
    db = tmp_path / "src.db"
    make_source_study(db, src("01"), [{"value": -0.2}])
    assert opt.select_warm_start(args_for(tmp_path, study_db=db, no_warm_start=True), META) == ([], [])
    parsed = opt.build_parser().parse_args([])
    assert parsed.no_warm_start is False and parsed.warm_start_max == 3
    assert opt.build_parser().parse_args(["--no-warm-start", "--warm-start-max", "5"]).warm_start_max == 5
    with pytest.raises(SystemExit):
        opt.build_parser().parse_args(["--warm-start-max", "0"])


def test_resolve_modes(tmp_path):
    db = tmp_path / "s.db"
    args = args_for(tmp_path, study_db=db)
    cold_name, cold = opt.study_identity(args, META)
    empty = opt.resolve_study(args, META)
    assert empty == {"name": cold_name, "identity": cold, "mode": "new_cold", "seeds": [], "skipped": []}
    assert not db.exists()                                    # resolution never creates the DB
    make_source_study(db, src("01"), [{"value": -0.2}])
    names = opt.list_study_names(db)
    resolved = opt.resolve_study(args, META)
    seeds, skipped = opt.select_warm_start(args, META)
    warm_name, warm = opt.study_identity(args, META, warm_start=True)
    assert resolved["mode"] == "new_warm" and resolved["name"] == warm_name   # seeds are not in the name
    assert resolved["seeds"] == seeds and resolved["skipped"] == skipped
    assert resolved["identity"] == {**warm, "warm_start": {"max": args.warm_start_max, "seeds": seeds}}
    no_warm = opt.resolve_study(args_for(tmp_path, study_db=db, no_warm_start=True), META)
    assert (no_warm["mode"], no_warm["name"], no_warm["identity"]) == ("new_cold", cold_name, cold)
    assert opt.list_study_names(db) == names                  # resolution never creates a study
    opt.open_study(args, resolved["name"], resolved["identity"])
    again = opt.resolve_study(args, META)
    assert again["mode"] == "resume_warm" and again["identity"] == resolved["identity"]
    # A study created but never given an identity (crash right after create_study) counts as non-existent.
    optuna.create_study(study_name=cold_name, storage=f"sqlite:///{db}")
    no_warm = opt.resolve_study(args_for(tmp_path, study_db=db, no_warm_start=True), META)
    assert no_warm["mode"] == "new_cold"
    opt.open_study(args, cold_name, cold)
    assert opt.resolve_study(args, META)["mode"] == "resume_cold"


def test_original_command_of_existing_cold_study_resumes_it(tmp_path, monkeypatch):
    feasible_folds(monkeypatch)
    args = args_for(tmp_path)                    # before cli_args: each call rewrites the dataset (new data fp)
    db, argv = tmp_path / "s.db", cli_args(tmp_path)
    cold = entry_of(opt.main(argv + ["--no-warm-start"]))
    name = cold["study_name"]
    assert [s["study"] for s in opt.select_warm_start(args, cold["dataset"])[0]] == [name]   # eligible seed
    assert opt.resolve_study(args, cold["dataset"])["mode"] == "resume_cold"
    again = entry_of(opt.main(argv + ["--n-trials", "3"]))
    assert again["study_name"] == name and opt.list_study_names(db) == [name]
    assert "warm_start" not in opt.read_study_attrs(db, name)["identity"]
    assert again["trial_counts"]["COMPLETE"] == 3 and again["warm_start"] is None


@pytest.mark.parametrize("mutation", ["better_trial", "holdout_attr"])
def test_warm_resume_ignores_source_changes(tmp_path, mutation):
    argv, args, meta = destination(tmp_path)
    db, source = args.study_db, src("01", fp=meta["fingerprint"])
    make_source_study(db, source, [{"value": -0.2}])
    first = entry_of(opt.main(argv))
    warm = first["study_name"]
    stored = opt.read_study_attrs(db, warm)["identity"]
    assert [s["study"] for s in stored["warm_start"]["seeds"]] == [source]
    sources = seed_sources(db, warm)
    if mutation == "better_trial":
        load(db, source).add_trial(source_trial(0.5, params=valid_params(min_composite_score=33.0)))
    else:
        load(db, source).set_user_attr("holdout::x", {"status": "evaluated"})
    resolved = opt.resolve_study(args, first["dataset"])
    assert resolved["mode"] == "resume_warm" and resolved["identity"] == stored
    again = entry_of(opt.main(argv + ["--n-trials", "4"]))
    assert again["study_name"] == warm and again["trial_counts"]["COMPLETE"] == 4
    assert opt.read_study_attrs(db, warm)["identity"] == stored
    assert seed_sources(db, warm) == sources
    assert [t for t in opt.read_study_trials(db, warm) if t["state"] == "WAITING"] == []
    assert sorted(opt.list_study_names(db)) == sorted([source, warm])


def test_derived_source_detected_from_stored_identity(tmp_path):
    db = tmp_path / "src.db"
    make_source_study(db, src("01"), [{"value": -0.2}], study_attrs={"identity": {"warm_start": {"max": 3}}})
    assert opt.select_warm_start(args_for(tmp_path, study_db=db), META) == (
        [], [{"study": src("01"), "trial": None, "reason": "derived_study"}])


def test_open_study_enqueues_each_seed_once_and_marks_done(tmp_path):
    args, resolved = warm_setup(tmp_path)
    name, identity, seeds = resolved["name"], resolved["identity"], resolved["seeds"]
    study = opt.open_study(args, name, identity)
    assert sorted(t.user_attrs["warm_start_source"] for t in study.trials) == keys(seeds)
    # A WAITING trial has no params yet; Optuna keeps the enqueued ones in system_attrs["fixed_params"].
    assert [t.system_attrs["fixed_params"] for t in study.trials if t.state.name == "WAITING"] == [
        s["params"] for s in seeds]
    assert study.user_attrs["warm_start_done"] is True
    assert study.user_attrs["identity"]["warm_start"]["seeds"] == seeds
    assert len(opt.open_study(args, name, identity).trials) == len(seeds)             # reopen adds none


def test_partial_initialization_is_reconciled(tmp_path, monkeypatch):
    # (1) crash between enqueues: identity (intent) persisted, one seed enqueued, no marker.
    args, resolved = warm_setup(tmp_path / "a")
    name, identity, seeds = resolved["name"], resolved["identity"], resolved["seeds"]
    real_enqueue, calls = optuna.study.Study.enqueue_trial, []

    def flaky_enqueue(self, *a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("crash between enqueues")
        return real_enqueue(self, *a, **kw)
    monkeypatch.setattr(optuna.study.Study, "enqueue_trial", flaky_enqueue)
    with pytest.raises(RuntimeError, match="between enqueues"):
        opt.open_study(args, name, identity)
    monkeypatch.undo()
    attrs = opt.read_study_attrs(args.study_db, name)
    assert len(opt.read_study_trials(args.study_db, name)) == 1
    assert attrs["identity"] == identity and "warm_start_done" not in attrs
    study = opt.open_study(args, name, identity)
    assert len(study.trials) == len(seeds) and seed_sources(args.study_db, name) == keys(seeds)
    assert study.user_attrs["warm_start_done"] is True

    # (2) crash after every enqueue but before the marker: reopen must not duplicate.
    args, resolved = warm_setup(tmp_path / "b")
    name, identity, seeds = resolved["name"], resolved["identity"], resolved["seeds"]
    real_set = optuna.study.Study.set_user_attr

    def flaky_set(self, key, value):
        if key == "warm_start_done":
            raise RuntimeError("crash before marker")
        return real_set(self, key, value)
    monkeypatch.setattr(optuna.study.Study, "set_user_attr", flaky_set)
    with pytest.raises(RuntimeError, match="before marker"):
        opt.open_study(args, name, identity)
    monkeypatch.undo()
    assert len(opt.read_study_trials(args.study_db, name)) == len(seeds)
    study = opt.open_study(args, name, identity)
    assert len(study.trials) == len(seeds) and seed_sources(args.study_db, name) == keys(seeds)
    assert study.user_attrs["warm_start_done"] is True


def test_concurrent_open_study_enqueues_once(tmp_path):
    args, resolved = warm_setup(tmp_path)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda _: opt.open_study(args, resolved["name"], resolved["identity"]), range(4)))
    assert len(opt.read_study_trials(args.study_db, resolved["name"])) == len(resolved["seeds"])
    assert seed_sources(args.study_db, resolved["name"]) == keys(resolved["seeds"])


def test_enqueued_trials_are_reevaluated_not_copied(tmp_path):
    argv, args, meta = destination(tmp_path)
    db, names = args.study_db, [src(f"0{i}", fp=meta["fingerprint"]) for i in range(2)]
    for i, name in enumerate(names):
        make_source_study(db, name, [{"value": 123.0, "params": valid_params(min_composite_score=20.0 + i),
                                      "attrs": {"replay_digest": "stale"}}])
    entry = entry_of(opt.main(argv + ["--n-trials", "3"]))
    seeds = entry["warm_start"]["seeds"]
    assert keys(seeds) == [f"{names[0]}#0", f"{names[1]}#0"] and entry["warm_start_skipped"] == []
    trials = opt.read_study_trials(db, entry["study_name"])
    assert len(trials) == 3 and all(t["state"] == "COMPLETE" for t in trials)   # seeds count toward --n-trials
    for trial, seed in zip(trials[:2], seeds):
        assert trial["params"] == seed["params"] and trial["value"] != 123.0
        assert trial["attrs"]["replay_digest"] != "stale"
        assert trial["attrs"]["warm_start_source"] == f'{seed["study"]}#{seed["trial"]}'
    saved = entry_of(json.loads((tmp_path / "r.json").read_text()))
    assert saved["warm_start"] == entry["warm_start"] and saved["warm_start"]["seeds"] == seeds


# -- dry-run and the campaign runbook ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = "python scripts/optimize_orderflow.py"


def dry_run_plan(tmp_path, *extra, argv=None):
    proc = run_cli((argv or cli_args(tmp_path)) + ["--dry-run", *extra])
    assert proc.returncode == 0, proc.stderr
    return json.loads([line for line in proc.stdout.splitlines() if line.strip()][-1])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fenced_bash_blocks(path, containing):
    blocks = re.findall(r"```bash\n(.*?)```", Path(path).read_text(), re.S)
    return [b.replace("\\\n", " ") for b in blocks if containing in b]


def shlex_args(block):
    tokens = shlex.split(block)
    assert " ".join(tokens[:2]) == LAUNCHER, block
    return tokens[2:]


def test_dry_run_creates_nothing_and_reports_plan(tmp_path):
    plan = dry_run_plan(tmp_path)
    assert plan["dry_run"] is True and plan["floors"]["n_splits"] == 3
    assert plan["costs"] == {"fee_bps": 5.0, "slippage_bps": 1.0}
    assert plan["study_exists"] is False and plan["mode"] == "new_cold"
    assert plan["bar_type"] == "fixed_dollar_imbalance" and plan["n_trials"] == 2 and plan["n_jobs"] == 1
    assert plan["warm_start"] == {"seeds": [], "skipped": []} and plan["widened_vs_default"] == []
    assert set(plan["window"]) == {"start", "end", "n_bars", "n_search_bars"}
    assert 0 < plan["window"]["n_search_bars"] < plan["window"]["n_bars"]
    assert not (tmp_path / "s.db").exists() and not (tmp_path / "r.json").exists()


def test_dry_run_reports_warm_start_seeds_and_widened_dims(tmp_path):
    argv, _, meta = destination(tmp_path)
    source = src("01", fp=meta["fingerprint"])
    make_source_study(tmp_path / "s.db", source, [{"value": -0.2, "params": valid_params()}])
    before = sha(tmp_path / "s.db")
    space = tmp_path / "space.json"
    space.write_text(json.dumps({"target_rr": {"low": 0.6, "high": 4.0},
                                 "lookback_bars": {"low": 5, "high": 30}}))   # second equals the default
    plan = dry_run_plan(tmp_path, "--search-space", str(space), argv=argv)
    assert plan["mode"] == "new_warm" and plan["study_exists"] is False
    assert plan["widened_vs_default"] == ["target_rr"]
    assert plan["search_space"]["target_rr"] == [0.6, 4.0, "float"]
    assert [f'{s["study"]}#{s["trial"]}' for s in plan["warm_start"]["seeds"]] == [f"{source}#0"]
    assert sha(tmp_path / "s.db") == before and not (tmp_path / "r.json").exists()


def test_dry_run_reports_resume_mode_for_existing_study(tmp_path):
    argv = cli_args(tmp_path)
    proc = run_cli(argv + ["--no-warm-start"])
    assert proc.returncode == 0, proc.stderr
    db, out = tmp_path / "s.db", tmp_path / "r.json"
    name, = opt.list_study_names(db)
    load(db, name).set_user_attr("holdout::2026-01-01T00:00:00+00:00::ab", {"status": "evaluated"})
    before = (sha(db), sha(out))
    plan = dry_run_plan(tmp_path, argv=argv)
    assert plan["mode"] == "resume_cold" and plan["study_exists"] is True
    assert plan["holdout"]["prior_evaluations"] == 1
    assert (sha(db), sha(out)) == before


def test_dry_run_reports_holdout_behaviour(tmp_path):
    argv = cli_args(tmp_path)
    off = dry_run_plan(tmp_path, argv=argv)
    on = dry_run_plan(tmp_path, "--evaluate-holdout", argv=argv)
    assert off["holdout"] == {"evaluate_holdout": False, "holdout_fraction": 0.2, "embargo": 0.02,
                              "prior_evaluations": 0}
    assert on["holdout"] == {**off["holdout"], "evaluate_holdout": True} and on != off


def test_runbook_commands_pin_costs_floors_and_dev_window():
    blocks = fenced_bash_blocks(ROOT / "USAGE.md", containing="scripts/search_spaces/")
    launches = [b for b in blocks if "--dry-run" not in b]
    assert len(launches) == 2 and len(blocks) == 4 and any("--dry-run" in b for b in blocks)
    parsed = {}
    for block in blocks:
        a = opt.build_parser().parse_args(shlex_args(block))
        assert (a.min_total_trades, a.min_trades_per_fold, a.n_splits) == (105, 11, 5)
        assert (a.fee_bps, a.slippage_bps) == (5.0, 1.0)
        assert a.end_date <= DEV_END and not a.evaluate_holdout and 300 <= a.n_trials <= 800
        assert a.search_space.is_file() or (ROOT / a.search_space).is_file()
        assert (a.seed, a.n_jobs, a.start_date) == (43, 1, date(2019, 9, 10))
        assert a.study_db == Path("output/optimize/main/studies.db")
        assert a.output == Path("output/optimize/campaign2/results.json")
        bar_type, = a.bar_type
        parsed[(bar_type, a.dry_run)] = vars(a)
    assert {b: parsed[(b, False)]["n_trials"] for b, _ in parsed} == {
        "fixed_tick_imbalance": 300, "fixed_volume_imbalance": 300}
    for (bar_type, dry), a in parsed.items():
        if dry:      # each dry-run is its launch command plus --dry-run, nothing else
            assert {**a, "dry_run": False} == parsed[(bar_type, False)]


# -- fix wave 1 -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case,reason", [
    ("hf_0", "holdout_overlap"), ("hf_0_1", "holdout_overlap"), ("hf_missing", "holdout_overlap"),
    ("no_identity", "holdout_overlap"), ("other_dates", "window_mismatch"), ("other_data_fp", "window_mismatch"),
    ("hf_equal", None), ("hf_larger", None)])
def test_source_must_share_window_data_and_cover_the_holdout(tmp_path, case, reason):
    db = tmp_path / "src.db"
    name = {"other_dates": src("x", start="2022-01-01"), "other_data_fp": src("x", fp="b" * 64)}.get(case, src("x"))
    kw = {"hf_0": {"holdout_fraction": 0.0}, "hf_0_1": {"holdout_fraction": 0.1},
          "hf_larger": {"holdout_fraction": 0.3}, "no_identity": {"holdout_fraction": None},
          "hf_missing": {"study_attrs": {"identity": {"seed": 42}}}}.get(case, {})
    make_source_study(db, name, [{"value": -0.2}], **kw)
    seeds, skipped = opt.select_warm_start(args_for(tmp_path, study_db=db), META)   # destination hf 0.2
    if reason:
        assert seeds == [] and skipped == [{"study": name, "trial": None, "reason": reason}]
    else:
        assert [s["study"] for s in seeds] == [name] and skipped == []


def test_duplicate_params_fall_through_to_the_next_eligible_trial(tmp_path):
    db = tmp_path / "src.db"
    same, other = valid_params(), valid_params(min_composite_score=21.0)
    make_source_study(db, src("00"), [{"value": -0.1, "params": same}])
    make_source_study(db, src("01"), [{"value": -0.2, "params": same}, {"value": -0.3, "params": other}])
    seeds, skipped = opt.select_warm_start(args_for(tmp_path, study_db=db), META)
    assert [(s["study"], s["trial"]) for s in seeds] == [(src("00"), 0), (src("01"), 1)]
    assert skipped == [{"study": src("01"), "trial": 0, "reason": "duplicate_params"}]


def test_uninitialised_db_resolves_and_open_study_initialises_it(tmp_path):
    db = tmp_path / "s.db"
    db.touch()                                                # e.g. from mktemp
    args = args_for(tmp_path, study_db=db)
    for no_warm_start in (False, True):
        args.no_warm_start = no_warm_start
        assert opt.resolve_study(args, META)["mode"] == "new_cold"
    assert db.stat().st_size == 0 and sorted(p.name for p in tmp_path.glob("s.db*")) == ["s.db"]
    resolved = opt.resolve_study(args, META)
    opt.open_study(args, resolved["name"], resolved["identity"])
    assert opt.list_study_names(db) == [resolved["name"]]
    half = tmp_path / "half.db"                               # schema creation interrupted after one table
    with sqlite3.connect(half) as con:
        con.execute("create table studies (study_id integer primary key, study_name text)")
        con.execute("insert into studies values (1, ?)", (src("01"),))
    assert opt.select_warm_start(args_for(tmp_path, study_db=half), META) == ([], [])


def test_readers_create_no_lock_file(tmp_path):
    make_source_study(tmp_path / "a.db", src("01"), [{"value": -0.2}])
    copy = tmp_path / "copy"
    copy.mkdir()
    shutil.copy2(tmp_path / "a.db", copy / "s.db")
    args = args_for(tmp_path, study_db=copy / "s.db")
    assert opt.select_warm_start(args, META)[0] and opt.resolve_study(args, META)["mode"] == "new_warm"
    assert sorted(p.name for p in copy.iterdir()) == ["s.db"]


def test_resume_warm_refuses_a_tampered_identity(tmp_path):
    args, resolved = warm_setup(tmp_path)
    study = opt.open_study(args, resolved["name"], resolved["identity"])
    study.set_user_attr("identity", {**resolved["identity"], "fee_bps": 6.0})
    with pytest.raises(ValueError, match="Study identity mismatch"):
        opt.resolve_study(args, META)


def results_entry(warm_start, name="S"):
    return {"symbol": "BTCUSDT", "bar_type": "fixed_dollar_imbalance", "study_name": name,
            "holdout_history": {}, "warm_start": warm_start}


def test_save_results_refuses_a_different_warm_start_under_the_same_name(tmp_path):
    seeds_a = {"max": 3, "seeds": [{"study": "A", "trial": 0, "value": -0.2, "params": valid_params()}]}
    seeds_b = {"max": 3, "seeds": [{"study": "B", "trial": 4, "value": -0.1, "params": valid_params()}]}
    cold, warm = tmp_path / "cold.json", tmp_path / "warm.json"
    opt.save_results(cold, {k: v for k, v in results_entry(None).items() if k != "warm_start"})   # legacy entry
    opt.save_results(cold, results_entry(None))                                 # same cold: overwrite
    before = cold.read_text()
    with pytest.raises(ValueError, match="S.*different warm start.*--output"):
        opt.save_results(cold, results_entry(seeds_a))                          # cold vs warm
    assert cold.read_text() == before
    opt.save_results(warm, results_entry(seeds_a))
    opt.save_results(warm, results_entry(seeds_a))                              # same seeds: overwrite
    with pytest.raises(ValueError, match="different warm start"):
        opt.save_results(warm, results_entry(seeds_b))
    opt.save_results(warm, results_entry(seeds_b, name="T"))                    # another study: fine
    assert set(entry_of_all(warm)) == {"S", "T"}


def entry_of_all(path):
    return json.loads(path.read_text())["symbols"]["BTCUSDT"]["fixed_dollar_imbalance"]


# -- fix wave 2 -------------------------------------------------------------------------------------------

def test_read_study_trials_sees_one_snapshot_while_a_trial_is_inserted(tmp_path, monkeypatch):
    db = tmp_path / "src.db"
    make_source_study(db, src("01"), [{"value": -0.2}])
    real_connect, fired = sqlite3.connect, []

    def insert_trial():           # another process committing a whole new trial (Optuna: one transaction)
        storage = optuna.storages.RDBStorage(f"sqlite:///{db}", engine_kwargs={"connect_args": {"timeout": 30}})
        optuna.load_study(study_name=src("01"), storage=storage).add_trial(
            source_trial(0.5, params=valid_params(min_composite_score=33.0)))

    class Interleaved:
        """Read-only connection that lets the insert run right after the reader's trial-id query."""
        def __init__(self, con):
            self.con = con

        def execute(self, sql, *a):
            if "from trial_values" in sql and not fired:
                fired.append(threading.Thread(target=insert_trial))
                fired[0].start()
                fired[0].join(timeout=1.0)    # completes at once unless the reader holds a read snapshot
            return self.con.execute(sql, *a)

        def __getattr__(self, name):
            return getattr(self.con, name)

    def connect(database, *a, **kw):
        con = real_connect(database, *a, **kw)
        return Interleaved(con) if "mode=ro" in str(database) else con
    monkeypatch.setattr(opt.sqlite3, "connect", connect)
    trials = opt.read_study_trials(db, src("01"))
    monkeypatch.undo()
    fired[0].join()
    assert [(t["number"], t["value"]) for t in trials] == [(0, -0.2)]          # the new trial wholly absent
    assert all(set(t["params"]) == set(SPACE) and t["attrs"] for t in trials)
    after = opt.read_study_trials(db, src("01"))
    assert [(t["number"], t["value"]) for t in after] == [(0, -0.2), (1, 0.5)]  # and wholly present afterwards
    assert set(after[1]["params"]) == set(SPACE) and after[1]["attrs"]["feasible"] is True


def test_duplicate_params_resolved_by_rank_before_the_cap(tmp_path):
    db = tmp_path / "src.db"
    p, q = valid_params(), valid_params(min_composite_score=21.0)
    make_source_study(db, src("a"), [{"value": 0.1, "params": p}])                      # first in DB order
    make_source_study(db, src("b"), [{"value": 1.0, "params": p}, {"value": 0.2, "params": q}])
    seeds, skipped = opt.select_warm_start(args_for(tmp_path, study_db=db, warm_start_max=1), META)
    assert [(s["study"], s["trial"], s["value"]) for s in seeds] == [(src("b"), 0, 1.0)]
    assert skipped == [{"study": src("a"), "trial": 0, "reason": "duplicate_params"},
                       {"study": src("a"), "trial": None, "reason": "no_eligible_trial"}]


def test_fallback_trial_never_takes_a_higher_ranked_duplicate(tmp_path):
    # Review repro: per-study ranking let B fall back to Q@0.05, so C's Q@0.8 became a duplicate and the
    # cap then cut Q entirely. Candidate-level ranking keeps Q@0.8 from C.
    db = tmp_path / "src.db"
    p, q, r = (valid_params(min_composite_score=v) for v in (20.0, 21.0, 22.0))
    make_source_study(db, src("a"), [{"value": 1.0, "params": p}])
    make_source_study(db, src("b"), [{"value": 0.9, "params": p}, {"value": 0.05, "params": q}])
    make_source_study(db, src("c"), [{"value": 0.8, "params": q}])
    make_source_study(db, src("d"), [{"value": 0.5, "params": r}])
    seeds, skipped = opt.select_warm_start(args_for(tmp_path, study_db=db, warm_start_max=2), META)
    assert [(s["study"], s["trial"], s["value"]) for s in seeds] == [(src("a"), 0, 1.0), (src("c"), 0, 0.8)]
    assert skipped == [{"study": src("b"), "trial": 0, "reason": "duplicate_params"},
                       {"study": src("b"), "trial": 1, "reason": "duplicate_params"},
                       {"study": src("b"), "trial": None, "reason": "no_eligible_trial"}]
    seeds, _ = opt.select_warm_start(args_for(tmp_path, study_db=db, warm_start_max=3), META)
    assert [s["value"] for s in seeds] == [1.0, 0.8, 0.5]                 # the cap only truncates


@pytest.mark.parametrize("order", [("00", "01"), ("01", "00")])
def test_duplicate_tie_break_is_deterministic(tmp_path, order):
    db = tmp_path / "src.db"
    p = valid_params()
    for tag in order:              # creation (DB) order must not matter: equal values -> study name decides
        make_source_study(db, src(tag), [{"value": 0.5, "params": p},
                                         {"value": 0.1, "params": valid_params(min_composite_score=20.0 + int(tag))}])
    seeds, skipped = opt.select_warm_start(args_for(tmp_path, study_db=db), META)
    assert [(s["study"], s["trial"]) for s in seeds] == [(src("00"), 0), (src("01"), 1)]
    assert skipped == [{"study": src("01"), "trial": 0, "reason": "duplicate_params"}]


def test_existing_read_only_init_lock_is_shared_not_appended(tmp_path):
    db = tmp_path / "s.db"
    make_source_study(db, src("01"), [{"value": -0.2}])
    lock = tmp_path / "s.db.init.lock"
    lock.touch()
    lock.chmod(0o444)                     # append-open fails even for the owner
    try:
        args = args_for(tmp_path, study_db=db)
        before = sorted(p.name for p in tmp_path.iterdir())
        assert opt.resolve_study(args, META)["mode"] == "new_warm"
        assert opt._study_names(db) == [src("01")]
        holder = os.open(lock, os.O_RDONLY)
        fcntl.flock(holder, fcntl.LOCK_EX)          # a schema creation in progress
        reader = threading.Thread(target=opt._study_names, args=(db,))
        reader.start()
        reader.join(timeout=0.5)
        assert reader.is_alive()                     # the reader honours the lock and waits
        os.close(holder)
        reader.join(timeout=5)
        assert not reader.is_alive()
        assert sorted(p.name for p in tmp_path.iterdir()) == before   # nothing created
    finally:
        lock.chmod(0o644)

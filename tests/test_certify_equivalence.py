"""Dataset/pipeline and Optuna research certificates (plan §3.4, R9, §21): four levels, gated on a passed
full-history alignment; only the research certificate reads Optuna storage."""
import hashlib
import json
import shutil
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from orderflow_system.data.models import BarType
from scripts import certify_equivalence as ce
from tests.conftest import STAMP, make_rows, write_dataset

BT = BarType.TICK_IMBALANCE
IDENTITY = {"dataset_id": "20261103T000000Z-abcdefabcdef", "root_hash": "ab" * 32}
ALIGNED = {"status": "passed", "full_history": True, **IDENTITY}
GAP = "2023-01-01T00:00:00+00:00"   # inside the synthetic bars' time range


STUDY = f"BTCUSDT__{BT.value}__2023-01-01_2023-01-01__d1__s1"
PARAMS = {"price_proximity_pct": 0.005, "min_composite_score": 30.0, "signal_cooldown_seconds": 30.0,
          "volume_decline_pct": 0.3, "lookback_bars": 10, "delta_failure_pct": 0.7}


def _results(path, start="2023-01-01", end="2023-01-01", **entry):
    path.write_text(json.dumps({"symbols": {"BTCUSDT": {BT.value: {STUDY: {
        "best_params": PARAMS, "best_trial_number": 0, "dataset": {"start": start, "end": end}, **entry}}}}}))
    return path


def _db(path, names, params=PARAMS, state="COMPLETE"):
    """Optuna's storage tables (the columns this repo reads); STUDY's trial 0 holds `params`."""
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE studies (study_id INTEGER PRIMARY KEY, study_name TEXT)")
        con.execute("CREATE TABLE trials (trial_id INTEGER PRIMARY KEY, number INTEGER, study_id INTEGER, state TEXT)")
        con.execute("CREATE TABLE trial_params (param_id INTEGER PRIMARY KEY, trial_id INTEGER, param_name TEXT, "
                    "param_value REAL, distribution_json TEXT)")
        for name in names:
            sid = con.execute("INSERT INTO studies (study_name) VALUES (?)", (name,)).lastrowid
            if name == STUDY:
                tid = con.execute("INSERT INTO trials (number, study_id, state) VALUES (0, ?, ?)", (sid, state)).lastrowid
                for k, v in params.items():
                    dist = {"name": "IntDistribution" if isinstance(v, int) else "FloatDistribution", "attributes": {}}
                    con.execute("INSERT INTO trial_params (trial_id, param_name, param_value, distribution_json) "
                                "VALUES (?, ?, ?, ?)", (tid, k, float(v), json.dumps(dist)))
    return path


def _mutate_db_trial(db, **params):
    db.unlink()
    _db(db, [STUDY], params=params)


@pytest.fixture
def roots(tmp_path):
    rows = make_rows(n=400, step_s=30)
    v0, v1 = tmp_path / "v0", tmp_path / "v1"
    write_dataset(v0, rows, bar_type=BT, missing_hours=[GAP])
    write_dataset(v1, rows, bar_type=BT, missing_hours=[GAP])
    results = _results(tmp_path / "results.json")
    db = _db(tmp_path / "studies.db", [STUDY, "BTCUSDT__fixed_dollar_imbalance__x"])   # other type: not required
    return v0, v1, results, db


def _certify(v0, v1, results, db, report=ALIGNED, **kw):
    return ce.certify_research(v0_root=v0, v0_version=STAMP, v1_root=v1, v1_identity=IDENTITY, alignment_report=report,
                      symbol="BTCUSDT", bar_types=[BT.value], results_path=results, studies_db=db, **kw)


def test_byte_identical_copy_is_strategy_equivalent_and_optuna_untouched(roots):
    v0, v1, results, db = roots
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (results, db)}
    cert = _certify(v0, v1, results, db)
    assert cert["verdict"] == "strategy-equivalent"
    t = cert["types"][BT.value]
    assert all(t[lvl]["passed"] for lvl in ("L1", "L2", "L3", "L4"))
    assert len(t["L4"]["configs"]) == 2                          # default + the stored study's best params
    assert t["dataset_ids_differ"] is True                       # different path -> new Optuna fingerprint
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (results, db)}
    assert before == after and cert["optuna_unchanged"] is True


def test_missing_tick_hours_change_fails_l2_and_l3(roots):
    """Codex counterexample: identical parquet, emptied missing_tick_hours flips data_gap."""
    v0, v1, results, db = roots
    side = next((v1 / "BTCUSDT" / BT.value).glob("*.json"))
    meta = json.loads(side.read_text())
    meta["missing_tick_hours"] = []
    side.write_text(json.dumps(meta))
    cert = _certify(v0, v1, results, db)
    t = cert["types"][BT.value]
    assert t["L1"]["passed"] and not t["L2"]["passed"] and not t["L3"]["passed"]
    assert "data_gap" in t["L3"]["first_mismatch"]
    assert cert["verdict"] == "bar-equivalent only"


def test_tie_order_perturbation_fails_l1_and_l3(roots, tmp_path):
    v0, v1, results, db = roots
    rows = make_rows(n=400, step_s=30)
    rows[10]["ts"] = rows[11]["ts"] = rows[11]["ts"]             # a tie...
    rows[10]["seq"], rows[11]["seq"] = rows[11]["seq"], rows[10]["seq"]   # ...in the other order
    rows[10]["nt"], rows[11]["nt"] = rows[11]["nt"], rows[10]["nt"]
    shutil.rmtree(v1)
    write_dataset(v1, rows, bar_type=BT, missing_hours=[GAP])
    t = _certify(v0, v1, results, db)["types"][BT.value]
    assert not t["L1"]["passed"] and not t["L3"]["passed"]


def test_replay_digest_difference_fails_l4(roots, monkeypatch):
    v0, v1, results, db = roots
    real = ce.replay_digests

    def skewed(root, *args, **kwargs):
        out = real(root, *args, **kwargs)
        return {**out, "digest": "0" * 64} if root == v1 else out
    monkeypatch.setattr(ce, "replay_digests", skewed)
    cert = _certify(v0, v1, results, db)
    assert not cert["types"][BT.value]["L4"]["passed"] and cert["verdict"] == "bar-equivalent only"


@pytest.mark.parametrize("report, match", [
    (None, "alignment report"),
    ({**ALIGNED, "status": "failed"}, "passed"),
    ({**ALIGNED, "full_history": False}, "full-history"),
    ({**ALIGNED, "root_hash": "cd" * 32}, "root hash"),
    ({**ALIGNED, "dataset_id": "other"}, "dataset"),
])
def test_certification_requires_matching_passed_full_history_alignment(roots, report, match):
    v0, v1, results, db = roots
    with pytest.raises(ce.CertificationRefused, match=match):
        _certify(v0, v1, results, db, report=report)


def test_full_study_coverage_records_exact_studies_and_configs(roots):
    v0, v1, results, db = roots
    cert = _certify(v0, v1, results, db)
    cov = cert["coverage"]
    assert cov["mode"] == "full-study" and cert["verdict"] == "strategy-equivalent"
    assert cov["studies"][BT.value] == [{"study": STUDY, "params": PARAMS, "params_source": "optuna_db_trial_0",
                                         "window": ["2023-01-01", "2023-01-02"]}]
    assert cov["db_studies"] == sorted([STUDY, "BTCUSDT__fixed_dollar_imbalance__x"])
    assert cov["uncataloged_db_studies"] == []
    assert [r["config"] for r in cert["types"][BT.value]["L4"]["configs"]] == ["default", STUDY]
    assert cert["types"][BT.value]["L4"]["configs"][1]["params"] == PARAMS


@pytest.mark.parametrize("break_it, match", [
    (lambda results, db: results.unlink(), "results catalog"),
    (lambda results, db: db.unlink(), "study DB"),
    (lambda results, db: db.write_bytes(b"not a database"), "study DB"),
    (lambda results, db: (db.unlink(), _db(db, [STUDY, f"BTCUSDT__{BT.value}__extra"])),
     "not in the results catalog"),
])
def test_full_study_mode_refuses_incomplete_coverage(roots, break_it, match):
    """A mistyped/missing catalog, missing DB or uncatalogued study must not shrink coverage silently."""
    v0, v1, results, db = roots
    break_it(results, db)
    with pytest.raises(ce.CertificationRefused, match=match):
        _certify(v0, v1, results, db)


@pytest.mark.parametrize("name, break_it, match", [
    # Codex: best_params=null used to replay the DEFAULT configuration and certify "full-study"
    ("null_params", lambda r, db: _results(r, best_params=None), "best_params"),
    ("catalog_params_differ_from_db", lambda r, db: _results(r, best_params={**PARAMS, "lookback_bars": 11}),
     "differ from the persisted trial"),
    ("catalog_missing_a_db_param", lambda r, db: _results(r, best_params={k: v for k, v in PARAMS.items()
                                                                          if k != "lookback_bars"}),
     "differ from the persisted trial"),
    ("db_trial_has_no_params", lambda r, db: _mutate_db_trial(db), "no persisted params"),
    ("missing_best_trial_number", lambda r, db: _results(r, best_trial_number=None), "best_trial_number"),
    ("best_trial_not_in_db", lambda r, db: _results(r, best_trial_number=7), "trial 7"),
    ("best_trial_not_complete", lambda r, db: (db.unlink(), _db(db, [STUDY], state="FAIL")), "not COMPLETE"),
    # Codex: an inverted window compared zero study bars and matching empty digests
    ("inverted_window", lambda r, db: _results(r, start="2023-01-02", end="2023-01-01"), "inverted"),
    ("malformed_window", lambda r, db: _results(r, start="2023-13-01"), "window"),
    ("window_differs_from_study_identity", lambda r, db: _results(r, start="2022-12-31"), "persisted study window"),
])
def test_full_study_certification_validates_configs_and_windows(roots, name, break_it, match):
    v0, v1, results, db = roots
    break_it(results, db)
    with pytest.raises(ce.CertificationRefused, match=match):
        _certify(v0, v1, results, db)


def test_required_replay_with_zero_bars_is_refused(tmp_path):
    """A valid, persisted study window that holds no bars cannot certify anything."""
    rows = make_rows(n=50, step_s=30)                       # all bars on 2023-01-01
    v0, v1 = tmp_path / "v0", tmp_path / "v1"
    write_dataset(v0, rows, bar_type=BT)
    write_dataset(v1, rows, bar_type=BT)
    study = f"BTCUSDT__{BT.value}__2024-06-01_2024-06-02__d1__s1"
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"symbols": {"BTCUSDT": {BT.value: {study: {
        "best_params": PARAMS, "best_trial_number": 0, "dataset": {"start": "2024-06-01", "end": "2024-06-02"}}}}}}))
    db = _db(tmp_path / "studies.db", [STUDY])
    with sqlite3.connect(db) as con:
        con.execute("UPDATE studies SET study_name = ?", (study,))
    with pytest.raises(ce.CertificationRefused, match="no bars"):
        _certify(v0, v1, results, db)


def test_replay_uses_the_persisted_trial_params(roots, monkeypatch):
    v0, v1, results, db = roots
    seen = []
    real = ce.replay_digests

    def spy(root, version, symbol, bar_type, window, params):
        seen.append(params)
        return real(root, version, symbol, bar_type, window, params)
    monkeypatch.setattr(ce, "replay_digests", spy)
    cert = _certify(v0, v1, results, db)
    study_params = [p for p in seen if p is not None]
    assert study_params == [PARAMS, PARAMS]                                  # v0 and v1, from the DB trial
    assert isinstance(study_params[0]["lookback_bars"], int)                 # IntDistribution decoded as int
    assert cert["coverage"]["studies"][BT.value][0]["params_source"] == "optuna_db_trial_0"


# ── per-bucket coverage over several bar types (Codex: a tick study relocated into the volume bucket passed) ──

BT2 = BarType.VOLUME_IMBALANCE
STUDY2 = f"BTCUSDT__{BT2.value}__2023-01-01_2023-01-01__d2__s2"
PARAMS2 = {**PARAMS, "lookback_bars": 12, "min_composite_score": 40.0}
WINDOW = {"start": "2023-01-01", "end": "2023-01-01"}


def _entry(study, params, **extra):
    bar_type = study.split("__")[1]
    return {"study_name": study, "symbol": "BTCUSDT", "bar_type": bar_type, "best_params": params,
            "best_trial_number": 0, "dataset": dict(WINDOW), **extra}


def _db2(path, studies):
    """studies: {name: params}; every listed study gets a COMPLETE trial 0."""
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE studies (study_id INTEGER PRIMARY KEY, study_name TEXT)")
        con.execute("CREATE TABLE trials (trial_id INTEGER PRIMARY KEY, number INTEGER, study_id INTEGER, state TEXT)")
        con.execute("CREATE TABLE trial_params (param_id INTEGER PRIMARY KEY, trial_id INTEGER, param_name TEXT, "
                    "param_value REAL, distribution_json TEXT)")
        for name, params in studies.items():
            sid = con.execute("INSERT INTO studies (study_name) VALUES (?)", (name,)).lastrowid
            tid = con.execute("INSERT INTO trials (number, study_id, state) VALUES (0, ?, 'COMPLETE')", (sid,)).lastrowid
            for k, v in params.items():
                dist = {"name": "IntDistribution" if isinstance(v, int) else "FloatDistribution"}
                con.execute("INSERT INTO trial_params (trial_id, param_name, param_value, distribution_json) "
                            "VALUES (?, ?, ?, ?)", (tid, k, float(v), json.dumps(dist)))
    return path


@pytest.fixture
def two_types(tmp_path):
    rows = make_rows(n=300, step_s=30)
    v0, v1 = tmp_path / "v0", tmp_path / "v1"
    for bt in (BT, BT2):
        write_dataset(v0, rows, bar_type=bt)
        write_dataset(v1, rows, bar_type=bt)
    db = _db2(tmp_path / "studies.db", {STUDY: PARAMS, STUDY2: PARAMS2})
    return v0, v1, tmp_path / "results.json", db


def _catalog(path, buckets):
    path.write_text(json.dumps({"symbols": {"BTCUSDT": buckets}}))
    return path


def _certify2(v0, v1, results, db, **kw):
    return ce.certify_research(v0_root=v0, v0_version=STAMP, v1_root=v1, v1_identity=IDENTITY, alignment_report=ALIGNED,
                      symbol="BTCUSDT", bar_types=[BT.value, BT2.value], results_path=results, studies_db=db, **kw)


def test_each_study_is_replayed_on_its_own_bar_type_with_its_own_config(two_types, monkeypatch):
    v0, v1, results, db = two_types
    _catalog(results, {BT.value: {STUDY: _entry(STUDY, PARAMS)}, BT2.value: {STUDY2: _entry(STUDY2, PARAMS2)}})
    calls, real = [], ce.replay_digests

    def spy(root, version, symbol, bar_type, window, params):
        calls.append((symbol, bar_type, window, params))
        return real(root, version, symbol, bar_type, window, params)
    monkeypatch.setattr(ce, "replay_digests", spy)
    cert = _certify2(v0, v1, results, db)
    assert cert["verdict"] == "strategy-equivalent" and cert["coverage"]["mode"] == "full-study"
    study_calls = {(bt, tuple(w), json.dumps(p, sort_keys=True)) for s, bt, w, p in calls if p is not None}
    assert study_calls == {(BT.value, ("2023-01-01", "2023-01-02"), json.dumps(PARAMS, sort_keys=True)),
                           (BT2.value, ("2023-01-01", "2023-01-02"), json.dumps(PARAMS2, sort_keys=True))}
    assert all(s == "BTCUSDT" for s, *_ in calls)
    assert cert["coverage"]["buckets"] == {
        BT.value: {"required": [STUDY], "covered": [STUDY], "uncataloged": []},
        BT2.value: {"required": [STUDY2], "covered": [STUDY2], "uncataloged": []}}
    assert [r["config"] for r in cert["types"][BT.value]["L4"]["configs"]] == ["default", STUDY]
    assert [r["config"] for r in cert["types"][BT2.value]["L4"]["configs"]] == ["default", STUDY2]


@pytest.mark.parametrize("name, buckets, match", [
    # Codex's reproduction: the tick study moved into the volume bucket, tick bucket left with defaults only
    ("tick_study_in_volume_bucket", lambda: {BT.value: {}, BT2.value: {STUDY2: _entry(STUDY2, PARAMS2),
                                                                      STUDY: _entry(STUDY, PARAMS)}},
     "persisted identity"),
    ("tick_study_parked_in_an_uncertified_bucket", lambda: {
        BT2.value: {STUDY2: _entry(STUDY2, PARAMS2)}, "fixed_dollar_imbalance": {STUDY: _entry(STUDY, PARAMS)}},
     "persisted identity|not in the results catalog"),
    ("study_in_two_buckets", lambda: {BT.value: {STUDY: _entry(STUDY, PARAMS)},
                                      BT2.value: {STUDY2: _entry(STUDY2, PARAMS2), STUDY: _entry(STUDY, PARAMS)}},
     "more than one catalog bucket"),
    ("tick_study_missing", lambda: {BT.value: {}, BT2.value: {STUDY2: _entry(STUDY2, PARAMS2)}},
     "not in the results catalog"),
    ("entry_claims_another_bar_type", lambda: {
        BT.value: {STUDY: _entry(STUDY, PARAMS, bar_type=BT2.value)}, BT2.value: {STUDY2: _entry(STUDY2, PARAMS2)}},
     "entry bar_type"),
    ("entry_claims_another_study_name", lambda: {
        BT.value: {STUDY: _entry(STUDY, PARAMS, study_name=STUDY2)}, BT2.value: {STUDY2: _entry(STUDY2, PARAMS2)}},
     "entry study_name"),
])
def test_misplaced_missing_or_ambiguous_studies_are_refused(two_types, name, buckets, match):
    v0, v1, results, db = two_types
    _catalog(results, buckets())
    with pytest.raises(ce.CertificationRefused, match=match):
        _certify2(v0, v1, results, db)


def test_unparseable_persisted_study_name_is_refused_in_full_mode(two_types):
    v0, v1, results, db = two_types
    _catalog(results, {BT.value: {STUDY: _entry(STUDY, PARAMS)}, BT2.value: {STUDY2: _entry(STUDY2, PARAMS2)}})
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO studies (study_name) VALUES ('adhoc-study')")
    with pytest.raises(ce.CertificationRefused, match="adhoc-study"):
        _certify2(v0, v1, results, db)


def test_l1_is_bit_exact(roots):
    v0, v1, _, _ = roots
    path = next((v1 / "BTCUSDT" / BT.value).glob("*.parquet"))
    table = pq.read_table(path)
    vol = table.column("Cumulative Volume").to_numpy().copy()
    vol[5] = vol[5] + 1e-12                                       # within any tolerance, still a different bar
    table = table.set_column(table.schema.get_field_index("Cumulative Volume"), "Cumulative Volume", pa.array(vol))
    pq.write_table(table, path)
    assert not ce.level1(next((v0 / "BTCUSDT" / BT.value).glob("*.parquet")), path)["passed"]


# ── dataset/pipeline certification: never reads Optuna storage (plan §21) ──────────────────────────────────

DATASET_PASS = "dataset-reproducible; pipeline-equivalent (default config)"


def _certify_dataset(v0, v1, report=ALIGNED, bar_types=(BT.value,)):
    return ce.certify_dataset(v0_root=v0, v0_version=STAMP, v1_root=v1, v1_identity=IDENTITY,
                              alignment_report=report, symbol="BTCUSDT", bar_types=list(bar_types))


def test_dataset_certification_needs_no_optuna_files(roots):
    v0, v1, results, db = roots
    results.unlink()
    db.unlink()
    cert = _certify_dataset(v0, v1)
    assert cert["certification"] == "dataset-pipeline" and cert["verdict"] == DATASET_PASS
    assert cert["dataset_reproducible"] is True and cert["pipeline_equivalent_default"] is True
    t = cert["types"][BT.value]
    assert all(t[lvl]["passed"] for lvl in ("L1", "L2", "L3", "L4"))
    assert t["L3"]["windows"] == 1                                         # the full range only
    assert [r["config"] for r in t["L4"]["configs"]] == ["default"]
    assert cert["not_certified"] == {"optuna_research": "not certified: run --mode research",
                                     "strategy_profitability": "not validated by any certificate"}
    assert not {"coverage", "optuna_files", "optuna_unchanged"} & set(cert)


def test_dataset_certification_never_opens_optuna_storage(roots, monkeypatch):
    v0, v1, _, _ = roots

    def forbidden(*a, **k):
        raise AssertionError("dataset certification opened an Optuna study DB")
    monkeypatch.setattr(ce.sqlite3, "connect", forbidden)
    assert _certify_dataset(v0, v1)["verdict"] == DATASET_PASS


@pytest.mark.parametrize("report, match", [
    (None, "alignment report"),
    ({**ALIGNED, "status": "failed"}, "passed"),
    ({**ALIGNED, "full_history": False}, "full-history"),
    ({**ALIGNED, "root_hash": "cd" * 32}, "root hash"),
    ({**ALIGNED, "dataset_id": "other"}, "dataset"),
])
def test_dataset_certification_requires_matching_passed_full_history_alignment(roots, report, match):
    v0, v1, _, _ = roots
    with pytest.raises(ce.CertificationRefused, match=match):
        _certify_dataset(v0, v1, report=report)


def _empty_gap_hours(v0, v1, tmp_path, monkeypatch):
    side = next((v1 / "BTCUSDT" / BT.value).glob("*.json"))
    side.write_text(json.dumps({**json.loads(side.read_text()), "missing_tick_hours": []}))


def _swap_tie(v0, v1, tmp_path, monkeypatch):
    rows = make_rows(n=400, step_s=30)
    rows[10]["ts"] = rows[11]["ts"] = rows[11]["ts"]
    rows[10]["seq"], rows[11]["seq"] = rows[11]["seq"], rows[10]["seq"]
    rows[10]["nt"], rows[11]["nt"] = rows[11]["nt"], rows[10]["nt"]
    shutil.rmtree(v1)
    write_dataset(v1, rows, bar_type=BT, missing_hours=[GAP])


def _skew_v1_replays(v0, v1, tmp_path, monkeypatch):
    real = ce.replay_digests

    def skewed(root, *args, **kwargs):
        out = real(root, *args, **kwargs)
        return {**out, "digest": "0" * 64} if root == v1 else out
    monkeypatch.setattr(ce, "replay_digests", skewed)


@pytest.mark.parametrize("perturb, failing, reproducible", [
    (_empty_gap_hours, {"L2", "L3"}, False),
    (_swap_tie, {"L1", "L3", "L4"}, False),                     # the reordered bars also replay differently
    (_skew_v1_replays, {"L4"}, True),
])
def test_dataset_failures_are_never_hidden_by_mode_or_label(roots, tmp_path, monkeypatch, perturb, failing,
                                                            reproducible):
    v0, v1, results, db = roots
    perturb(v0, v1, tmp_path, monkeypatch)
    data, research = _certify_dataset(v0, v1), _certify(v0, v1, results, db)
    for cert in (data, research):
        t = cert["types"][BT.value]
        assert {lvl for lvl in ("L1", "L2", "L3", "L4") if not t[lvl]["passed"]} == failing
        assert cert["dataset_reproducible"] is reproducible and cert["pipeline_equivalent_default"] is False
        assert "pipeline-equivalent" not in cert["verdict"] and "strategy-equivalent" not in cert["verdict"]
        assert cert["verdict"] in ("bar-equivalent only", "different")
    for lvl in ("L1", "L2", "L3"):
        assert data["types"][BT.value][lvl]["passed"] == research["types"][BT.value][lvl]["passed"]
    assert data["types"][BT.value]["L4"]["configs"][0] == research["types"][BT.value]["L4"]["configs"][0]


def test_research_certificate_is_labelled_as_optuna_research_not_profitability(roots):
    v0, v1, results, db = roots
    cert = _certify(v0, v1, results, db)
    assert cert["certification"] == "optuna-research" and cert["verdict"] == "strategy-equivalent"
    assert cert["dataset_reproducible"] is True and cert["pipeline_equivalent_default"] is True
    assert cert["not_certified"] == {"strategy_profitability": "not validated by any certificate"}


def test_research_refuses_an_unfinished_uncatalogued_study(roots):
    """A stopped study (running/failed trials, never catalogued) still blocks RESEARCH certification."""
    v0, v1, results, db = roots
    stopped = f"BTCUSDT__{BT.value}__2023-01-01_2023-01-01__d9__s9"
    with sqlite3.connect(db) as con:
        sid = con.execute("INSERT INTO studies (study_name) VALUES (?)", (stopped,)).lastrowid
        con.execute("INSERT INTO trials (number, study_id, state) VALUES (0, ?, 'RUNNING')", (sid,))
        con.execute("INSERT INTO trials (number, study_id, state) VALUES (1, ?, 'FAIL')", (sid,))
    with pytest.raises(ce.CertificationRefused, match=stopped):
        _certify(v0, v1, results, db)
    assert _certify_dataset(v0, v1)["verdict"] == DATASET_PASS             # ...but never the dataset certificate


# ── CLI: --mode, --expected-root, output names (plan §21) ─────────────────────────────────────────────────

from tests import paired_fixture as fx   # noqa: E402

PBT = "fixed_tick_imbalance"


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """A published paired dataset, a v0 copy of its bars, a passed alignment report, and a stopped,
    uncatalogued study in an Optuna DB with no results catalog."""
    trades = fx.make_trades(days=3, per_day=600)
    ds_dir, root_hash = fx.write_paired(tmp_path, trades, {PBT: list(range(37, len(trades) - 5, 37))})
    v0 = tmp_path / "v0"
    shutil.copytree(ds_dir / fx.SYMBOL, v0 / fx.SYMBOL)
    report = tmp_path / "alignment.json"
    report.write_text(json.dumps({"status": "passed", "full_history": True, "dataset_id": ds_dir.name,
                                  "root_hash": root_hash}))
    db = tmp_path / "studies.db"
    _db(db, [])
    with sqlite3.connect(db) as con:
        sid = con.execute("INSERT INTO studies (study_name) VALUES (?)",
                          (f"BTCUSDT__{PBT}__2023-01-01_2023-01-03__d9__s9",)).lastrowid
        con.execute("INSERT INTO trials (number, study_id, state) VALUES (0, ?, 'RUNNING')", (sid,))
    registry = tmp_path / "registry.json"
    registry.write_text("{}")
    monkeypatch.setattr(ce, "REGISTRY_PATH", registry)
    monkeypatch.setattr(ce, "CERT_DIR", tmp_path / "certs")
    base = ["--paired-root", str(ds_dir.parent), "--dataset-id", ds_dir.name, "--alignment-report", str(report),
            "--v0-root", str(v0), "--v0-version", fx.STAMP,
            "--results", str(tmp_path / "results.json"), "--studies-db", str(db)]
    return {"base": base, "root": root_hash, "id": ds_dir.name, "db": db, "registry": registry,
            "certs": tmp_path / "certs", "results": tmp_path / "results.json"}


def _sha_or_none(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def test_cli_dataset_mode_is_the_default_and_ignores_unfinished_optuna_studies(cli):
    before = {p: _sha_or_none(p) for p in (cli["db"], cli["results"], cli["registry"])}
    ce.main([*cli["base"], "--expected-root", cli["root"]])
    cert = json.loads((cli["certs"] / f"dataset-{cli['id']}.json").read_text())
    assert cert["certification"] == "dataset-pipeline" and cert["verdict"] == DATASET_PASS
    assert cert["pin_source"] == "--expected-root" and cert["v1"]["root_hash"] == cli["root"]
    assert not (cli["certs"] / f"research-{cli['id']}.json").exists()
    assert {p: _sha_or_none(p) for p in before} == before                  # Optuna files and registry untouched


def test_cli_research_mode_stays_strict(cli):
    with pytest.raises(ce.CertificationRefused, match="results catalog"):
        ce.main([*cli["base"], "--expected-root", cli["root"], "--mode", "research"])
    cli["results"].write_text(json.dumps({"symbols": {"BTCUSDT": {PBT: {}}}}))
    with pytest.raises(ce.CertificationRefused, match="not in the results catalog"):
        ce.main([*cli["base"], "--expected-root", cli["root"], "--mode", "research"])
    assert not cli["certs"].exists() or not list(cli["certs"].iterdir())


@pytest.mark.parametrize("pin, error", [
    ("cd" * 32, "root hash"),                 # well-formed, wrong
    ("", "expected-root"),                    # empty must not fall back to the registry
    ("abc", "expected-root"),                 # not a SHA-256
])
def test_cli_expected_root_must_match_the_dataset(cli, pin, error):
    from orderflow_system.data.infobar_feed import DataIntegrityError
    with pytest.raises((DataIntegrityError, ce.CertificationRefused, SystemExit), match=error):
        ce.main([*cli["base"], "--expected-root", pin])
    assert not cli["certs"].exists()


def test_cli_without_a_trusted_root_refuses(cli):
    from orderflow_system.data.infobar_feed import DataIntegrityError
    with pytest.raises(DataIntegrityError, match="not registered"):
        ce.main(cli["base"])


def test_cli_expected_root_conflicting_with_the_registry_refuses(cli):
    cli["registry"].write_text(json.dumps({cli["id"]: "cd" * 32}))
    with pytest.raises(ce.CertificationRefused, match="registry"):
        ce.main([*cli["base"], "--expected-root", cli["root"]])


def test_cli_registered_pin_is_verified_and_recorded(cli):
    cli["registry"].write_text(json.dumps({cli["id"]: cli["root"]}))
    ce.main(cli["base"])
    cert = json.loads((cli["certs"] / f"dataset-{cli['id']}.json").read_text())
    assert cert["pin_source"] == "registry" and cert["verdict"] == DATASET_PASS


def test_cli_limited_coverage_flag_is_gone(cli):
    with pytest.raises(SystemExit):
        ce.main([*cli["base"], "--expected-root", cli["root"], "--limited-coverage"])


@pytest.mark.parametrize("mode", ["dataset", "research"])
def test_certifying_no_bar_types_is_refused(roots, mode):
    """all([]) is True: an empty bar-type list must not yield a passing label with nothing compared."""
    v0, v1, results, db = roots
    with pytest.raises(ce.CertificationRefused, match="no bar types"):
        if mode == "dataset":
            _certify_dataset(v0, v1, bar_types=())
        else:
            ce.certify_research(v0_root=v0, v0_version=STAMP, v1_root=v1, v1_identity=IDENTITY,
                                alignment_report=ALIGNED, symbol="BTCUSDT", bar_types=[], results_path=results,
                                studies_db=db)

"""Certify a paired dataset (v1) against v0: dataset/pipeline certificate, or Optuna research certificate.

Read-only. Both refuse to run unless a passed, full-history trade-alignment report for the same dataset id
and root hash is given (scripts/verify_paired_dataset.py). Then, per bar type:

  L1 bars       all parquet columns, all rows, bit-exact
  L2 metadata   sidecar fields the reader/pipeline consume (symbol, config.bar_type, missing_tick_hours);
                other sidecar differences are listed as informational
  L3 reader     InfoBarReader output (incl. duration_ms, data_gap) over the full range [+ study windows]
  L4 replay     legacy ReplayDigest (+ prefix digests) for the default config [+ every stored study's
                best params over its window]

--mode dataset (default): "Are the bars reproducible and processed identically by the pipeline?"
  Never opens the Optuna results catalog or study DB. L3 over the full range, L4 default config only.
  Verdict "dataset-reproducible; pipeline-equivalent (default config)" (all levels pass for all types),
  "bar-equivalent only" (L1 passes everywhere, a later level fails), else "different". It certifies
  neither Optuna research results nor strategy profitability (`not_certified` says so).

--mode research: "Do the stored Optuna studies replay identically?" Adds every catalogued study's window
  (L3) and best params (L4). Refuses when the results catalog or the study DB is missing or unreadable,
  or when the DB holds a study of a certified symbol/bar type that the catalog lacks (finished or not):
  coverage never shrinks silently. Verdict "strategy-equivalent" / "bar-equivalent only" / "different".
  The Optuna files are hashed before and after and must be unchanged; no study is resumed, relabelled or
  mutated. v1 always has a new DatasetInfo.dataset_id (path-based), so Optuna treats it as new data.

The dataset root hash is pinned by the registry (config/paired_datasets.json) or, before registration, by
--expected-root (a trusted 64-hex SHA-256 that must also agree with any registry entry). Certification
never writes the registry. Output: output/certificates/<mode>-<id>.json.

    python scripts/certify_equivalence.py --paired-root <root> --dataset-id <id> \
        --alignment-report <alignment-<id>.json> [--expected-root <sha256>] [--mode dataset|research]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orderflow_system.config.settings import INFO_BARS  # noqa: E402
from orderflow_system.data.infobar_feed import DataIntegrityError, find_dataset, open_bars  # noqa: E402
from orderflow_system.data.paired_dataset import (  # noqa: E402
    REGISTRY_PATH, PairedDataset, open_registered, pinned_root)
from orderflow_system.pipeline import InfoBarPipeline, ReplayDigest  # noqa: E402
from scripts.optimize_orderflow import (  # noqa: E402
    baseline_extra_params, build_instrument_config, instrument_config, run_backtest)

CONSUMED_SIDECAR = ("symbol", "config.bar_type", "missing_tick_hours")
PREFIX_ENDS = tuple(range(99_999, 50_000_000, 100_000))
CERT_DIR = ROOT / "output" / "certificates"
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
DATASET_PASS = "dataset-reproducible; pipeline-equivalent (default config)"
NOT_PROFITABILITY = "not validated by any certificate"
LIMITATION = "Proves v1 reproduces v0 bars and behaviour; not v0's build-time is_buyer_maker (plan §3.1)."


class CertificationRefused(RuntimeError):
    """A precondition for certification is not met; nothing was compared."""


def _get(d: dict, dotted: str):
    for key in dotted.split("."):
        d = d.get(key) if isinstance(d, dict) else None
    return d


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def level1(v0: Path, v1: Path) -> dict:
    a, b = pq.read_table(v0), pq.read_table(v1)
    if a.schema != b.schema or a.num_rows != b.num_rows:
        return {"passed": False, "first_mismatch": f"schema/rows {a.schema.names}/{a.num_rows} vs {b.schema.names}/{b.num_rows}"}
    for name in a.schema.names:
        x, y = a.column(name).to_numpy(), b.column(name).to_numpy()
        if x.dtype.kind in "fiM":
            x, y = x.view(np.uint64 if x.dtype.itemsize == 8 else x.dtype), y.view(np.uint64 if y.dtype.itemsize == 8 else y.dtype)
        bad = x != y
        if bad.any():
            return {"passed": False, "first_mismatch": f"{name} row {int(np.argmax(bad))}"}
    return {"passed": True, "rows": a.num_rows}


def level2(side0: dict, side1: dict) -> dict:
    consumed = {k: (_get(side0, k), _get(side1, k)) for k in CONSUMED_SIDECAR}
    diffs = sorted(k for k, (x, y) in consumed.items() if x != y)
    info = sorted(k for k in set(side0) | set(side1) if k not in {"symbol", "config", "missing_tick_hours"}
                  and side0.get(k) != side1.get(k))
    return {"passed": not diffs, "consumed_diffs": diffs, "informational_diffs": info}


def _bar_tuple(bar) -> tuple:
    return (bar.timestamp_ms, bar.seq, bar.open, bar.high, bar.low, bar.close, bar.volume, bar.buy_volume,
            bar.sell_volume, bar.tick_count, bar.dollar_value, bar.threshold, bar.duration_ms, bar.data_gap)


_FIELDS = ("timestamp_ms", "seq", "open", "high", "low", "close", "volume", "buy_volume", "sell_volume",
           "tick_count", "dollar_value", "threshold", "duration_ms", "data_gap")


def level3(roots: tuple[Path, Path], versions: tuple, symbol: str, bar_type: str, windows: list) -> dict:
    checked = 0
    for start, end in windows:
        in_window = 0
        try:
            r0 = open_bars(roots[0], symbol, bar_type, start, end, version=versions[0])
            r1 = open_bars(roots[1], symbol, bar_type, start, end, version=versions[1])
            for i, (b0, b1) in enumerate(_zip_strict(r0, r1)):
                t0, t1 = _bar_tuple(b0), _bar_tuple(b1)
                if t0 != t1:
                    field = next(f for f, x, y in zip(_FIELDS, t0, t1) if x != y)
                    return {"passed": False, "first_mismatch": f"window {start}..{end} bar {i}: {field} {t0} vs {t1}"}
                in_window += 1
        except DataIntegrityError as exc:
            return {"passed": False, "first_mismatch": f"window {start}..{end}: reader refused: {exc}"}
        if not in_window:   # an empty window would "match" trivially and certify nothing
            raise CertificationRefused(f"{bar_type}: required window {start}..{end} contains no bars")
        checked += in_window
    return {"passed": True, "bars_compared": checked, "windows": len(windows)}


def _zip_strict(a, b):
    sentinel = object()
    for x, y in __import__("itertools").zip_longest(a, b, fillvalue=sentinel):
        if x is sentinel or y is sentinel:
            raise DataIntegrityError("bar streams differ in length")
        yield x, y


def replay_digests(root: Path, version, symbol: str, bar_type: str, window: tuple, params: dict | None) -> dict:
    counted = {"n": 0}

    def bars():
        for bar in open_bars(root, symbol, bar_type, window[0], window[1], version=version):
            counted["n"] += 1
            yield bar
    if params is None:   # default configuration, plain pipeline
        pipe, digest, prefix = InfoBarPipeline(instrument_config(symbol)), ReplayDigest(), {}
        for idx, bar in enumerate(bars()):
            digest.update(pipe.process_bar(bar))
            if idx in PREFIX_ENDS:
                prefix[idx] = digest.hexdigest()
        return {"digest": digest.hexdigest(), "prefix": prefix, "n_bars": counted["n"]}
    full = {**baseline_extra_params(symbol), **params}
    out = run_backtest(bars(), build_instrument_config(full, symbol), full["price_proximity_pct"], bar_type,
                       prefix_ends=PREFIX_ENDS)
    return {"digest": out["replay_digest"], "prefix": out["prefix_digests"], "n_bars": counted["n"]}


def level4(roots, versions, symbol, bar_type, configs: list[tuple[str, dict | None, tuple]]) -> dict:
    rows = []
    for name, params, window in configs:
        row = {"config": name, "params": params, "window": list(window)}
        try:
            d0 = replay_digests(roots[0], versions[0], symbol, bar_type, window, params)
            d1 = replay_digests(roots[1], versions[1], symbol, bar_type, window, params)
        except DataIntegrityError as exc:   # a stream the reader refuses cannot be strategy-equivalent
            rows.append({**row, "equal": False, "error": str(exc)})
            continue
        if not d0["n_bars"] or not d1["n_bars"]:   # equal empty digests would certify nothing
            raise CertificationRefused(f"{bar_type}: required replay {name} over {window} has no bars "
                                       f"(v0 {d0['n_bars']}, v1 {d1['n_bars']})")
        rows.append({**row, "v0": d0["digest"], "v1": d1["digest"], "n_bars": d0["n_bars"], "equal": d0 == d1})
    return {"passed": all(r["equal"] for r in rows), "configs": rows}


_INT_DISTRIBUTIONS = ("IntDistribution", "IntUniformDistribution", "IntLogUniformDistribution")
_FLOAT_DISTRIBUTIONS = ("FloatDistribution", "UniformDistribution", "LogUniformDistribution",
                        "DiscreteUniformDistribution")


def _persisted_params(con: sqlite3.Connection, study: str, number) -> dict:
    """The params Optuna persisted for trial `number` of `study`: the authoritative study configuration."""
    if not isinstance(number, int) or isinstance(number, bool):
        raise CertificationRefused(f"{study}: catalog best_trial_number {number!r} is not a trial number")
    row = con.execute("SELECT t.trial_id, t.state FROM trials t JOIN studies s ON s.study_id = t.study_id "
                      "WHERE s.study_name = ? AND t.number = ?", (study, number)).fetchone()
    if row is None:
        raise CertificationRefused(f"{study}: trial {number} is not in the study DB")
    if row[1] != "COMPLETE":
        raise CertificationRefused(f"{study}: trial {number} is {row[1]}, not COMPLETE")
    params = {}
    for name, value, dist in con.execute(
            "SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id = ?", (row[0],)):
        kind = json.loads(dist).get("name")
        if kind in _INT_DISTRIBUTIONS:
            params[name] = int(value)
        elif kind in _FLOAT_DISTRIBUTIONS:
            params[name] = float(value)
        else:
            raise CertificationRefused(f"{study}: param {name} has unsupported distribution {kind!r}")
    if not params:
        raise CertificationRefused(f"{study}: trial {number} has no persisted params")
    return params


def _study_identity(name: str) -> tuple[str, str]:
    """(symbol, bar_type) persisted in an optimizer study name `SYM__bar_type__START_END__d…__s…`."""
    parts = name.split("__")
    if len(parts) < 3 or not parts[0] or not parts[1]:
        raise CertificationRefused(f"study {name!r}: name does not carry a SYM__bar_type__START_END identity")
    return parts[0], parts[1]


def _persisted_study_configs(catalog: dict, con: sqlite3.Connection, symbol: str, bar_type: str) -> list[tuple]:
    """(study, persisted params, replay window, trial number) per catalogued study, each checked against the DB."""
    out = []
    for name, entry in sorted(catalog.get("symbols", {}).get(symbol, {}).get(bar_type, {}).items()):
        entry = entry if isinstance(entry, dict) else {}
        if _study_identity(name) != (symbol, bar_type):   # never replay a study on another type's bars
            raise CertificationRefused(f"{name}: persisted identity {_study_identity(name)} is catalogued under "
                                       f"({symbol!r}, {bar_type!r})")
        for field, want in (("study_name", name), ("symbol", symbol), ("bar_type", bar_type)):
            if field in entry and entry[field] != want:
                raise CertificationRefused(f"{name}: catalog entry {field} {entry[field]!r} != {want!r}")
        params = entry.get("best_params")
        if not isinstance(params, dict) or not params:
            raise CertificationRefused(f"{name}: catalog best_params {params!r} is not a parameter set")
        number = entry.get("best_trial_number")
        persisted = _persisted_params(con, name, number)
        if params != persisted:
            diff = sorted(k for k in set(params) | set(persisted) if params.get(k) != persisted.get(k))
            raise CertificationRefused(f"{name}: catalog best_params differ from the persisted trial {number}: {diff}")
        ds = entry.get("dataset") if isinstance(entry.get("dataset"), dict) else {}
        try:
            start, end = date.fromisoformat(ds["start"]), date.fromisoformat(ds["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CertificationRefused(f"{name}: catalog window {ds.get('start')}..{ds.get('end')} is invalid ({exc})")
        if start > end:
            raise CertificationRefused(f"{name}: catalog window {start}..{end} is inverted")
        parts = name.split("__")
        persisted_window = parts[2] if len(parts) > 2 else None   # optimizer study names embed START_END
        if persisted_window != f"{start}_{end}":
            raise CertificationRefused(f"{name}: catalog window {start}_{end} differs from the persisted study "
                                       f"window {persisted_window}")
        out.append((name, persisted, (start.isoformat(), (end + timedelta(days=1)).isoformat()), number))
    return out


def _db_studies(studies_db: Path) -> list[str] | None:
    """Study names in the Optuna DB (read-only); None if the DB does not exist; unreadable refuses."""
    if not studies_db.exists():
        return None
    try:
        with sqlite3.connect(f"file:{studies_db}?mode=ro", uri=True) as con:
            return sorted(r[0] for r in con.execute("SELECT study_name FROM studies"))
    except sqlite3.DatabaseError as exc:
        raise CertificationRefused(f"study DB {studies_db} is unreadable: {exc}") from exc


def _coverage(results_path: Path, studies_db: Path, symbol: str, bar_types: list[str]):
    """Which stored studies are certified (research mode). Refuses anything that would shrink coverage."""
    catalog = json.loads(results_path.read_text()) if results_path.is_file() else None
    if catalog is None:
        raise CertificationRefused(f"results catalog {results_path} is missing; research certification needs it "
                                   "(--mode dataset certifies the bars and default-config pipeline without Optuna)")
    db = _db_studies(studies_db)
    if db is None:
        raise CertificationRefused(f"study DB {studies_db} is missing; full-study coverage cannot be established")
    persisted = {bt: [] for bt in bar_types}
    if catalog:
        placements: dict[str, list[str]] = {}
        for sym, buckets in catalog.get("symbols", {}).items():
            for bt, studies in (buckets or {}).items():
                for name in studies or {}:
                    placements.setdefault(name, []).append(f"{sym}/{bt}")
        ambiguous = {n: where for n, where in placements.items() if len(where) > 1}
        if ambiguous:
            raise CertificationRefused(f"studies placed in more than one catalog bucket: {ambiguous}")
        with sqlite3.connect(f"file:{studies_db}?mode=ro", uri=True) as con:
            persisted = {bt: _persisted_study_configs(catalog, con, symbol, bt) for bt in bar_types}
    configs = {bt: [(n, p, w) for n, p, w, _ in rows] for bt, rows in persisted.items()}
    # Required vs covered PER (symbol, bar_type) bucket: a study counts only where its persisted identity says.
    identities = {n: _study_identity(n) for n in db}
    buckets = {}
    for bt in bar_types:
        required = sorted(n for n, ident in identities.items() if ident == (symbol, bt))
        covered = sorted(n for n, _, _ in configs[bt])
        buckets[bt] = {"required": required, "covered": covered,
                       "uncataloged": sorted(set(required) - set(covered))}
    uncataloged = sorted(n for b in buckets.values() for n in b["uncataloged"])
    if uncataloged:
        raise CertificationRefused(f"studies in the DB are not in the results catalog: {uncataloged}")
    record = {
        "mode": "full-study",
        "results_catalog": str(results_path), "studies_db": str(studies_db),
        "studies": {bt: [{"study": n, "params": p, "params_source": f"optuna_db_trial_{k}", "window": list(w)}
                         for n, p, w, k in rows] for bt, rows in persisted.items()},
        "buckets": buckets, "db_studies": db, "uncataloged_db_studies": uncataloged,
    }
    return configs, record


def _check_alignment(report: dict | None, identity: dict) -> None:
    if report is None:
        raise CertificationRefused("no alignment report: run scripts/verify_paired_dataset.py first")
    if report.get("status") != "passed":
        raise CertificationRefused("alignment report has not passed")
    if report.get("full_history") is not True:
        raise CertificationRefused("alignment report is not a full-history verification")
    if report.get("dataset_id") != identity["dataset_id"]:
        raise CertificationRefused("alignment report is for another dataset")
    if report.get("root_hash") != identity["root_hash"]:
        raise CertificationRefused("alignment report root hash differs from the dataset's")


def _run_levels(v0_root, v0_version, v1_root, symbol: str, bar_types: list[str], study_configs: dict) -> dict:
    """L1-L4 per bar type; `study_configs` adds Optuna study windows (L3) and params (L4) to the defaults."""
    if not bar_types:   # every verdict is an all(): an empty list would pass with nothing compared
        raise CertificationRefused("no bar types to certify")
    roots, versions = (Path(v0_root), Path(v1_root)), (v0_version, None)
    types = {}
    for bt in bar_types:
        i0, i1 = (find_dataset(r, symbol, bt, v) for r, v in zip(roots, versions))
        studies = study_configs.get(bt, [])
        windows = [(None, None)] + [w for _, _, w in studies]
        types[bt] = {
            "v0": {"parquet": str(i0.parquet_path), "dataset_id": i0.dataset_id},
            "v1": {"parquet": str(i1.parquet_path), "dataset_id": i1.dataset_id},
            "dataset_ids_differ": i0.dataset_id != i1.dataset_id,
            "L1": level1(i0.parquet_path, i1.parquet_path),
            "L2": level2(i0.sidecar, i1.sidecar),
            "L3": level3(roots, versions, symbol, bt, windows),
            "L4": level4(roots, versions, symbol, bt, [("default", None, (None, None)), *studies]),
        }
    return types


def _summary(types: dict, full_label: str) -> dict:
    """Verdict + flags from the per-type levels; a failing level can never yield a passing label."""
    reproducible = all(t["L1"]["passed"] and t["L2"]["passed"] for t in types.values())
    pipeline = reproducible and all(t["L3"]["passed"] and t["L4"]["configs"][0]["equal"] for t in types.values())
    l1 = all(t["L1"]["passed"] for t in types.values())
    full = all(t[lvl]["passed"] for t in types.values() for lvl in ("L1", "L2", "L3", "L4"))
    return {"verdict": full_label if full else "bar-equivalent only" if l1 else "different",
            "dataset_reproducible": reproducible, "pipeline_equivalent_default": pipeline}


def _header(v0_root, v0_version, v1_identity: dict, alignment_report: dict) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "v1": v1_identity, "v0_root": str(v0_root), "v0_version": v0_version,
        "alignment_report": {k: alignment_report[k] for k in ("dataset_id", "root_hash", "n_trades")
                             if k in alignment_report},
    }


def certify_dataset(*, v0_root: Path, v0_version: str | None, v1_root: Path, v1_identity: dict,
                    alignment_report: dict | None, symbol: str, bar_types: list[str]) -> dict:
    """Dataset/pipeline certificate: alignment gate, L1, L2, L3 (full range), L4 (default config). No Optuna."""
    _check_alignment(alignment_report, v1_identity)
    types = _run_levels(v0_root, v0_version, v1_root, symbol, bar_types, {})
    return {
        "certification": "dataset-pipeline", **_summary(types, DATASET_PASS),
        "not_certified": {"optuna_research": "not certified: run --mode research",
                          "strategy_profitability": NOT_PROFITABILITY},
        **_header(v0_root, v0_version, v1_identity, alignment_report), "types": types, "limitation": LIMITATION,
    }


def certify_research(*, v0_root: Path, v0_version: str | None, v1_root: Path, v1_identity: dict,
                     alignment_report: dict | None, symbol: str, bar_types: list[str], results_path: Path,
                     studies_db: Path) -> dict:
    """Optuna research certificate: the dataset levels plus every catalogued study, with strict coverage."""
    _check_alignment(alignment_report, v1_identity)
    results_path, studies_db = Path(results_path), Path(studies_db)
    optuna_before = {str(p): _sha(p) for p in (results_path, studies_db)}
    study_configs, coverage = _coverage(results_path, studies_db, symbol, bar_types)
    types = _run_levels(v0_root, v0_version, v1_root, symbol, bar_types, study_configs)
    optuna_after = {str(p): _sha(p) for p in (results_path, studies_db)}
    if optuna_before != optuna_after:
        raise CertificationRefused("Optuna results or study DB changed during certification")
    return {
        "certification": "optuna-research", **_summary(types, "strategy-equivalent"),
        "not_certified": {"strategy_profitability": NOT_PROFITABILITY},
        "coverage": coverage, **_header(v0_root, v0_version, v1_identity, alignment_report),
        "optuna_files": optuna_before, "optuna_unchanged": True, "types": types, "limitation": LIMITATION,
    }


def _open_pinned(paired_root: Path, dataset_id: str, expected_root: str | None) -> tuple[PairedDataset, str]:
    """Open the dataset against a trusted root hash: the registry pin, or --expected-root (which must also
    agree with any registry entry). PairedDataset.open recomputes the root hash and checks every artifact."""
    if expected_root is None:
        return open_registered(paired_root, dataset_id, REGISTRY_PATH), "registry"
    if not SHA256_HEX.fullmatch(expected_root):
        raise CertificationRefused(f"--expected-root {expected_root!r} is not a lowercase 64-hex SHA-256")
    try:
        registered = pinned_root(dataset_id, REGISTRY_PATH)
    except DataIntegrityError:                     # not registered yet: the trusted value alone pins it
        registered = None
    if registered is not None and registered != expected_root:
        raise CertificationRefused(f"--expected-root differs from the registry pin for {dataset_id}")
    return PairedDataset.open(Path(paired_root) / dataset_id, expected_root), "--expected-root"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paired-root", type=Path, required=True)
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--alignment-report", type=Path, required=True)
    ap.add_argument("--expected-root", help="trusted root hash (64 hex) for a dataset not registered yet; "
                                            "must match the registry pin if there is one")
    ap.add_argument("--mode", choices=("dataset", "research"), default="dataset",
                    help="dataset (default): bars + default-config pipeline, never reads Optuna; "
                         "research: also replays every catalogued Optuna study, strict coverage")
    ap.add_argument("--v0-root", type=Path, default=Path(INFO_BARS.root))
    ap.add_argument("--v0-version", default=INFO_BARS.version)
    ap.add_argument("--results", type=Path, default=ROOT / "output" / "optimize" / "results.json",
                    help="research mode only")
    ap.add_argument("--studies-db", type=Path, default=ROOT / "output" / "optimize" / "studies.db",
                    help="research mode only")
    args = ap.parse_args(argv)
    ds, pin_source = _open_pinned(args.paired_root, args.dataset_id, args.expected_root)
    report = json.loads(args.alignment_report.read_text()) if args.alignment_report.is_file() else None
    common = dict(v0_root=args.v0_root, v0_version=args.v0_version, v1_root=ds.bars_root,
                  v1_identity={"dataset_id": ds.dataset_id, "root_hash": ds.root_hash}, alignment_report=report,
                  symbol=ds.manifest["symbol"], bar_types=ds.bar_types)
    if args.mode == "research":
        cert = certify_research(**common, results_path=args.results, studies_db=args.studies_db)
    else:
        cert = certify_dataset(**common)
    cert["pin_source"] = pin_source
    ds.recheck()
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    out = CERT_DIR / f"{args.mode}-{ds.dataset_id}.json"
    out.write_text(json.dumps(cert, indent=2, default=str))
    print(f"{cert['verdict']}: {out}")


if __name__ == "__main__":
    main()

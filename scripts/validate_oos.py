"""Out-of-sample validation of frozen Optuna trials on 2024-2025 (dataset policy).

Single pass: the candidate set is frozen before the first read and is never iterated. A candidate is
confirmed only if its net expectancy lower bound is above zero (Bonferroni over the candidates, standard
error inflated for positive serial correlation). Deflated Sharpe is attached as a diagnostic only.
Assumes roughly independent trades, which is unverified.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts import dataset_policy as policy
from scripts import diagnose_trades as dx
from scripts import frozen_eval as fe
from scripts import optimize_orderflow as opt

VALIDATION_WARMUP_DAYS = 90
# Strategy fingerprints of older studies whose behavior was reviewed as compatible with the current code.
# Empty by default: replay equality on the development inputs cannot prove equivalence on validation inputs.
COMPATIBLE_LEGACY_STRATEGY_FPS = frozenset()


def subset_diffs(stored, current, path=""):
    """Paths where a stored default differs from today's; keys that exist only today are allowed (legacy)."""
    if isinstance(stored, dict) and isinstance(current, dict):
        return [d for key, value in stored.items()
                for d in (subset_diffs(value, current[key], f"{path}{key}.") if key in current
                          else [f"{path}{key} (removed)"])]
    return [] if stored == current else [path.rstrip(".")]


def frozen_strategy_problems(identity, stored_versions, symbol):
    """Reasons the stored study no longer describes the code that would run (empty when compatible)."""
    problems = []
    stored = identity.get("instrument_defaults")
    if stored is None:
        problems.append("stored identity lacks instrument_defaults")
    else:
        current = opt.json_safe(asdict(opt.instrument_config(symbol)))
        problems += [f"instrument default changed since the study: {d}" for d in subset_diffs(stored, current)]
    if stored_versions is None:
        problems.append("study lacks stored library versions")
    else:
        now = opt.library_versions()
        problems += [f"library {name} {version} is not compatible with installed {now[name]}"
                     for name, version in stored_versions.items()
                     if name in now and version.split(".")[:2] != now[name].split(".")[:2]]
    return problems


def preflight_candidate(study_db, study_name, trial, *, cache_dir, start=None, end=None):
    """Resolve and verify one candidate. Reads the development cache only; never validation data."""
    start, end = start or policy.VALIDATION_START, end or policy.VALIDATION_END
    policy.ensure_validation_window(start, end)
    params, number = dx.load_trial(study_db, study_name, trial)
    identity = dx.load_identity(study_db, study_name)
    settings = dx.bind_replay_settings(identity)
    path, meta = dx.find_cached_bars(cache_dir, identity["data_fp"])    # manifest only, no market data
    # Everything below runs before any validation-data access: the frozen strategy must still be this code.
    problems = frozen_strategy_problems(identity, dx.load_study_attr(study_db, study_name, "versions"), meta["symbol"])
    if problems:
        raise ValueError(f"{study_name} trial {number} is not reproducible here: " + "; ".join(problems))
    frozen_fp = identity.get("strategy_fp")
    if frozen_fp != opt.STRATEGY_FP and frozen_fp not in COMPATIBLE_LEGACY_STRATEGY_FPS:
        raise ValueError(f"{study_name} trial {number}: strategy fingerprint {frozen_fp} is not the current "
                         f"{opt.STRATEGY_FP} and has no reviewed compatibility mapping; the strategy sources changed "
                         "after tuning (re-tune, or add a reviewed mapping)")
    policy.ensure_dev_window(date.fromisoformat(meta["start"]), date.fromisoformat(meta["end"]))
    params = {**opt.baseline_extra_params(meta["symbol"]), **params}
    dx.replay_and_verify(path, meta, params, settings, dx.stored_digest(study_db, study_name, number))
    return {"study": study_name, "trial": number, "params": params, "settings": settings, "meta": meta,
            "start": start, "end": end, "strategy_fp_matches": identity.get("strategy_fp") == opt.STRATEGY_FP}


def evaluate_candidate(prep, *, study_db, data_root, trades_root=None):
    """The only step that reads validation data. Call it after the campaign manifest is frozen."""
    meta = prep["meta"]
    result = fe.evaluate_frozen(prep["params"], symbol=meta["symbol"], bar_type=meta["bar_type"],
                                start=prep["start"], end=prep["end"], settings=prep["settings"],
                                data_root=data_root, cache_dir=prep.get("cache_dir"), trades_root=trades_root,
                                warmup_days=VALIDATION_WARMUP_DAYS, guard=policy.ensure_validation_window,
                                expected_dataset={k: meta.get(k) for k in ("source", "source_metadata", "data_proc_fp")})
    import optuna
    study = optuna.load_study(study_name=prep["study"], storage=f"sqlite:///{Path(study_db)}")
    gate = opt.deflated_gate(study, np.asarray(result["daily_returns"], dtype=float),
                             campaign=opt.campaign_trials(study_db, meta["symbol"]))
    return {**result, "study": prep["study"], "trial": prep["trial"], "params": prep["params"],
            "deflated_sharpe_gate": gate, "strategy_fp_matches": prep["strategy_fp_matches"],
            "development_digest_verified": True}


def validate(study_db, study_name, trial, *, cache_dir, data_root, trades_root=None, start=None, end=None):
    prep = preflight_candidate(study_db, study_name, trial, cache_dir=cache_dir, start=start, end=end)
    return evaluate_candidate({**prep, "cache_dir": cache_dir}, study_db=study_db, data_root=data_root,
                              trades_root=trades_root)


def confirmation(results, alpha=0.05):
    """Per-candidate D10 rule: solvent, expectancy > 0 and mean - z * se_inflated > 0 (z Bonferroni over k).

    A bankrupt equity path or missing solvency evidence is never confirmed, whatever its mean R.
    """
    z = NormalDist().inv_cdf(1 - alpha / len(results))
    rows = []
    for result in results:
        d = result["decomposition"]
        mean, se, r = d["net_expectancy_r"], d["se_r"], np.asarray(result.get("trade_r") or [], dtype=float)
        rho = None
        if len(r) >= 3 and r.std() > 0:
            rho = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        inflated = se
        if se is not None and rho is not None and rho > 0:
            inflated = se * math.sqrt((1 + min(rho, 0.99)) / (1 - min(rho, 0.99)))
        bound = None if mean is None or inflated is None else mean - z * inflated
        blown = (result.get("metrics") or {}).get("account_blown")      # None = no solvency evidence: fail closed
        rows.append({"account_blown": blown, "net_expectancy_r": mean, "se_r": se, "rho1": rho, "se_r_inflated": inflated,
                     "n_trades": d["n_trades"], "z": z, "lower_bound": bound,
                     "confirmed": bool(blown is False and mean is not None and mean > 0
                                   and bound is not None and bound > 0)})
    return rows


ALPHA = 0.05


def parse_candidates(text):
    out = []
    for item in text.split(","):
        name, _, trial = item.partition(":")
        out.append((name, int(trial) if trial else None))
    return out


def candidate_key(prep):
    return (prep["study"], prep["trial"], opt.fingerprint(prep["params"]))


def canonical_candidates(prepared):
    """Deterministic order independent of the command line; duplicates would double-count the Bonferroni k."""
    ordered = sorted(prepared, key=candidate_key)
    keys = [candidate_key(p) for p in ordered]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate candidate in the campaign (same study, trial and parameters)")
    return ordered


def campaign_id(prepared, start, end):
    frozen = [candidate_key(p) for p in canonical_candidates(prepared)]
    return opt.fingerprint({"candidates": frozen, "alpha": ALPHA, "start": str(start), "end": str(end)})[:16]


def write_manifest(path, manifest):
    with opt.file_lock(str(path) + ".lock"):
        opt.atomic_json(path, manifest)


def record_result(path, index, result):
    with opt.file_lock(str(path) + ".lock"):
        manifest = json.loads(path.read_text())
        entry = manifest["candidates"][index]
        if (result.get("study"), result.get("trial")) != (entry["study"], entry["trial"]):
            raise ValueError(f"result for {result.get('study')} trial {result.get('trial')} does not belong in the "
                             f"slot of {entry['study']} trial {entry['trial']}")
        entry.update(status="done", result=result)
        opt.atomic_json(path, manifest)


def admit_campaign(output_dir, prepared, start, end, acknowledge):
    """Create the frozen manifest, or find the identical one. Exposure discovery and creation are one critical
    section, so concurrent invocations cannot both be admitted as first."""
    prepared = canonical_candidates(prepared)
    cid = campaign_id(prepared, start, end)
    output_dir = Path(output_dir)
    path = output_dir / f"campaign-{cid}.json"
    with opt.file_lock(output_dir / ".admission.lock"):
        if path.exists():
            return path
        prior = sorted(p.name for p in output_dir.glob("campaign-*.json"))
        if prior and not acknowledge:
            raise ValueError(f"validation data already exposed by {', '.join(prior)}; a different candidate set "
                             "would iterate the validation window (--acknowledge-prior-exposure records and proceeds)")
        write_manifest(path, {
            "id": cid, "created_utc": datetime.now(timezone.utc).isoformat(), "alpha": ALPHA,
            "window": {"start": str(start), "end": str(end)}, "prior_campaigns": prior,
            "candidates": [{"study": p["study"], "trial": p["trial"], "params_fp": opt.fingerprint(p["params"]),
                            "status": "pending"} for p in prepared]})
    return path


def run_campaign(path, prepared, evaluate):
    """Evaluate the pending candidates under an exclusive run lock; results are matched by identity, not position."""
    with opt.file_lock(str(path) + ".run", nonblocking=True, message="campaign already running"):
        manifest = json.loads(path.read_text())       # re-read under the lock: another run may have finished some
        slots = {(c["study"], c["trial"], c["params_fp"]): i for i, c in enumerate(manifest["candidates"])}
        for prep in canonical_candidates(prepared):
            key = candidate_key(prep)
            if key not in slots:
                raise ValueError(f"candidate {key[0]} trial {key[1]} is not in campaign {manifest['id']}")
            if manifest["candidates"][slots[key]]["status"] == "pending":
                record_result(path, slots[key], opt.json_safe(evaluate(prep)))
        return json.loads(path.read_text())


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-db", type=Path, default=Path("output/optimize/studies.db"))
    parser.add_argument("--candidates", required=True, help="study[:trial],... frozen before the first read")
    parser.add_argument("--cache-dir", type=Path, default=Path("output/optimize/cache"))
    parser.add_argument("--data-root", type=Path, default=opt.INFO_BARS.root)
    parser.add_argument("--trades-root", type=Path, default=opt.build_parser().get_default("trades_root"),
                        help="tick archive for time_1m studies (default: the optimizer's, honouring ORDERFLOW_TRADES_ROOT)")
    parser.add_argument("--output-dir", type=Path, default=Path("output/validate"))
    parser.add_argument("--acknowledge-prior-exposure", action="store_true",
                        help="record and proceed although another campaign already read the validation window")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    start, end = policy.VALIDATION_START, policy.VALIDATION_END
    # 1. Resolve and verify every candidate before any validation data is read; a failure here exposes nothing.
    prepared = [preflight_candidate(args.study_db, name, trial, cache_dir=args.cache_dir, start=start, end=end)
                for name, trial in parse_candidates(args.candidates)]
    prepared = canonical_candidates([{**p, "cache_dir": args.cache_dir} for p in prepared])
    # 2. Freeze the campaign (or resume the identical one); a different set after exposure is refused.
    path = admit_campaign(args.output_dir, prepared, start, end, args.acknowledge_prior_exposure)
    # 3. Evaluate the pending candidates; every result is persisted as soon as it exists.
    manifest = run_campaign(path, prepared, lambda prep: evaluate_candidate(
        prep, study_db=args.study_db, data_root=args.data_root, trades_root=args.trades_root))
    results = [c["result"] for c in manifest["candidates"]]
    rows = confirmation(results, ALPHA)
    confirmed = [(row, result) for row, result in zip(rows, results) if row["confirmed"]]
    selected = None
    if confirmed:
        row, result = max(confirmed, key=lambda pair: pair[0]["net_expectancy_r"])
        selected = {"study": result["study"], "trial": result["trial"], "net_expectancy_r": row["net_expectancy_r"]}
        print(f"selected {selected['study']} trial {selected['trial']} "
              f"(net expectancy {selected['net_expectancy_r']:+.3f}R)")
    else:
        print("no candidate confirmed; stop (the validation window is not to be iterated)")
    summary = {"campaign": manifest["id"], "candidates": [{"study": r["study"], "trial": r["trial"]} for r in results],
               "confirmation": rows, "selected": selected}
    write_manifest(path, {**json.loads(path.read_text()), "summary": opt.json_safe(summary)})
    return summary


if __name__ == "__main__":
    print(json.dumps(opt.json_safe(main()), indent=2))

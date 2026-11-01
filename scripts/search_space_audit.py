"""Read-only boundary audit of a finished Optuna study: do good / recent trials pile up on a search-space edge?

Rules (documented in scripts/search_spaces/FINDINGS.md):
  * position x in [0, 1] is linear in the value, or in log(value) for ``float_log``;
  * top  = the TOP_K best *feasible* trials by value, counted when x < TOP_EDGE_LOW / x > TOP_EDGE_HIGH;
  * tail = the last TAIL COMPLETE trials by number (feasible or not), fraction with x < TAIL_EDGE_LOW / x > TAIL_EDGE_HIGH;
  * widen_* is true when top_* >= TOP_COUNT_MIN or tail_* >= TAIL_FRACTION_MIN.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import optimize_orderflow as opt  # noqa: E402

TOP_K = 10
TAIL = 50
TOP_EDGE_LOW, TOP_EDGE_HIGH = 0.10, 0.90
TAIL_EDGE_LOW, TAIL_EDGE_HIGH = 0.05, 0.95
TOP_COUNT_MIN = 5
TAIL_FRACTION_MIN = 0.30


def _position(value, lo, hi, kind):
    if kind == "float_log":
        return (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (value - lo) / (hi - lo)


def boundary_hits(trials, space, *, top_k=TOP_K, tail=TAIL):
    feasible = [t for t in trials if t.get("feasible") and t.get("value") is not None and math.isfinite(t["value"])]
    top = sorted(feasible, key=lambda t: t["value"], reverse=True)[:top_k]
    last = sorted((t for t in trials if t["state"] == "COMPLETE"), key=lambda t: t["number"])[-tail:]
    out = {}
    for name, (lo, hi, kind) in space.items():
        if hi <= lo:      # fixed dimension (load_space allows low == high): no edge to press against
            out[name] = {"top_low": 0, "top_high": 0, "tail_low": 0.0, "tail_high": 0.0,
                         "widen_low": False, "widen_high": False}
            continue
        top_x = [_position(t["params"][name], lo, hi, kind) for t in top if name in t["params"]]
        tail_x = [_position(t["params"][name], lo, hi, kind) for t in last if name in t["params"]]
        row = {"top_low": sum(x < TOP_EDGE_LOW for x in top_x), "top_high": sum(x > TOP_EDGE_HIGH for x in top_x),
               "tail_low": sum(x < TAIL_EDGE_LOW for x in tail_x) / len(tail_x) if tail_x else 0.0,
               "tail_high": sum(x > TAIL_EDGE_HIGH for x in tail_x) / len(tail_x) if tail_x else 0.0}
        row["widen_low"] = row["top_low"] >= TOP_COUNT_MIN or row["tail_low"] >= TAIL_FRACTION_MIN
        row["widen_high"] = row["top_high"] >= TOP_COUNT_MIN or row["tail_high"] >= TAIL_FRACTION_MIN
        out[name] = row
    return out


def format_table(hits, space):
    lines = [f"{'dimension':<38}{'bounds':<22}{'top_lo':>7}{'top_hi':>7}{'tail_lo':>8}{'tail_hi':>8}  widen",
             "-" * 98]
    for name, h in hits.items():
        lo, hi, kind = space[name]
        flags = ",".join(s for s, on in (("LOW", h["widen_low"]), ("HIGH", h["widen_high"])) if on) or "-"
        lines.append(f"{name:<38}{f'{lo:g}..{hi:g} {kind}':<22}{h['top_low']:>7}{h['top_high']:>7}"
                     f"{h['tail_low']:>8.2f}{h['tail_high']:>8.2f}  {flags}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--study-db", required=True)
    p.add_argument("--study-name", required=True)
    args = p.parse_args(argv)
    trials = opt.read_study_trials(args.study_db, args.study_name)
    space = {k: tuple(v) for k, v in opt.read_study_attrs(args.study_db, args.study_name)["identity"]["search_space"].items()}
    n_complete = sum(t["state"] == "COMPLETE" for t in trials)
    n_feasible = sum(t["feasible"] for t in trials)
    print(f"{args.study_name}: {len(trials)} trials, {n_complete} complete, {n_feasible} feasible")
    print(format_table(boundary_hits(trials, space), space))
    return 0


if __name__ == "__main__":
    sys.exit(main())

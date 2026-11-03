"""Peak RSS of ONE child process (plan §11): native DuckDB/Arrow memory included.

The child is reaped with os.wait4(pid), whose rusage belongs to that PID only, so a run's peak never
inherits an earlier, larger child's (RUSAGE_CHILDREN is a running maximum over all reaped children).

    python scripts/measure_rss.py -- python -I scripts/verify_paired_dataset.py --paired-root ... --dataset-id ...

Growth over a continuous run: the workload itself samples its current RSS (e.g. the alignment report's
rss.per_day_bytes); `rss_growth_ok` applies the plan's rule last <= 1.10 * first + 50 MiB.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

MIB = 1 << 20


def maxrss_bytes(ru_maxrss: int, platform: str = sys.platform) -> int:
    """ru_maxrss is bytes on macOS and KiB on Linux."""
    return ru_maxrss if platform == "darwin" else ru_maxrss * 1024


def run_measured(argv: list[str]) -> dict:
    t0 = time.monotonic()
    proc = subprocess.Popen(argv)
    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)   # reaped here; keep Popen consistent
    return {"argv": argv, "returncode": proc.returncode, "peak_rss_bytes": maxrss_bytes(usage.ru_maxrss),
            "wall_seconds": round(time.monotonic() - t0, 3)}


def rss_growth_ok(samples: list[int], factor: float = 1.10, slack: int = 50 * MIB) -> bool:
    return len(samples) < 2 or samples[-1] <= factor * samples[0] + slack


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv or argv.index("--") == len(argv) - 1:
        sys.exit("usage: measure_rss.py -- <command> [args...]")
    result = run_measured(argv[argv.index("--") + 1:])
    print(json.dumps(result, indent=2))
    sys.exit(result["returncode"])


if __name__ == "__main__":
    main()

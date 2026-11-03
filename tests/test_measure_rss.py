"""R7 (plan §9, rev 4): per-process peak RSS that sees native memory and never inherits an earlier child's peak."""
import sys

from scripts.measure_rss import MIB, maxrss_bytes, rss_growth_ok, run_measured

PY = [sys.executable, "-I", "-c"]
TOUCH = ("import numpy as np, pyarrow as pa; b = pa.allocate_buffer({n} * 2**20, resizable=False); "
         "np.frombuffer(b, dtype=np.uint8)[:] = 1")          # write every page -> physically resident
NOOP = "import numpy as np, pyarrow as pa"


def test_native_arrow_allocation_is_seen():               # R7(a)
    base = run_measured([*PY, NOOP])["peak_rss_bytes"]
    big = run_measured([*PY, TOUCH.format(n=200)])
    assert big["returncode"] == 0
    assert big["peak_rss_bytes"] - base >= 190 * MIB


def test_later_smaller_child_reports_its_own_peak():      # R7(b), Codex counterexample
    first = run_measured([*PY, TOUCH.format(n=64)])["peak_rss_bytes"]
    second = run_measured([*PY, TOUCH.format(n=4)])["peak_rss_bytes"]
    assert second < first - 40 * MIB


def test_unit_normalisation():                           # R7(c)
    assert maxrss_bytes(1000, "darwin") == 1000
    assert maxrss_bytes(1000, "linux") == 1000 * 1024


LEAKER = """
import gc, json, psutil
keep, samples = [], []
for day in range(7):
    block = bytearray({leak} * 2**20)                   # bytearray is zero-filled -> resident
    if {retain}:
        keep.append(block)
    del block
    gc.collect()
    samples.append(psutil.Process().memory_info().rss)
print(json.dumps(samples))
"""


def _samples(leak, retain):
    import json
    import subprocess
    out = subprocess.run([*PY, LEAKER.format(leak=leak, retain=retain)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_growth_rule_flags_a_leak_and_passes_a_steady_run():   # R7(d)
    assert not rss_growth_ok(_samples(leak=10, retain=True), slack=20 * MIB)   # +60 MiB over 7 "days"
    assert rss_growth_ok(_samples(leak=10, retain=False), slack=20 * MIB)
    assert rss_growth_ok([100 * MIB] * 7) and not rss_growth_ok([100 * MIB, 200 * MIB])

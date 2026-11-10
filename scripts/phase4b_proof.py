"""Amendment A1: the Phase 4b real-data proof reads only the search segment.

H is the earlier of the two arms' last search-bar close times (read with CachedBars(stop=n_bars_search)):
fixed_volume_imbalance 2023-04-29T03:32:55.751Z (fixed_tick_imbalance: 2023-05-09T19:14:04.024Z). It applies to
every bar type. Reader end dates are exclusive, so a window may end at 2023-04-29 at the latest.
"""
from __future__ import annotations

HORIZON_UTC = '2023-04-29T03:32:55.751Z'
LAST_SAFE_EXCLUSIVE_END = '2023-04-29'
REPLACEMENT_FILE = 'tests/test_phase4b_real_search_only.py'
DESELECTED = (
    'tests/test_real_dataset.py::test_pipeline_on_real_bars',
    'tests/test_real_dataset.py::test_tick_run_real_distant_absorption_is_not_entered',
    'tests/test_optimize_orderflow.py::test_real_prefix_and_notebook_parity',
    'tests/test_optimize_orderflow.py::test_real_folds_match_reference',
    'tests/test_phase4_real.py::test_bounded_real_cache_and_live[2023-10-16-2023-10-17]',
    'tests/test_phase4_real.py::test_bounded_real_cache_and_live[2023-12-30-2024-01-01]',
)
# 9 retained (#2, #6 x6, #7, #8) + A1-1 x6, A1-2, A1-3, A1-4.
EXPECTED_COLLECTED = 18


def real_data_selection():
    argv = ['-p', 'tests.phase4b_search_guard', '-m', 'real_data']
    for node in DESELECTED:
        argv += ['--deselect', node]
    return argv


A1_PATH_RELATIVE = 'docs/plans/phase4b-proof-amendment-A1-search-only-real-data.md'


def _root():
    from pathlib import Path
    return Path(__file__).resolve().parents[1]


A1_PATH = _root() / A1_PATH_RELATIVE


def proof_argv(role):
    import sys
    base = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider']
    if role == 'unit':
        return base + ['-m', 'not real_data']
    if role == 'real_data':
        return base + real_data_selection()
    raise ValueError('proof role must be unit or real_data')


def _counts(text):
    import re
    tail = text.strip().splitlines()[-1] if text.strip() else ''
    words = {'passed': 'passed', 'failed': 'failed', 'skipped': 'skipped', 'errors': r'errors?'}
    return {key: int(m.group(1)) if (m := re.search(rf'(\d+) {word}\b', tail)) else 0
            for key, word in words.items()} | {'summary': tail}


def write_artifact(role, path, *, runner=None):
    """Run the A1 proof command for ``role`` and write its JSON artifact (the gate checks its content)."""
    import hashlib
    import json
    import os
    import subprocess
    from pathlib import Path
    runner = runner or subprocess.run
    argv = proof_argv(role)
    # Host-input tests fail instead of skipping; the post-look holdout opt-in never reaches a proof run.
    env = {k: v for k, v in os.environ.items() if k != 'PHASE4B_ALLOW_HOLDOUT_TESTS'} | {'PHASE4B_REQUIRE_HOST': '1'}
    collected = runner(argv + ['--collect-only'], cwd=_root(), capture_output=True, text=True, env=env)
    nodes = [line for line in collected.stdout.splitlines() if '::' in line]
    result = runner(argv, cwd=_root(), capture_output=True, text=True, env=env)
    value = {'role': role, 'argv': argv, 'returncode': result.returncode, 'require_host': True,
             **_counts(result.stdout)}
    if role == 'real_data':
        value.update(collected=nodes, horizon_utc=HORIZON_UTC,
                     amendment_sha256=hashlib.sha256(Path(A1_PATH).read_bytes()).hexdigest())
    else:
        value.update(collected_count=len(nodes))
    path = Path(path)
    path.write_bytes(json.dumps(value, sort_keys=True, separators=(',', ':')).encode() + b'\n')
    return path

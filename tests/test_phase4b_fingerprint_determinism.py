"""runtime_fingerprint() must not depend on a process's import history (S1 attempt 9056f902 integrity_stop).

The S1 worker lazily imported scripts.phase4b_proof while verifying gates; the post-flight child did not, so the
hashed module set differed although no code or input changed. Each check runs in fresh interpreters.
"""
import json
import subprocess
import sys

from scripts.phase4_policy import ROOT

FINGERPRINT = 'import json;from scripts.phase4b_runtime import runtime_fingerprint as f;print(json.dumps(f()))'

# Imports every scripts/orderflow_system module named by any import (top-level or lazy) in a fingerprinted module,
# to a fixpoint, so any lazy import that a worker, post-flight, gate or recovery path may trigger is loaded.
FIXPOINT = r'''
import ast, importlib, json, sys
from pathlib import Path
from scripts.phase4b_runtime import runtime_fingerprint as f
before = f()
def named(path):
    out = set()
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
            out |= {node.module + '.' + a.name for a in node.names}
    return {n for n in out if n.split('.')[0] in ('scripts', 'orderflow_system')}
seen = set()
while True:
    files = {getattr(m, '__file__', None) for n, m in list(sys.modules.items()) if n.split('.')[0] in ('scripts', 'orderflow_system')}
    wanted = set().union(*(named(p) for p in files if p and p.endswith('.py'))) - seen
    if not wanted:
        break
    for name in sorted(wanted):
        seen.add(name)
        try:
            importlib.import_module(name)
        except ImportError:
            pass  # "from pkg import attr" names that are attributes, not modules
after = f()
print(json.dumps({'equal': before == after,
                  'extra': sorted(set(after['module_sha256']) - set(before['module_sha256']))}))
'''


def run(code, *prelude):
    script = ';'.join([*prelude, code])
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True,
                            env={'PYTHONPATH': str(ROOT), 'PATH': '/usr/bin:/bin'})
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_worker_gate_verification_import_does_not_change_the_fingerprint():
    postflight = run(FINGERPRINT, 'import scripts.phase4b_campaign')
    worker = run(FINGERPRINT, 'import scripts.phase4b_campaign', 'import scripts.phase4b_proof')
    assert worker == postflight


def test_fingerprint_is_closed_under_every_reachable_project_import():
    result = run(FIXPOINT)
    assert result == {'equal': True, 'extra': []}

"""Amendment A1 guards: the Phase 4b real-data proof reads only the search segment (no real data read here)."""
import ast
import re
import subprocess
import sys
from pathlib import Path

from scripts import phase4b_proof as proof

ROOT = Path(__file__).resolve().parents[1]
DATE = re.compile(r'20\d\d-\d\d-\d\d')


def dates_in(source):
    return {date for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and isinstance(node.value, str) for date in DATE.findall(node.value)}


def test_a1_horizon_is_the_earlier_arm_boundary():
    assert proof.HORIZON_UTC == '2023-04-29T03:32:55.751Z'
    assert proof.LAST_SAFE_EXCLUSIVE_END == '2023-04-29'


def test_g_static_replacement_windows_end_before_the_horizon():
    source = (ROOT / proof.REPLACEMENT_FILE).read_text()
    found = dates_in(source)
    assert found, 'replacement file must state its windows literally'
    assert all(d <= proof.LAST_SAFE_EXCLUSIVE_END for d in found), sorted(found)
    assert {'2023-03-19', '2023-03-24', '2023-03-01', '2023-03-04', '2023-04-17', '2023-04-18'} <= found


def collected(*extra):
    result = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q', '-p', 'no:cacheprovider',
                             *proof.real_data_selection(), *extra], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode in (0, 5), result.stdout + result.stderr
    return [line for line in result.stdout.splitlines() if '::' in line]


def test_g_collect_every_deselected_node_exists():
    full = collected()
    everything = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q', '-p', 'no:cacheprovider',
                                 '-m', 'real_data'], cwd=ROOT, capture_output=True, text=True).stdout
    for node in proof.DESELECTED:
        assert node in everything, node
        assert not any(line.startswith(node) for line in full), node


def test_g_collect_no_collected_real_data_test_states_a_window_at_or_after_the_horizon():
    nodes = collected()
    assert len(nodes) == proof.EXPECTED_COLLECTED, nodes
    for node in nodes:
        path, _, name = node.partition('::')
        function, _, params = name.partition('[')
        tree = ast.parse((ROOT / path).read_text())
        body = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
        # The function body plus this node's own parameters; decorators list other (deselected) nodes' windows.
        stated = dates_in('\n'.join(ast.unparse(stmt) for stmt in body.body)) | set(DATE.findall(params))
        assert all(d <= proof.LAST_SAFE_EXCLUSIVE_END for d in stated), (node, sorted(stated))


def test_a1_command_is_the_selection_plus_replacements():
    argv = proof.real_data_selection()
    assert argv[2:4] == ['-m', 'real_data']
    assert [argv[i + 1] for i, a in enumerate(argv) if a == '--deselect'] == list(proof.DESELECTED)


# ---- Runtime guard plugin (loaded by the proof command): refuses unbounded or late readers before any read.
def test_plugin_refuses_unbounded_and_late_readers(monkeypatch):
    from tests import phase4b_search_guard as guard
    import orderflow_system.data.infobar_feed as feed
    from scripts import trade_feature_cache as tfc
    original = feed.InfoBarReader
    guard.install(monkeypatch)
    assert feed.InfoBarReader is not original and issubclass(feed.InfoBarReader, original)
    info = object()
    for end_ms in (None, guard.LIMIT_MS + 1):
        try:
            feed.InfoBarReader(info, 0, end_ms)
        except AssertionError as exc:
            assert 'search horizon' in str(exc)
        else:
            raise AssertionError('reader past the horizon was constructed')
    try:  # Admitted by the guard; the dummy info then fails inside the real constructor, not in the guard.
        feed.InfoBarReader(info, 0, guard.LIMIT_MS)
    except AssertionError:
        raise
    except Exception as exc:
        assert 'search horizon' not in str(exc)
    for end in (None, '2023-04-30', '2023-12-31'):
        try:
            tfc.build(object(), 'x', start='2023-04-01', end=end)
        except AssertionError as exc:
            assert 'search horizon' in str(exc)
        else:
            raise AssertionError('bounded build past the horizon was admitted')


def test_plugin_limit_is_midnight_before_the_horizon():
    from tests import phase4b_search_guard as guard
    from datetime import datetime, timezone
    assert datetime.fromtimestamp(guard.LIMIT_MS / 1000, timezone.utc).isoformat() == '2023-04-29T00:00:00+00:00'


def test_selection_loads_the_guard_plugin():
    assert proof.real_data_selection()[:2] == ['-p', 'tests.phase4b_search_guard']


def test_dates_in_finds_dates_inside_timestamps():
    assert dates_in("x = '2023-11-19T00:00'") == {'2023-11-19'}


def test_plugin_activates_in_a_p_run(tmp_path):
    probe = tmp_path / 'test_probe.py'
    probe.write_text('def test_guarded():\n'
                     '    import orderflow_system.data.infobar_feed as feed\n'
                     '    assert feed.InfoBarReader.__name__ == "GuardedReader"\n')
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-p',
                             'tests.phase4b_search_guard', str(probe)], cwd=ROOT, capture_output=True, text=True)
    assert '1 passed' in result.stdout, result.stdout + result.stderr


def test_deselected_holdout_tests_are_skipped_unless_explicitly_allowed():
    # Safe red/green: the guard plugin is loaded, so if the skip were missing the read would be refused first.
    node = 'tests/test_optimize_orderflow.py::test_real_prefix_and_notebook_parity'
    env = {k: v for k, v in __import__('os').environ.items() if k != 'PHASE4B_ALLOW_HOLDOUT_TESTS'}
    # --setup-only: even if the skip regressed, the test body (and any read) never executes.
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-rs', '--setup-only', '-p', 'no:cacheprovider',
                             '-p', 'tests.phase4b_search_guard', node], cwd=ROOT, capture_output=True, text=True,
                            env=env)
    assert '1 skipped' in result.stdout and 'amendment A1' in result.stdout, result.stdout + result.stderr


def test_proof_writer_records_command_collection_and_counts(tmp_path, monkeypatch):
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        out = ('\n'.join(f'tests/test_x.py::t{i}' for i in range(proof.EXPECTED_COLLECTED)) + '\n'
               if '--collect-only' in argv else f'{proof.EXPECTED_COLLECTED} passed in 1.0s\n')
        return __import__('types').SimpleNamespace(returncode=0, stdout=out, stderr='')
    a1 = tmp_path / 'a1.md';a1.write_bytes(b'A1')
    monkeypatch.setattr(proof, 'A1_PATH', a1)
    path = proof.write_artifact('real_data', tmp_path / 'real_data.json', runner=runner)
    value = __import__('json').loads(path.read_bytes())
    assert value['argv'][-len(proof.real_data_selection()):] == proof.real_data_selection()
    assert len(value['collected']) == proof.EXPECTED_COLLECTED and value['passed'] == proof.EXPECTED_COLLECTED
    assert value['failed'] == value['errors'] == value['skipped'] == 0
    assert value['horizon_utc'] == proof.HORIZON_UTC
    assert value['amendment_sha256'] == __import__('hashlib').sha256(b'A1').hexdigest()
    unit = __import__('json').loads(proof.write_artifact('unit', tmp_path / 'unit.json', runner=runner).read_bytes())
    assert unit['argv'][-2:] == ['-m', 'not real_data']


def test_proof_counts_parse_singular_and_plural():
    assert proof._counts('x\n1 failed, 17 passed, 1 error in 2s')['errors'] == 1
    assert proof._counts('x\n2 errors in 1s')['errors'] == 2
    assert proof._counts('18 passed in 9s')['passed'] == 18


def test_replacements_skip_without_the_guard_plugin():
    # Safe: the replacement module skips itself before any read when the plugin is absent.
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-rs', '--setup-only', '-p', 'no:cacheprovider',
                             proof.REPLACEMENT_FILE + '::test_a1_3_real_prefix_and_notebook_parity'],
                            cwd=ROOT, capture_output=True, text=True)
    assert '1 skipped' in result.stdout and 'guard plugin' in result.stdout, result.stdout + result.stderr

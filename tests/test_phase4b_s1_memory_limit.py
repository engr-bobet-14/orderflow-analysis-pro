"""A3: S1-only 12 GiB RSS limit (owner decision 2026-11-08); private 4 GiB and every other stage unchanged.

S1 run 42d5c2da stopped at 4.15 GiB RSS with 0.46 GiB private. The read-only probe showed the excess is the
two production trade caches' file-backed residency while both arms are open (bounded, released on close).
"""
import ast
from contextlib import contextmanager
from pathlib import Path

import pytest

from scripts import phase4_resources as res
from scripts import phase4b_campaign as driver
from scripts import optimize_orderflow as opt
from scripts.phase4_policy import PhaseLayout

GiB = 1024**3


def sample(rss, private, fallback=False):
    return {'rss_bytes': rss, 'peak_rss_bytes': rss, 'private_bytes': private, 'peak_private_bytes': private,
            'private_is_fallback': fallback, 'private_metric': 'test'}


def test_default_bounds_are_unchanged_for_phase4_and_other_stages():
    bounds = res.memory_bounds(sample(int(4.15 * GiB), int(.46 * GiB)))
    assert not bounds['pass'] and bounds['rss_limit_bytes'] == res.PRIVATE_BUDGET == 4 * GiB


def test_s1_rss_limit_admits_the_observed_residency_but_not_more_or_private_overruns():
    ok = res.memory_bounds(sample(int(4.38 * GiB), int(.46 * GiB)), 0, 12 * GiB)
    assert ok['pass'] and ok['rss_limit_bytes'] == 12 * GiB and ok['private_limit_bytes'] == 4 * GiB
    assert not res.memory_bounds(sample(12 * GiB + 1, int(.46 * GiB)), 0, 12 * GiB)['pass']
    assert not res.memory_bounds(sample(5 * GiB, 4 * GiB + 1), 0, 12 * GiB)['pass']          # private stays 4 GiB
    assert not res.memory_bounds(sample(5 * GiB, None, fallback=True), 0, 12 * GiB)['pass']  # unavailable fails
    with pytest.raises(res.MemoryBoundsError):
        res.enforce_memory(sample(12 * GiB + 1, int(.46 * GiB)), 0, 12 * GiB)


def test_raised_limit_is_s1_only():
    assert driver.stage_rss_limit('S1') == 12 * GiB
    assert driver.stage_min_available('S1') == 12 * GiB + res.MIN_AVAILABLE
    for stage in ('S2t', 'S2v', 'S3', None):
        assert driver.stage_rss_limit(stage) == res.PRIVATE_BUDGET
        assert driver.stage_min_available(stage) == res.MIN_AVAILABLE


def test_memory_plan_minimum_available_is_a_parameter():
    assert res.memory_plan([{}], available_bytes=19 * GiB)['pass']
    report = res.memory_plan([{}], available_bytes=19 * GiB, min_available=20 * GiB)
    assert not report['pass'] and report['minimum_available_bytes'] == 20 * GiB
    with pytest.raises(res.MemoryAdmissionError):
        res.require_memory_plan([{}], min_available=10**18)


class Fatal(BaseException):
    """phase4_fatal calls os._exit, which no retry loop can catch; a BaseException mirrors that."""


@pytest.fixture
def fatal(monkeypatch):
    def raise_(message, *a, **k):
        raise Fatal(message)
    monkeypatch.setattr(opt, 'phase4_fatal', raise_)


def test_live_watchdog_uses_the_stage_limit(tmp_path, monkeypatch, fatal):
    layout = PhaseLayout('phase4b-s1', root=tmp_path, approvals_base=tmp_path / 'approvals')
    monkeypatch.setattr(res, 'current_memory', lambda: sample(int(4.38 * GiB), int(.46 * GiB)))
    with driver.file_timing(layout, 'd' * 64, 'S1-a', 600, rss_limit=12 * GiB):
        pass
    with pytest.raises(Fatal, match='memory bounds'):
        with driver.file_timing(layout, 'd' * 64, 'S1-b', 600):
            pass


def test_s1_replay_guards_and_arm_opens_use_the_s1_limit(tmp_path, monkeypatch):
    seen = {'timing': [], 'open': []}
    @contextmanager
    def timing(layout, digest, label, ceiling, *, audit=None, rss_limit=None):
        seen['timing'].append(rss_limit); yield
    def open_prefix(layout, manifest, bt, stage=None):
        seen['open'].append(stage); return object(), object(), None
    def run_s1(prefix, comparator, *, guard=None):
        with guard('defaults'):
            return {'ok': True}
    monkeypatch.setattr(driver, 'file_timing', timing)
    monkeypatch.setattr(driver, 'open_prefix', open_prefix)
    monkeypatch.setattr(driver, 'stage_progress', lambda *a, **k: None)
    monkeypatch.setattr(driver.p4, 'run_s1', run_s1)
    manifest = {'arms': {f'BTCUSDT:{bt}:full': {'max_trial_seconds': 1800} for bt in driver.BAR_TYPES}}
    driver.execute_stage('S1', manifest, 'd' * 64, PhaseLayout('phase4b-s1', root=tmp_path))
    assert seen == {'timing': [12 * GiB, 12 * GiB], 'open': ['S1', 'S1']}


def test_arm_open_point_check_uses_the_stage_limit_first(monkeypatch):
    calls = []
    def enforce_current(mapped=0, rss_limit=res.PRIVATE_BUDGET):
        calls.append(rss_limit); raise RuntimeError('checked')
    monkeypatch.setattr(res, 'enforce_current', enforce_current)
    for stage, limit in (('S1', 12 * GiB), ('S2t', 4 * GiB)):
        with pytest.raises(RuntimeError, match='checked'):
            driver.open_prefix(None, {}, 'fixed_tick_imbalance', stage=stage)
    assert calls == [12 * GiB, 4 * GiB]


def test_every_production_file_timing_call_passes_the_stage_limit():
    tree = ast.parse(Path(driver.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'file_timing']
    assert len(calls) == 2
    for call in calls:
        value = next(k.value for k in call.keywords if k.arg == 'rss_limit')
        assert (isinstance(value, ast.Call) and value.func.id == 'stage_rss_limit'
                and [a.id for a in value.args] == ['stage'])


def test_post_exit_check_enforces_the_stage_limit(monkeypatch):
    seen = []
    monkeypatch.setattr(res, 'enforce_memory', lambda m, mapped=0, rss_limit=res.PRIVATE_BUDGET: seen.append(rss_limit))
    driver.enforce_stage_memory({'peak_rss_bytes': 1}, 'S1')
    driver.enforce_stage_memory({'peak_rss_bytes': 1}, 'S2t')
    assert seen == [12 * GiB, 4 * GiB]
    tree = ast.parse(Path(driver.__file__).read_text())
    launch = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'launch_stage')
    names = {getattr(n.func, 'id', getattr(n.func, 'attr', None)) for n in ast.walk(launch) if isinstance(n, ast.Call)}
    assert 'enforce_stage_memory' in names and 'enforce_memory' not in names


def driver_fixture(tmp_path, monkeypatch, available, phase='phase4b-s1', stage='S1'):
    from scripts import phase4_approvals as approvals
    layout = PhaseLayout(phase, root=tmp_path, approvals_base=tmp_path / 'approvals')
    events = []
    monkeypatch.setattr(driver, 'PhaseLayout', lambda *a: layout)
    monkeypatch.setattr(driver, 'read_manifest', lambda layout: ({'stages': [stage]}, 'd' * 64))
    monkeypatch.setattr(driver.shutil, 'which', lambda *a: '/synthetic/timeout')
    @contextmanager
    def locks(layout, manifest):
        yield {}
    monkeypatch.setattr(driver, 'stage_locks', locks)
    monkeypatch.setattr(driver, 'termination_admission', lambda *a: None)
    layout.approval_root.mkdir(parents=True)
    (layout.approval_root / ('a' * 64 + '.json')).write_text('{}')
    monkeypatch.setattr(approvals, 'verify_phase4b_approval', lambda *a: None)
    monkeypatch.setattr(res, 'available_memory', lambda: (available, 'test'))
    monkeypatch.setattr(driver, 'require_phase4b_approval', lambda *a, **k: events.append('consume'))
    monkeypatch.setattr(driver, 'launch_stage', lambda *a, **k: events.append('launch'))
    def recovery(*a):
        events.append('recovery'); raise ValueError('recovery reached')
    monkeypatch.setattr(driver, 'recovery_admission', recovery)
    return events


@pytest.mark.parametrize('available,admitted', [(19 * GiB, False), (21 * GiB, True)])
def test_driver_checks_host_memory_before_consuming_the_s1_approval(tmp_path, monkeypatch, capsys, available,
                                                                     admitted):
    events = driver_fixture(tmp_path, monkeypatch, available)
    if admitted:
        driver.main(['run', '--phase', 'phase4b-s1', '--stage', 'S1', '--owner-approved', 'b' * 64])
        assert events == ['consume', 'launch']
    else:
        with pytest.raises(SystemExit):
            driver.main(['run', '--phase', 'phase4b-s1', '--stage', 'S1', '--owner-approved', 'b' * 64])
        assert events == [] and 'total-memory admission refused' in capsys.readouterr().err


@pytest.mark.parametrize('available,reached', [(19 * GiB, False), (21 * GiB, True)])
def test_s1_recovery_path_checks_host_memory_before_consuming_the_recovery_approval(tmp_path, monkeypatch, capsys,
                                                                                    available, reached):
    events = driver_fixture(tmp_path, monkeypatch, available)
    with pytest.raises(SystemExit):
        driver.main(['run', '--phase', 'phase4b-s1', '--stage', 'S1', '--recovery-approved', 'r' * 64])
    err = capsys.readouterr().err
    assert events == (['recovery'] if reached else [])
    assert not list((tmp_path / 'approvals').rglob('*.admitted.json'))
    assert ('recovery reached' in err) == reached and ('total-memory admission refused' in err) != reached


def test_other_stages_have_no_driver_memory_admission(tmp_path, monkeypatch):
    events = driver_fixture(tmp_path, monkeypatch, 1 * GiB, phase='phase4b', stage='S2t')
    driver.main(['run', '--phase', 'phase4b', '--stage', 'S2t', '--owner-approved', 'b' * 64])
    assert events == ['consume', 'launch']

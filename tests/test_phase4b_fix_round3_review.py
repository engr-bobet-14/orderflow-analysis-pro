"""Interim Superpowers review of fix round 3: confirmed findings (critical 1, important 2-6, minors)."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
from datetime import datetime, timezone

import pytest

from scripts import phase4b_campaign as driver, phase4_approvals as approvals, phase4_policy as policy
from scripts import optimize_orderflow as opt, phase4_campaign as campaign
from tests.test_phase4b_fix_round1 import runtime_fixture
from tests.test_phase4b_fix_round3 import launched, recovery_record, synthetic_study, postflight_fixture


def at(monkeypatch, value):
    monkeypatch.setattr(driver, '_now', lambda: value)


# ---- Critical 1: GNU timeout kill-after reaches Popen as -SIGKILL, never 137.
@pytest.mark.parametrize('code,elapsed,kind', [
    # -SIGKILL is never first-party evidence: any killer produces it. timeout --foreground reports its own
    # kill-after as exit 137 (pre-inspection -9 investigation, GNU coreutils 9.12 on darwin).
    (-signal.SIGKILL, 630., 'external_or_unknown_termination'),
    (-signal.SIGKILL, 10_000., 'external_or_unknown_termination'),
    (137, 630., 'timeout_confirmed'),
    (-signal.SIGKILL, 629., 'external_or_unknown_termination'),
    (-signal.SIGKILL, 610., 'external_or_unknown_termination'),
    (124, 600., 'timeout_confirmed'), (124, 599., 'external_or_unknown_termination')])
def test_c1_kill_after_sigkill_after_deadline_is_confirmed(tmp_path, monkeypatch, code, elapsed, kind):
    layout, manifest, digest = launched(tmp_path, monkeypatch)
    at(monkeypatch, 10_000.)  # Record time is irrelevant: the driver passes elapsed observed at child exit.
    record = driver.record_termination(layout, manifest, digest, 'S2t', returncode=code, elapsed=elapsed)
    assert record['kind'] == kind and record['elapsed_seconds'] == elapsed
    if kind != 'timeout_confirmed':
        assert record['cause'] == 'unknown'


def test_c1_driver_measures_elapsed_at_child_exit_not_after_scans(tmp_path, monkeypatch):
    from scripts import measure_rss
    layout, manifest, digest = launched(tmp_path, monkeypatch)
    clock = {'now': 100.}
    monkeypatch.setattr(driver, '_now', lambda: clock['now'])
    launch = {'utc': 'u', 'monotonic': 100., 'attempt': 'exit-clock'}
    def measured(argv, **kw):
        clock['now'] = 100. + 630.;return {'returncode': 137}  # timeout --foreground kill-after
    def slow_scan(*a):
        clock['now'] += 5_000.;return None
    monkeypatch.setattr(measure_rss, 'run_measured', measured)
    monkeypatch.setattr(driver, 'first_party_memory_stop', slow_scan)
    with pytest.raises(ValueError, match='timeout_confirmed'):
        driver.launch_stage(layout, manifest, digest, 'S2t', ['worker'], {}, launch, '/synthetic/timeout')
    assert driver.recover_termination_audit(layout, digest, 'S2t')['elapsed_seconds'] == 630.


# ---- Important 2: the open_study admission path must not promote the pending estimate.
def test_i2_prepare_before_open_study_admission_never_promotes(tmp_path):
    from scripts import phase4b_accounting as accounting
    study = synthetic_study(tmp_path)
    study.set_user_attr('phase4_accounting_pending', {'trial': 1, 'wall_seconds': 540.,
        'heartbeat_utc': '2026-11-07T00:09:00+00:00', 'started_utc': '2026-11-07T00:00:00+00:00',
        'runtime_seconds_before': 0., 'stale_bound': 90.,
        'limits': {'max_trial_seconds': 600., 'max_study_seconds': 2000.}})
    study.set_user_attr('identity', {'phase': 'phase4b'})
    db = tmp_path / 'synthetic.db'
    accounting.prepare(db, 'same-study')
    attrs = opt.read_study_attrs(db, 'same-study')
    assert attrs.get('phase4_accounting_pending') is None and attrs.get('phase4_runtime_stop') is None
    import optuna
    reloaded = optuna.load_study(storage='sqlite:///' + str(db), study_name='same-study')
    assert opt.phase4_runtime_baseline(reloaded) == 630. and reloaded.trials[1].state.name == 'FAIL'


# ---- Important 3: a production worker requires the driver launch and proof of inherited locks.
def test_i3_production_worker_without_driver_launch_refused(tmp_path, monkeypatch):
    layout, manifest, digest = postflight_fixture(tmp_path, monkeypatch)
    monkeypatch.delenv(driver.LAUNCH_ENV, raising=False)
    with pytest.raises(ValueError, match='driver'):
        driver.run_stage(layout, 'S2t', 's2-approval')


def test_i3_inherited_lock_check_requires_the_lock_to_be_held(tmp_path, monkeypatch):
    from tests.test_phase4b_fix_round3 import origin_lock_fixture
    layout, origin, path = origin_lock_fixture(tmp_path, monkeypatch)
    layout.checkpoints.mkdir(parents=True, exist_ok=True)
    fds = {}
    for name, target in (('phase4', path), ('execution', layout.checkpoints / 'execution.lock'),
                         ('driver', layout.checkpoints / 'driver.lock')):
        if not target.exists(): target.write_bytes(b'')
        fds[name] = os.open(target, os.O_RDONLY)  # Opened but NOT locked by anyone.
    monkeypatch.setenv(driver.LOCK_ENV, json.dumps(fds))
    try:
        with pytest.raises(ValueError, match='not held|inherit'):
            driver.require_inherited_locks(layout, {'phase4_origin': origin})
        with driver.stage_locks(layout, {'phase4_origin': origin}) as held:
            monkeypatch.setenv(driver.LOCK_ENV, json.dumps(held))
            driver.require_inherited_locks(layout, {'phase4_origin': origin})
    finally:
        for fd in fds.values(): os.close(fd)


# ---- Important 4: every non-successful end after the timed child is classified.
def driver_fixture(tmp_path, monkeypatch):
    from scripts import measure_rss, phase4_resources
    layout, manifest, digest = postflight_fixture(tmp_path, monkeypatch)
    launch = {'utc': 'u', 'monotonic': driver._now(), 'attempt': 'post'}
    return layout, manifest, digest, launch, measure_rss, phase4_resources


def test_i4_driver_memory_breach_after_rc0_is_strict_memory_stop(tmp_path, monkeypatch):
    layout, manifest, digest, launch, measure_rss, resources = driver_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(measure_rss, 'run_measured', lambda *a, **k: {'returncode': 0})
    def breach(measurement): raise resources.MemoryBoundsError(measurement, {'pass': False, 'rss': 'above 4 GiB'})
    monkeypatch.setattr(resources, 'enforce_memory', breach)
    with pytest.raises((ValueError, resources.MemoryBoundsError)):
        driver.launch_stage(layout, manifest, digest, 'S2t', ['worker'], {}, launch, '/t')
    assert driver.recover_termination_audit(layout, digest, 'S2t')['kind'] == 'strict_memory_stop'


def test_i4_postflight_child_crash_is_classified_unknown(tmp_path, monkeypatch):
    layout, manifest, digest, launch, measure_rss, resources = driver_fixture(tmp_path, monkeypatch)
    codes = iter([0, -signal.SIGKILL])
    monkeypatch.setattr(measure_rss, 'run_measured', lambda *a, **k: {'returncode': next(codes)})
    monkeypatch.setattr(resources, 'enforce_memory', lambda *a: None)
    with pytest.raises(ValueError, match='post-flight'):
        driver.launch_stage(layout, manifest, digest, 'S2t', ['worker'], {}, launch, '/t')
    record = driver.recover_termination_audit(layout, digest, 'S2t')
    assert record['kind'] == 'external_or_unknown_termination' and record['attempt'] == 'post-postflight'


def test_i4_pending_result_reported_and_recovery_runs_only_postflight(tmp_path, monkeypatch):
    layout, manifest, digest, launch, measure_rss, resources = driver_fixture(tmp_path, monkeypatch)
    driver.run_stage(layout, 'S2t', 's2-approval', operation=lambda *a: {'synthetic': True}, finalize=False)
    assert driver.s2_status(layout, digest, driver.BAR_TYPES[0]) == 'pending_postflight'
    commands = []
    monkeypatch.setattr(measure_rss, 'run_measured', lambda c, **k: (commands.append(c) or {'returncode': 0}))
    monkeypatch.setattr(resources, 'enforce_memory', lambda *a: None)
    driver.launch_stage(layout, manifest, digest, 'S2t', ['worker'], {}, launch, '/t', postflight_only=True)
    assert len(commands) == 1 and 'postflight' in commands[0]


def test_i4_orphaned_started_attempt_requires_recovery(tmp_path, monkeypatch):
    layout, manifest, digest, *_ = driver_fixture(tmp_path, monkeypatch)
    started = layout.checkpoints / digest / 'attempts' / 'orphan.json'
    started.parent.mkdir(parents=True, exist_ok=True)
    started.write_bytes(policy.canonical_bytes({'stage': 'S2t', 'status': 'started', 'attempt': 'lost-driver',
                                                'campaign_sha256': digest}))
    with pytest.raises(ValueError, match='recovery'):
        driver.termination_admission(layout, manifest, digest, 'S2t', None, ['driver'])
    record = driver.recover_termination_audit(layout, digest, 'S2t')
    assert record['kind'] == 'external_or_unknown_termination' and record['attempt'] == 'lost-driver'


# ---- Important 5: recovery re-verifies the host proof gates.
def test_i5_recovery_admission_verifies_gates(tmp_path, monkeypatch):
    layout, manifest, digest = launched(tmp_path, monkeypatch)
    stop = driver.record_termination(layout, manifest, digest, 'S2t', returncode=-9)
    argv = ['driver']
    approvals.install_recovery_approval(layout, recovery_record(layout, digest, 'S2t', stop, argv))
    selector = next((layout.approval_root / 'recovery').glob('*.json')).stem
    with pytest.raises(ValueError, match='gates'):
        driver.recovery_admission(layout, manifest, digest, 'S2t', selector, argv)
    assert not list((layout.approval_root / 'recovery').glob('*.consumed'))


# ---- Important 6: the SIGTERM handler never writes; below the ceiling it is an unknown termination.
def test_i6_sigterm_below_ceiling_is_unknown_and_never_deadlocks(tmp_path):
    script = r'''
import sys,time,threading
from pathlib import Path
from scripts import phase4b_campaign as d,phase4_policy as p,phase4_resources as r
layout=p.PhaseLayout('phase4b',root=Path(sys.argv[1]),approvals_base=Path(sys.argv[1])/'approvals')
m={'_launch':{'utc':'u','monotonic':d._now(),'attempt':'term'},'max_stage_seconds':{'S2t':60.},'runtime_ceilings':{'inputs':{}}}
r.current_memory=lambda:{'rss_bytes':1,'private_bytes':1}
lock=layout.checked_write(layout.checkpoints/('a'*64)/'terminations'/'S2t'/'termination.lock')
lock.parent.mkdir(parents=True,exist_ok=True)
with d.stage_watchdog(layout,m,'a'*64,'S2t'):
    from scripts import optimize_orderflow as o
    with o.file_lock(lock):  # The main thread is inside a termination write when SIGTERM lands.
        print('ready',flush=True);time.sleep(.5)
    time.sleep(5)
'''
    proc = subprocess.Popen([sys.executable, '-c', script, str(tmp_path)], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == 'ready'
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=10) == 70
    layout = policy.PhaseLayout('phase4b', root=tmp_path, approvals_base=tmp_path / 'approvals')
    record = driver.recover_termination_audit(layout, 'a' * 64, 'S2t')
    assert record['kind'] == 'external_or_unknown_termination' and record['observed_signal'] == 'SIGTERM'


# ---- Minors.
def test_m1_exclusive_write_is_atomic(tmp_path, monkeypatch):
    layout, *_ = runtime_fixture(tmp_path)
    target = layout.checkpoints / 'x' / 'record.json'
    monkeypatch.setattr(driver.os, 'fsync', lambda fd: (_ for _ in ()).throw(OSError('disk stall')))
    with pytest.raises(OSError):
        driver.write_json(layout, target, {'a': 1}, exclusive=True)
    assert not target.exists()


def test_m2_final_record_without_pending_marker_is_refused(tmp_path, monkeypatch):
    layout, manifest, digest = launched(tmp_path, monkeypatch)
    driver.record_termination(layout, manifest, digest, 'S2t', returncode=-9)
    next((layout.checkpoints / digest / 'terminations' / 'S2t').glob('*.pending.json')).unlink()
    with pytest.raises(ValueError, match='pending'):
        driver.terminations(layout, digest, 'S2t')


def test_m3_integrity_record_keeps_cause_and_real_launch(tmp_path, monkeypatch):
    layout, manifest, digest = postflight_fixture(tmp_path, monkeypatch)
    real = {'utc': '2026-11-07T00:00:00+00:00', 'monotonic': driver._now(), 'attempt': 'real-launch'}
    monkeypatch.setenv(driver.LAUNCH_ENV, json.dumps(real))
    driver.run_stage(layout, 'S2t', 's2-approval', operation=lambda *a: {'synthetic': True}, finalize=False)
    tampered = Path(manifest['snapshots'][0]['path']);tampered.write_bytes(tampered.read_bytes() + b'x')
    with pytest.raises(ValueError):driver.postflight_stage(layout, 'S2t', synthetic=True)
    record = driver.recover_termination_audit(layout, digest, 'S2t')
    assert record['L'] == real and 'protected' in record['cause']


def test_m4_production_postflight_ignores_snapshot_mode_in_pending_file(tmp_path, monkeypatch):
    layout, manifest, digest = postflight_fixture(tmp_path, monkeypatch)
    driver.run_stage(layout, 'S2t', 's2-approval', operation=lambda *a: {'synthetic': True}, finalize=False)
    called = []
    monkeypatch.setattr(driver, 'protected_hashes', lambda *a: (called.append(1) or {}))
    with pytest.raises(ValueError):driver.postflight_stage(layout, 'S2t')  # production: full verification
    assert called


def test_m5_watchdog_audit_write_is_retried(tmp_path, monkeypatch):
    layout, manifest, digest = launched(tmp_path, monkeypatch)
    original = driver.record_termination;calls = []
    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1: raise OSError('transient')
        return original(*a, **k)
    monkeypatch.setattr(driver, 'record_termination', flaky)
    driver.durable_audit(layout, manifest, digest, 'S2t', stop={'reason': 'max_stage_seconds',
        'measured_seconds': 600., 'limit_seconds': 600., 'live': True})
    assert driver.recover_termination_audit(layout, digest, 'S2t')['kind'] == 'timeout_confirmed'


def test_m6_consumed_marker_without_matching_approval_is_ignored(tmp_path, monkeypatch):
    layout, manifest, digest = launched(tmp_path, monkeypatch)
    stop = driver.record_termination(layout, manifest, digest, 'S2t', returncode=-9)
    root = layout.approval_root / 'recovery';root.mkdir(parents=True, mode=0o700)
    forged = root / ('f' * 64 + '.consumed')
    forged.write_bytes(policy.canonical_bytes({'termination_sha256': hashlib.sha256(
        policy.canonical_bytes(stop)).hexdigest()}));forged.chmod(0o600)
    assert approvals.consumed_recoveries(layout) == {}
    with pytest.raises(ValueError, match='recovery'):
        driver.termination_admission(layout, manifest, digest, 'S2t', None, ['driver'])


def test_m7_parent_approval_not_consumed_when_locks_unavailable(tmp_path, monkeypatch):
    layout, manifest, digest = runtime_fixture(tmp_path)
    layout.approval_root.mkdir(parents=True);(layout.approval_root / ('a' * 64 + '.json')).write_text('{}')
    monkeypatch.setattr(driver, 'PhaseLayout', lambda *a: layout)
    monkeypatch.setattr(driver, 'read_manifest', lambda layout: (manifest, digest))
    monkeypatch.setattr(driver.shutil, 'which', lambda *a: '/t')
    monkeypatch.setattr(approvals, 'verify_phase4b_approval', lambda *a: None)
    consumed = []
    monkeypatch.setattr(driver, 'require_phase4b_approval', lambda *a, **k: consumed.append(k.get('consume')))
    @contextmanager
    def busy(layout, manifest):
        raise ValueError('canonical Phase 4 execution lock is held');yield
    monkeypatch.setattr(driver, 'stage_locks', busy)
    with pytest.raises(SystemExit):
        driver.main(['run', '--phase', 'phase4b', '--stage', 'S3', '--owner-approved', 'c' * 64])
    assert True not in consumed


def test_m8_study_stop_records_attempt_start_and_prior_basis(tmp_path, monkeypatch):
    from scripts import phase4b_accounting as accounting
    monkeypatch.setattr(opt.time, 'perf_counter', lambda: 30.)
    study = synthetic_study(tmp_path);seen = []
    accounting.reconcile(study, SimpleNamespace(phase4=True))
    args = SimpleNamespace(max_trial_seconds=None, max_study_seconds=600., phase4_termination_audit=seen.append)
    assert accounting.runtime_stop(study, args, 660., prior=630., begun=0.)
    boundaries = seen[0]['boundaries']
    assert boundaries['current_attempt_started_perf_counter'] == 0.
    assert boundaries['prior_charge_basis'] == 'upper_bound'


def test_m9_stage_clock_counts_host_sleep():
    import time
    assert driver._now.__name__ == '_now'
    if sys.platform == 'darwin':
        # CLOCK_MONOTONIC on darwin continues across sleep; time.monotonic (mach_absolute_time) does not.
        assert abs(driver._now() - time.clock_gettime(time.CLOCK_MONOTONIC)) < 1.


# ---- Test gap: end-to-end owner recovery through the driver.
def test_gap_main_recovery_path_admits_once_and_launches_recovery_child(tmp_path, monkeypatch):
    from scripts import measure_rss, phase4_resources
    layout, manifest, digest = postflight_fixture(tmp_path, monkeypatch)
    manifest_launch = {**manifest, '_launch': {'utc': 'u', 'monotonic': driver._now(), 'attempt': 'first'}}
    stop = driver.record_termination(layout, manifest_launch, digest, 'S2t', returncode=-9)
    argv = ['orderflow', 'run', '--phase', 'phase4b', '--stage', 'S2t']
    approvals.install_recovery_approval(layout, recovery_record(layout, digest, 'S2t', stop, argv))
    selector = next((layout.approval_root / 'recovery').glob('*.json')).stem
    monkeypatch.setattr(driver, 'PhaseLayout', lambda *a: layout)
    monkeypatch.setattr(driver.shutil, 'which', lambda *a: '/t')
    monkeypatch.setattr(driver, 'verify_phase4b_gates', lambda *a: 'gates-sha')
    @contextmanager
    def locks(layout, manifest): yield {}
    monkeypatch.setattr(driver, 'stage_locks', locks)
    monkeypatch.setattr(phase4_resources, 'enforce_memory', lambda *a: None)
    commands = []
    monkeypatch.setattr(measure_rss, 'run_measured', lambda c, **k: (commands.append(c) or {'returncode': 0}))
    driver.main([*argv[1:], '--recovery-approved', selector], recovery_argv=argv)
    assert '--recovery' in commands[0] and commands[0][commands[0].index('--recovery') + 1] == selector
    marker = json.loads((layout.approval_root / 'recovery' / f'{selector}.admitted.json').read_bytes())
    assert marker['gates_sha256'] == 'gates-sha' and marker['stage'] == 'S2t'
    with pytest.raises(SystemExit):
        driver.main([*argv[1:], '--recovery-approved', selector], recovery_argv=argv)


# ---- Repeat review: N1/N2 and minors.
def test_n1_postflight_kill_is_never_a_confirmed_timeout(tmp_path, monkeypatch):
    layout, manifest, digest, launch, measure_rss, resources = driver_fixture(tmp_path, monkeypatch)
    manifest['max_stage_seconds']['S2t'] = 600.
    launch = {**launch, 'monotonic': 0.}
    clock = {'now': 0.}
    monkeypatch.setattr(driver, '_now', lambda: clock['now'])
    def measured(command, **kw):
        clock['now'] += 300. if 'worker' in command else 400.  # post-flight ends at L+700 > ceiling
        return {'returncode': 0 if 'worker' in command else -signal.SIGKILL}
    monkeypatch.setattr(measure_rss, 'run_measured', measured)
    monkeypatch.setattr(resources, 'enforce_memory', lambda *a: None)
    with pytest.raises(ValueError, match='post-flight'):
        driver.launch_stage(layout, manifest, digest, 'S2t', ['worker'], {}, launch, '/t')
    record = driver.recover_termination_audit(layout, digest, 'S2t')
    assert record['kind'] == 'external_or_unknown_termination' and record['cause'] == 'unknown'


def test_n2_integrity_failure_in_postflight_only_recovery_is_consuming(tmp_path, monkeypatch):
    layout, manifest, digest = postflight_fixture(tmp_path, monkeypatch)
    first = {'utc': 'u', 'monotonic': driver._now(), 'attempt': 'A'}
    monkeypatch.setenv(driver.LAUNCH_ENV, json.dumps(first))
    driver.run_stage(layout, 'S2t', 's2-approval', operation=lambda *a: {'synthetic': True}, finalize=False)
    # The original driver recorded the first post-flight child's kill under A-postflight.
    driver.record_termination(layout, {**manifest, '_launch': first}, digest, 'S2t', returncode=-signal.SIGKILL,
                              attempt='A-postflight', elapsed=None)
    recovery = {'utc': 'u', 'monotonic': driver._now(), 'attempt': 'C'}
    monkeypatch.setenv(driver.LAUNCH_ENV, json.dumps(recovery))
    tampered = Path(manifest['snapshots'][0]['path']);tampered.write_bytes(tampered.read_bytes() + b'x')
    with pytest.raises(ValueError, match='integrity'):driver.postflight_stage(layout, 'S2t', synthetic=True)
    kinds = [r['kind'] for r in driver.terminations(layout, digest, 'S2t')]
    assert 'integrity_stop' in kinds
    with pytest.raises(ValueError, match='consumed'):
        driver.termination_admission(layout, manifest, digest, 'S2t', None, ['driver'])


def test_n_minor_gates_rehash_the_plan_file(tmp_path, monkeypatch):
    layout, manifest, digest = runtime_fixture(tmp_path)
    fake = tmp_path / 'plan.md';fake.write_text('changed plan')
    monkeypatch.setattr(approvals, 'PHASE4B_PLAN_PATH', fake)
    with pytest.raises(ValueError, match='plan'):
        approvals.verify_phase4b_gates(layout, digest)


def test_n_minor_durable_audit_retries_any_transient_error(tmp_path, monkeypatch):
    layout, manifest, digest = launched(tmp_path, monkeypatch)
    original = driver.record_termination;calls = []
    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1: raise RuntimeError('transient')
        return original(*a, **k)
    monkeypatch.setattr(driver, 'record_termination', flaky)
    record = driver.durable_audit(layout, manifest, digest, 'S2t', stop={'reason': 'max_stage_seconds',
        'measured_seconds': 600., 'limit_seconds': 600., 'live': True})
    assert record['kind'] == 'timeout_confirmed'


def test_n_minor_prepare_only_touches_phase4b_studies(tmp_path):
    from scripts import phase4b_accounting as accounting
    study = synthetic_study(tmp_path)
    study.set_user_attr('phase4_accounting_pending', {'trial': 1, 'wall_seconds': 540.,
        'heartbeat_utc': '2026-11-07T00:09:00+00:00', 'started_utc': '2026-11-07T00:00:00+00:00',
        'runtime_seconds_before': 0., 'stale_bound': 90., 'limits': {}})
    study.set_user_attr('identity', {'phase': 'phase4'})
    assert accounting.prepare(tmp_path / 'synthetic.db', 'same-study') is False
    assert opt.read_study_attrs(tmp_path / 'synthetic.db', 'same-study').get('phase4_accounting_pending')


# ---- Pre-inspection verification: rev 13 C7 audit fields require the trial-start boundary.
def test_v1_live_trial_timeout_records_trial_start_from_its_own_stop(tmp_path, monkeypatch):
    from tests.test_phase4b_fix_round3 import launched as launched3
    layout, manifest, digest = launched3(tmp_path, monkeypatch)
    record = driver.record_termination(layout, manifest, digest, 'S2t', stop={
        'reason': 'max_trial_seconds', 'measured_seconds': 30., 'limit_seconds': 30., 'live': True, 'trial': 3,
        'utc': '2026-11-07T00:00:50+00:00'})
    assert record['kind'] == 'timeout_confirmed'
    assert record['boundaries']['trial_started_utc'] == '2026-11-07T00:00:20+00:00'
    assert record['boundaries']['current_seconds'] == 30.


def test_v1_replayed_stop_from_earlier_attempt_keeps_its_own_trial_start(tmp_path, monkeypatch):
    # A persisted live stop replayed under a new launch L: the trial start must not mix the two attempts.
    from tests.test_phase4b_fix_round3 import launched as launched3
    layout, manifest, digest = launched3(tmp_path, monkeypatch)  # L.utc = 2026-11-07T00:00:00
    record = driver.record_termination(layout, manifest, digest, 'S2t', stop={
        'reason': 'max_trial_seconds', 'measured_seconds': 600., 'limit_seconds': 600., 'live': True, 'trial': 3,
        'utc': '2026-11-06T12:10:00+00:00'})
    assert record['boundaries']['trial_started_utc'] == '2026-11-06T12:00:00+00:00'
    assert 'trial_started_from_L_seconds' not in record['boundaries']


def test_v1_reconciled_upper_bound_stop_gets_no_derived_trial_start(tmp_path, monkeypatch):
    from tests.test_phase4b_fix_round3 import launched as launched3
    layout, manifest, digest = launched3(tmp_path, monkeypatch)
    record = driver.record_termination(layout, manifest, digest, 'S2t', stop={
        'reason': 'max_trial_seconds', 'measured_seconds': 630., 'limit_seconds': 600., 'live': False,
        'reconciled': True, 'trial': 3, 'utc': '2026-11-07T00:00:50+00:00'})
    assert record['kind'] == 'external_or_unknown_termination'
    assert not {'trial_started_utc', 'trial_started_from_L_seconds'} & set(record['boundaries'])


def test_v1_terminal_trial_overrun_records_trial_start_utc(tmp_path, monkeypatch):
    from scripts import phase4b_accounting as accounting
    study = synthetic_study(tmp_path)
    trial = next(t for t in study.trials if t.state.is_finished())
    study.set_user_attr('phase4_terminal_wall_seconds', {str(trial.number): 700.})
    seen = []
    args = SimpleNamespace(max_trial_seconds=600., max_study_seconds=None, phase4_termination_audit=seen.append)
    assert accounting.runtime_stop(study, args, 0., prior=0., begun=0.)
    assert seen[0]['trial'] == trial.number
    started = seen[0]['boundaries']['trial_started_utc']
    # Optuna stores naive local time; the audit field is UTC (review I1).
    assert started.endswith('+00:00')
    assert started == trial.datetime_start.astimezone(timezone.utc).isoformat()
    assert datetime.fromisoformat(started) <= datetime.fromisoformat(seen[0]['utc'])


# ---- Pre-inspection: macOS -9 ambiguity. Real GNU timeout, shortened kill-after.
def _real_launch(tmp_path, monkeypatch, body):
    import shutil, time as _time
    timeout = shutil.which('timeout') or shutil.which('gtimeout')
    if not timeout:
        pytest.skip('GNU timeout not installed')
    from tests.test_phase4b_fix_round3 import launched as launched3
    layout, manifest, digest = launched3(tmp_path, monkeypatch)
    monkeypatch.setattr(driver, '_now', lambda: _time.clock_gettime(driver._CLOCK))
    monkeypatch.setattr(driver, 'KILL_AFTER_SECONDS', 1)
    manifest['max_stage_seconds']['S2t'] = 1.
    launch = {'utc': '2026-11-08T00:00:00+00:00', 'monotonic': driver._now(), 'attempt': 'real-timeout'}
    child = [sys.executable, '-c', body]
    with pytest.raises(ValueError, match='stage process ended'):
        driver.launch_stage(layout, manifest, digest, 'S2t', child, {}, launch, timeout)
    return driver.recover_termination_audit(layout, digest, 'S2t')


IGNORE_TERM = 'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);'


def test_m9_real_timeout_kill_after_is_observed_137_and_confirmed(tmp_path, monkeypatch):
    record = _real_launch(tmp_path, monkeypatch, IGNORE_TERM + 'time.sleep(20)')
    assert record['observed_exit_code'] == 137 and record['kind'] == 'timeout_confirmed'


def test_m9_real_external_sigkill_after_deadline_term_is_unknown(tmp_path, monkeypatch):
    # TERM at the 1 s deadline is ignored; an outside SIGKILL at 1.4 s precedes the 2 s kill-after.
    record = _real_launch(tmp_path, monkeypatch,
                          IGNORE_TERM + 'import os;time.sleep(1.4);os.kill(os.getpid(),signal.SIGKILL)')
    assert record['kind'] == 'external_or_unknown_termination' and record['cause'] == 'unknown'


def test_m9_real_external_sigkill_before_deadline_is_unknown(tmp_path, monkeypatch):
    record = _real_launch(tmp_path, monkeypatch, 'import os,signal;os.kill(os.getpid(),signal.SIGKILL)')
    assert record['observed_exit_code'] == -signal.SIGKILL
    assert record['kind'] == 'external_or_unknown_termination' and 'oom' not in json.dumps(record).lower()


# ---- Review I2: the memory evidence describes the worker, not the timeout wrapper.
def test_i2_real_kill_records_worker_memory_sample(tmp_path, monkeypatch):
    record = _real_launch(tmp_path, monkeypatch,
        'import os,signal,time;b=bytearray(300<<20);b[::4096]=b"x"*len(b[::4096]);time.sleep(.5);'
        'os.kill(os.getpid(),signal.SIGKILL)')
    sample = record['last_memory_sample']
    assert sample['source'] == 'worker' and sample['pid'] != sample['wrapper_pid']
    assert sample['last']['private_bytes'] >= 200 << 20
    assert sample['peak_private_bytes'] >= 200 << 20


def test_i2_run_measured_default_is_unchanged_for_phase4_callers():
    from scripts import measure_rss
    result = measure_rss.run_measured([sys.executable, '-c', 'pass'], memory=True)
    assert set(result) == {'argv', 'returncode', 'peak_rss_bytes', 'wall_seconds', 'peak_private_bytes',
                           'private_metric', 'private_is_fallback', 'memory_samples'}


def test_i2_zombie_sample_never_replaces_last_live_sample(monkeypatch):
    # A SIGKILLed worker is reaped late: proc_pid_rusage still succeeds, keeps resident_size but reports a
    # 0 phys_footprint (measured on darwin: rss 314 MiB, private 0, status idle).
    import shutil
    from scripts import measure_rss, phase4_resources
    timeout = shutil.which('timeout') or shutil.which('gtimeout')
    if not timeout:
        pytest.skip('GNU timeout not installed')
    live = {'rss_bytes': 300 << 20, 'private_bytes': 290 << 20, 'peak_rss_bytes': 300 << 20,
            'peak_private_bytes': 295 << 20, 'private_metric': 'phys_footprint', 'private_is_fallback': False}
    dead = {**live, 'private_bytes': 0}
    calls = []
    def fake(pid):
        calls.append(pid);return dict(live if len(calls) == 1 else dead)
    monkeypatch.setattr(phase4_resources, 'process_memory', fake)
    result = measure_rss.run_measured([timeout, '--foreground', '5', sys.executable, '-c', 'import time;time.sleep(.3)'],
                                      memory=True, worker=True)
    assert len(calls) >= 2
    assert result['last_memory_sample']['last']['private_bytes'] == 290 << 20
    assert result['peak_private_bytes'] == 295 << 20 and result['last_memory_sample']['samples'] == 1

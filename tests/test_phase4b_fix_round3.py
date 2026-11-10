"""Rev 13: launch clock, termination evidence, recovery and canonical locks."""
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import phase4b_campaign as driver,phase4_approvals as approvals,phase4_policy as policy
from scripts import optimize_orderflow as opt,phase4_campaign as campaign
from tests.test_phase4b_fix_round1 import runtime_fixture


def launched(tmp_path,monkeypatch,stage='S2t'):
    layout,manifest,digest=runtime_fixture(tmp_path)
    monkeypatch.setattr(driver,'_now',lambda:150.)
    manifest['_launch']={'utc':'2026-11-07T00:00:00+00:00','monotonic':100.,'attempt':'synthetic'}
    manifest['max_stage_seconds'][stage]=600.
    return layout,manifest,digest


@pytest.mark.parametrize('code,elapsed,kind',[
    (-signal.SIGKILL,550.,'external_or_unknown_termination'),
    (-signal.SIGTERM,550.,'external_or_unknown_termination'),
    (124,599.,'external_or_unknown_termination'),(124,600.,'timeout_confirmed'),
    (137,630.,'timeout_confirmed'),(137,600.,'external_or_unknown_termination'),(-signal.SIGKILL,610.,'external_or_unknown_termination'),
    (1,610.,'external_or_unknown_termination')])
def test_r12_02_observed_exit_classification(tmp_path,monkeypatch,code,elapsed,kind):
    layout,manifest,digest=launched(tmp_path,monkeypatch)
    monkeypatch.setattr(driver,'_now',lambda:100.+elapsed)
    record=driver.record_termination(layout,manifest,digest,'S2t',returncode=code,memory={'rss_bytes':123,'private_bytes':99})
    assert record['kind']==kind and record['elapsed_seconds']==elapsed
    assert record['L']==manifest['_launch'] and record['last_memory_sample']=={'rss_bytes':123,'private_bytes':99}
    assert record['observed_exit_code']==code
    if kind=='external_or_unknown_termination':
        assert record['cause']=='unknown' and 'oom' not in json.dumps(record).lower()


@pytest.mark.parametrize('reason,duration,prior',[
    ('max_trial_seconds',600.,0.),('max_study_seconds',30.,570.)])
def test_r13_01_confirm_each_own_clock_and_boundaries(tmp_path,monkeypatch,reason,duration,prior):
    layout,manifest,digest=launched(tmp_path,monkeypatch)
    stop={'reason':reason,'measured_seconds':prior+duration,'limit_seconds':600.,'live':True,
        'charge_basis':'observed','boundaries':{'start_monotonic':150.-duration,'end_monotonic':150.,
            'prior_charge_seconds':prior,'current_seconds':duration}}
    record=driver.record_termination(layout,manifest,digest,'S2t',stop=stop)
    assert record['kind']=='timeout_confirmed' and record['elapsed_seconds']==50.
    assert record['duration_seconds']==600. and record['boundaries']==stop['boundaries']
    assert record['charge_basis']=='observed'


def test_r13_01_estimated_or_under_limit_evidence_is_never_timeout(tmp_path,monkeypatch):
    layout,manifest,digest=launched(tmp_path,monkeypatch)
    record=driver.record_termination(layout,manifest,digest,'S2t',stop={'reason':'max_trial_seconds',
        'measured_seconds':630.,'limit_seconds':600.,'reconciled':True,'charge_basis':'upper_bound'})
    assert record['kind']=='external_or_unknown_termination' and record['charge_basis']=='upper_bound'


def test_r12_02_audit_recovery_after_failed_append(tmp_path,monkeypatch):
    layout,manifest,digest=launched(tmp_path,monkeypatch)
    append=campaign.append_log
    monkeypatch.setattr(campaign,'append_log',lambda *a:(_ for _ in ()).throw(OSError('append failed')))
    with pytest.raises(OSError):driver.record_termination(layout,manifest,digest,'S2t',returncode=-9)
    monkeypatch.setattr(campaign,'append_log',append)
    record=driver.recover_termination_audit(layout,digest,'S2t')
    assert record['kind']=='external_or_unknown_termination'
    assert driver.recover_termination_audit(layout,digest,'S2t')==record
    lines=(layout.output/'campaign-log.jsonl').read_text().splitlines()
    assert len(lines)==1 and json.loads(lines[0])['event']=='termination_audit'


def recovery_record(layout,digest,stage,termination,argv):
    from scripts.phase4b_runtime import runtime_fingerprint
    return {'kind':'recovery','phase':layout.phase,'campaign_sha256':digest,'stage':stage,
        'termination_sha256':hashlib.sha256(policy.canonical_bytes(termination)).hexdigest(),
        'runtime_fingerprint':runtime_fingerprint(),'command_argv':argv}


@pytest.mark.parametrize('change',['manifest','stage','termination','fingerprint','argv','mode','symlink'])
def test_r12_02_recovery_binding_refusals(tmp_path,monkeypatch,change):
    layout,manifest,digest=launched(tmp_path,monkeypatch)
    stop=driver.record_termination(layout,manifest,digest,'S2t',returncode=-9)
    argv=['python','driver','run','--stage','S2t']
    record=recovery_record(layout,digest,'S2t',stop,argv)
    if change=='manifest':record['campaign_sha256']='b'*64
    if change=='stage':record['stage']='S2v'
    if change=='termination':record['termination_sha256']='b'*64
    if change=='fingerprint':record['runtime_fingerprint']={}
    if change=='argv':record['command_argv']=['different']
    path=approvals.install_recovery_approval(layout,record)
    if change=='mode':path.chmod(0o644)
    if change=='symlink':
        target=tmp_path/'target';path.rename(target);path.symlink_to(target)
    with pytest.raises(ValueError,match='recovery|trusted'):
        approvals.require_recovery_approval(layout,path.stem,digest,'S2t',stop,argv,consume=True)


def test_r12_02_recovery_exclusive_single_use_and_no_timeout_recovery(tmp_path,monkeypatch):
    layout,manifest,digest=launched(tmp_path,monkeypatch)
    stop=driver.record_termination(layout,manifest,digest,'S2t',returncode=-9)
    argv=['driver'];record=recovery_record(layout,digest,'S2t',stop,argv)
    path=approvals.install_recovery_approval(layout,record)
    assert path.stat().st_mode&0o777==0o600
    assert path.stem==hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):approvals.install_recovery_approval(layout,record)
    approvals.require_recovery_approval(layout,path.stem,digest,'S2t',stop,argv,consume=True)
    with pytest.raises(ValueError,match='consumed|used'):
        approvals.require_recovery_approval(layout,path.stem,digest,'S2t',stop,argv,consume=True)
    for kind in ('timeout_confirmed','study_allowance_exhausted','integrity_stop','strict_memory_stop'):
        with pytest.raises(ValueError,match='recovery|consumed'):
            approvals.require_recovery_approval(layout,path.stem,digest,'S2t',{**stop,'kind':kind},argv)


def test_r12_02_unknown_admission_refused_without_approval(tmp_path,monkeypatch):
    layout,manifest,digest=launched(tmp_path,monkeypatch)
    driver.record_termination(layout,manifest,digest,'S2t',returncode=-9)
    with pytest.raises(ValueError,match='recovery'):
        driver.termination_admission(layout,manifest,digest,'S2t',None,['driver'])


def synthetic_study(tmp_path):
    import optuna
    study=optuna.create_study(storage='sqlite:///'+str(tmp_path/'synthetic.db'),study_name='same-study')
    study.add_trial(optuna.trial.create_trial(value=1.,user_attrs={'preserve':'exact'}))
    active=study.ask()
    active.set_user_attr('phase4_timing',{'started_utc':'2026-11-07T00:00:00+00:00',
        'heartbeat_utc':'2026-11-07T00:09:00+00:00','wall_seconds':540.,'runtime_seconds':540.,'stale_seconds':90.})
    study.set_user_attr('phase4_runtime_seconds',540.)
    return study


def test_r13_02_upper_bound_550_540_600_is_accounting_only(tmp_path):
    from scripts import phase4b_accounting as accounting
    study=synthetic_study(tmp_path);before=study.trials[0]
    accounting.reconcile(study,SimpleNamespace(phase4=True,max_trial_seconds=600.,max_study_seconds=2000.))
    assert study.trials[0]==before and study.trials[1].state.name=='FAIL'
    assert opt.phase4_runtime_baseline(study)==630.
    assert study.user_attrs.get('phase4_runtime_stop') is None
    assert study.user_attrs['phase4_reconciliation_audit'][0]['charge_basis']=='upper_bound'
    accounting.reconcile(study,SimpleNamespace(phase4=True,max_trial_seconds=600.,max_study_seconds=2000.))
    assert opt.phase4_runtime_baseline(study)==630.


def test_r13_02_allowance_exhausted_consumes_without_timeout(tmp_path,monkeypatch):
    from scripts import phase4b_accounting as accounting
    layout,manifest,digest=launched(tmp_path,monkeypatch)
    study=synthetic_study(tmp_path)
    args=SimpleNamespace(phase4=True,max_trial_seconds=600.,max_study_seconds=600.,
        phase4_termination_audit=lambda stop:driver.record_termination(layout,manifest,digest,'S2t',stop=stop))
    accounting.reconcile(study,args)
    with pytest.raises(ValueError,match='allowance'):accounting.admit(study,args)
    record=driver.recover_termination_audit(layout,digest,'S2t')
    assert record['kind']=='study_allowance_exhausted' and record['charge_basis']=='upper_bound'
    assert study.user_attrs.get('phase4_runtime_stop') is None
    with pytest.raises(ValueError,match='consumed'):
        driver.termination_admission(layout,manifest,digest,'S2t','approval',['driver'])


def test_r13_01_live_study_confirmation_uses_prior_plus_current(tmp_path,monkeypatch):
    from scripts import phase4b_accounting as accounting
    layout,manifest,digest=launched(tmp_path,monkeypatch)
    monkeypatch.setattr(opt.time,'perf_counter',lambda:30.)
    study=synthetic_study(tmp_path);seen=[]
    args=SimpleNamespace(max_trial_seconds=600.,max_study_seconds=600.,phase4_termination_audit=seen.append,
        phase4_launch=manifest['_launch'])
    assert accounting.runtime_stop(study,args,600.,prior=570.,begun=0.)
    stop=seen[0];assert stop['boundaries']['prior_charge_seconds']==570.
    assert stop['boundaries']['current_seconds']==30. and stop['charge_basis']=='observed'
    record=driver.record_termination(layout,manifest,digest,'S2t',stop=stop)
    assert record['kind']=='timeout_confirmed' and record['elapsed_seconds']==50.


def test_r13_02_pending_accounting_does_not_promote_estimate(tmp_path):
    from scripts import phase4b_accounting as accounting
    study=synthetic_study(tmp_path)
    study.set_user_attr('phase4_accounting_pending',{'trial':1,'wall_seconds':540.,
        'heartbeat_utc':'2026-11-07T00:09:00+00:00','started_utc':'2026-11-07T00:00:00+00:00',
        'runtime_seconds_before':0.,'stale_bound':90.,'limits':{'max_trial_seconds':600.,'max_study_seconds':600.}})
    accounting.reconcile(study,SimpleNamespace(phase4=True,max_trial_seconds=600.,max_study_seconds=600.))
    assert study.user_attrs.get('phase4_runtime_stop') is None
    assert opt.phase4_runtime_baseline(study)==630. and study.trials[1].state.name=='FAIL'


def test_r12_01_stage_watchdog_includes_slow_preflight(tmp_path):
    script=r'''
import time,sys,json
from pathlib import Path
from scripts import phase4b_campaign as d,phase4_policy as p,phase4_resources as r
layout=p.PhaseLayout('phase4b',root=Path(sys.argv[1]),approvals_base=Path(sys.argv[1])/'approvals')
m={'_launch':{'utc':'2026-11-07T00:00:00+00:00','monotonic':d._now()-.08,'attempt':'slow'},
 'max_stage_seconds':{'S2t':.15},'runtime_ceilings':{'inputs':{}}}
r.current_memory=lambda:{'rss_bytes':1,'private_bytes':1,'peak_rss_bytes':1,'peak_private_bytes':1,'private_is_fallback':False}
with d.stage_watchdog(layout,m,'a'*64,'S2t'):time.sleep(2)
'''
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path)],capture_output=True,text=True,timeout=10)
    assert result.returncode==70,result.stderr
    layout=policy.PhaseLayout('phase4b',root=tmp_path,approvals_base=tmp_path/'approvals')
    record=driver.recover_termination_audit(layout,'a'*64,'S2t')
    assert record['kind']=='timeout_confirmed' and .15<=record['elapsed_seconds']<.5


def test_r13_01_trial_watchdog_uses_own_start_and_launch_context(tmp_path):
    script=r'''
import time,sys,json
from pathlib import Path
from types import SimpleNamespace
from scripts import phase4b_campaign as d,phase4_policy as p,phase4_resources as r,optimize_orderflow as o
layout=p.PhaseLayout('phase4b',root=Path(sys.argv[1]),approvals_base=Path(sys.argv[1])/'approvals')
m={'_launch':{'utc':'2026-11-07T00:00:00+00:00','monotonic':d._now()-.4,'attempt':'trial'},
 'max_stage_seconds':{'S2t':2.},'runtime_ceilings':{'inputs':{}}}
r.current_memory=lambda:{'rss_bytes':1,'private_bytes':1,'peak_rss_bytes':1,'peak_private_bytes':1,'private_is_fallback':False}
o.PHASE4_HEARTBEAT_SECONDS=.02
class A:
 number=3
 user_attrs={}
 def set_user_attr(self,k,v):self.user_attrs[k]=v
args=SimpleNamespace(phase='phase4b',max_trial_seconds=.06,max_study_seconds=2.,phase4_launch=m['_launch'],
 phase4_timeout_audit=lambda stop:d.record_termination(layout,m,'a'*64,'S2t',stop=stop))
with o.phase4_trial_timing(A(),A(),0.,time.perf_counter(),args):time.sleep(2)
'''
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path)],capture_output=True,text=True,timeout=10)
    assert result.returncode==70,result.stderr
    layout=policy.PhaseLayout('phase4b',root=tmp_path,approvals_base=tmp_path/'approvals')
    record=driver.recover_termination_audit(layout,'a'*64,'S2t')
    assert record['kind']=='timeout_confirmed' and .06<=record['duration_seconds']<.25
    assert record['elapsed_seconds']>=.4 and record['boundaries']['prior_charge_seconds']==0.


def origin_lock_fixture(tmp_path,monkeypatch):
    from tests.test_phase4b_fix_round2 import pinned_inputs
    layout,snapshots,comparators,origin=pinned_inputs(tmp_path,monkeypatch)
    path=Path(origin['root'])/'output/phase4/campaign-execution.lock';path.write_bytes(b'canonical lock bytes')
    return layout,origin,path


@pytest.mark.parametrize('state',['absent','symlink','directory','held'])
def test_r12_03_canonical_lock_refusals(tmp_path,monkeypatch,state):
    layout,origin,path=origin_lock_fixture(tmp_path,monkeypatch)
    if state=='absent':path.unlink()
    elif state=='symlink':
        target=tmp_path/'lock-target';path.rename(target);path.symlink_to(target)
    elif state=='directory':path.unlink();path.mkdir()
    proc=None
    if state=='held':
        proc=subprocess.Popen([sys.executable,'-c',
            'import fcntl,sys; f=open(sys.argv[1],"rb"); fcntl.flock(f,fcntl.LOCK_EX); print("held",flush=True); sys.stdin.read()',str(path)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        assert proc.stdout.readline().strip()=='held'
    try:
        with pytest.raises(ValueError,match='Phase 4.*lock|canonical.*lock'):
            with driver.canonical_execution_lock(origin):pytest.fail('invalid canonical lock admitted')
    finally:
        if proc:proc.communicate('',timeout=5)


def test_r12_03_lock_unchanged_and_held_through_postflight(tmp_path,monkeypatch):
    layout,origin,path=origin_lock_fixture(tmp_path,monkeypatch)
    before=(path.stat().st_mtime_ns,path.stat().st_size,path.read_bytes())
    with driver.stage_locks(layout,{'phase4_origin':origin}):
        probe=subprocess.run([sys.executable,'-c',
            'import fcntl,sys; f=open(sys.argv[1],"rb"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)',str(path)],capture_output=True)
        assert probe.returncode!=0
        assert (layout.checkpoints/'execution.lock').exists() and (layout.checkpoints/'driver.lock').exists()
    assert (path.stat().st_mtime_ns,path.stat().st_size,path.read_bytes())==before


def postflight_fixture(tmp_path,monkeypatch):
    from tests.test_phase4b_fix_round2 import calibrated
    layout,_,register=calibrated(tmp_path,monkeypatch);manifest=register()
    return layout,manifest,hashlib.sha256(layout.manifest.read_bytes()).hexdigest()


def test_r12_01_deferred_stage_is_not_complete_until_postflight(tmp_path,monkeypatch):
    layout,manifest,digest=postflight_fixture(tmp_path,monkeypatch)
    pending=driver.run_stage(layout,'S2t','s2-approval',operation=lambda *a:{'synthetic':True},finalize=False)
    assert pending['status']=='pending_postflight'
    assert not (layout.checkpoints/digest/'S2t.json').exists()
    record=driver.postflight_stage(layout,'S2t',synthetic=True)
    assert record['status']=='complete' and (layout.checkpoints/digest/'S2t.json').exists()


def test_r12_01_postflight_integrity_mismatch_is_durable_stop(tmp_path,monkeypatch):
    layout,manifest,digest=postflight_fixture(tmp_path,monkeypatch)
    driver.run_stage(layout,'S2t','s2-approval',operation=lambda *a:{'synthetic':True},finalize=False)
    tampered=Path(manifest['snapshots'][0]['path']);tampered.write_bytes(tampered.read_bytes()+b'x')
    with pytest.raises(ValueError,match='integrity|protected'):driver.postflight_stage(layout,'S2t',synthetic=True)
    assert not (layout.checkpoints/digest/'S2t.json').exists()
    assert driver.recover_termination_audit(layout,digest,'S2t')['kind']=='integrity_stop'
    with pytest.raises(ValueError,match='consumed'):
        driver.termination_admission(layout,manifest,digest,'S2t',None,['driver'])


def test_r12_01_main_runs_postflight_as_separate_untimed_child(tmp_path,monkeypatch):
    from scripts import measure_rss,phase4_resources
    layout,manifest,digest=postflight_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(driver,'PhaseLayout',lambda *a:layout)
    monkeypatch.setattr(driver.shutil,'which',lambda *a:'/synthetic/timeout')
    layout.approval_root.mkdir(parents=True,exist_ok=True);(layout.approval_root/('a'*64+'.json')).write_text('{}')
    monkeypatch.setattr(approvals,'verify_phase4b_approval',lambda *a:None)
    monkeypatch.setattr(phase4_resources,'enforce_memory',lambda *a:None)
    @contextmanager
    def locks(layout,manifest):yield {}
    monkeypatch.setattr(driver,'stage_locks',locks)
    commands=[]
    def measured(command,**kw):
        commands.append((command,kw.get('env',{}).get(driver.LAUNCH_ENV)))
        return {'returncode':0,'wall_seconds':1.}
    monkeypatch.setattr(measure_rss,'run_measured',measured)
    driver.main(['run','--phase','phase4b','--stage','S2t','--owner-approved','parent'])
    (timed,launch),(post,post_launch)=commands
    assert timed[0]=='/synthetic/timeout' and 'worker' in timed
    assert post[0]!='/synthetic/timeout' and 'postflight' in post and launch==post_launch
    assert json.loads(launch)['attempt']


def test_phase4_reconciliation_functions_remain_byte_identical():
    import ast
    old=subprocess.check_output(['git','show','95bde47:scripts/optimize_orderflow.py'],text=True)
    current=Path('scripts/optimize_orderflow.py').read_text()
    def functions(source):
        return {n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    for name in ('reconcile_phase4_trials','reconciled_runtime_stop'):
        assert functions(old)[name]==functions(current)[name]


def test_rev13_plan_binding_preserves_prior_revisions():
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert approvals.PHASE4B_PLAN_SHA256=='75b22452c0b168f43fd91704a7aee23a4cb1c84c637f4a8f775b87be13857b7f'
    assert sha(approvals.PHASE4B_PLAN_PATH)==approvals.PHASE4B_PLAN_SHA256
    assert sha('docs/plans/phase4b-full-strategy-plan-rev12.md')=='dbb407fb611720e42d70ac6c051d82077350247ef83fb433f13e9d97d8849099'
    assert sha('docs/plans/phase4b-full-strategy-plan-rev11.md')=='2c9cf6a80c912500552241a3cd4d223dbc5cc8e3847e0ef86c39862f6e457af7'
    assert sha('docs/plans/phase4b-full-strategy-plan.md')=='89f64efe62090c271045d33b4ef9cbabb257f3ca5d782aa9032144d322f029c7'

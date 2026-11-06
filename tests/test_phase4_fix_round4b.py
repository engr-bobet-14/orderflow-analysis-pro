"""Inspection-4 regressions; records and studies are synthetic and temporary."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import optuna
import pytest

from scripts import optimize_orderflow as opt,phase4_campaign as campaign,phase4_approvals as approvals
from scripts import phase4_policy as policy,phase4_studies as p4
from test_phase4_fix_round4 import bound_gate,write_json
from test_optimize_orderflow import args_for


def test_4b_f1_sigkill_clock_gap_not_runtime(tmp_path):
    code=r'''
import sys,time,json
from datetime import datetime,timedelta
from pathlib import Path
from types import SimpleNamespace
import optuna
from scripts import optimize_orderflow as o
root=Path(sys.argv[1]);stage=sys.argv[2]
s=optuna.create_study(storage="sqlite:///"+str(root/"unit.db"),study_name="unit",load_if_exists=True)
a=SimpleNamespace(phase4=True,n_trials=2,seed=42,n_jobs=1,max_study_seconds=10.)
o.PHASE4_HEARTBEAT_SECONDS=.02
o.PHASE4_HEARTBEAT_STALE_SECONDS=.06
if stage=="resume":
    # Advance the reconciliation/storage clock by two days without sleeping two days.
    import optuna.storages._rdb.storage as storage
    class Later(datetime):
        @classmethod
        def now(cls,tz=None):return datetime.now(tz)+timedelta(days=2)
    storage.datetime=Later;o.datetime=Later
    o.reconcile_phase4_trials(s,a)
def objective(trial,*args):
    if stage=="kill" and len(o.complete_trials(s))==1:
        time.sleep(.1);(root/"ready").touch();time.sleep(60)
    time.sleep(.015)
    return trial.suggest_float("x",0.,1.)
o.objective=objective;o.optimize_loop(s,None,None,a)
(root/(stage+".json")).write_text(json.dumps({"elapsed":s.user_attrs["phase4_runtime_seconds"],
    "complete":len(o.complete_trials(s)),"attrs":s.user_attrs,"trials":[t.user_attrs for t in s.trials]}))
'''
    env={**os.environ,'PYTHONPATH':str(Path.cwd())}
    proc=subprocess.Popen([sys.executable,'-c',code,str(tmp_path),'kill'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+25
        while not (tmp_path/'ready').exists():
            assert proc.poll() is None,proc.communicate()[1].decode()
            assert time.monotonic()<deadline
            time.sleep(.02)
        proc.send_signal(signal.SIGKILL);proc.communicate(timeout=10)
        result=subprocess.run([sys.executable,'-c',code,str(tmp_path),'resume'],env=env,capture_output=True,timeout=30)
        assert result.returncode==0,result.stderr.decode()
        data=json.loads((tmp_path/'resume.json').read_text())
        assert data['complete']==2 and data['elapsed']<10.
        failed=next(t for t in data['trials'] if 'phase4_reconciled' in t)
        audit=failed['phase4_reconciled']
        assert .02<audit['accounted_seconds']<1.
        assert failed['phase4_heartbeat_utc']
        assert 'phase4_runtime_stop' not in data['attrs']
    finally:
        if proc.poll() is None:proc.kill();proc.communicate(timeout=10)


def test_4b_f1_heartbeat_and_normal_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(opt,'PHASE4_HEARTBEAT_SECONDS',.01,raising=False)
    a=args_for(tmp_path,phase4=True,n_jobs=1,n_trials=1)
    study=optuna.create_study();seen=[]
    def objective(trial,*args):
        first=study.trials[-1].user_attrs['phase4_heartbeat_utc']
        time.sleep(.045)
        seen.append(study.trials[-1].user_attrs['phase4_heartbeat_utc']!=first)
        return 1.
    monkeypatch.setattr(opt,'objective',objective)
    opt.optimize_loop(study,None,None,a);first=study.user_attrs['phase4_runtime_seconds']
    time.sleep(.08);a.n_trials=2
    opt.optimize_loop(study,None,None,a);second=study.user_attrs['phase4_runtime_seconds']
    assert all(seen) and first+.04<second<first+.08


def test_4b_f1_legacy_reconciled_duration_excluded():
    start=datetime(2026,11,1)
    interrupted=SimpleNamespace(datetime_start=start,datetime_complete=start+timedelta(days=2),
        user_attrs={'phase4_reconciled':{'reason':'orphan after interruption'}})
    complete=SimpleNamespace(datetime_start=start,datetime_complete=start+timedelta(seconds=1),user_attrs={})
    study=SimpleNamespace(user_attrs={},trials=[interrupted,complete])
    assert opt.phase4_runtime_baseline(study)==1.


def test_4b_f1_checkpoint_survives_missing_study_write(tmp_path):
    study=optuna.create_study()
    trial=study.ask();trial.set_user_attr('phase4_timing',{'runtime_seconds':2.});study.tell(trial,1.)
    study.set_user_attr('phase4_runtime_seconds',1.)
    a=args_for(tmp_path,phase4=True,n_jobs=1,n_trials=1)
    opt.optimize_loop(study,None,None,a)
    assert study.user_attrs['phase4_runtime_seconds']==2.


@pytest.mark.parametrize('selector_damage',['missing','wrong_identity','wrong_stop','tampered','symlink','writable','none'])
def test_4b_f1_runtime_stop_clear_requires_host_record(tmp_path,monkeypatch,selector_damage):
    import pwd
    monkeypatch.setattr(pwd,'getpwuid',lambda uid:SimpleNamespace(pw_dir=str(tmp_path/'home')))
    for module in (campaign,policy,opt):monkeypatch.setattr(module,'ROOT',tmp_path)
    path=tmp_path/'output/optimize/phase4/unit.db';path.parent.mkdir(parents=True)
    identity={'phase':'phase4','unit':'identity'};stop={'reason':'max_trial_seconds','measured_seconds':1.,'limit_seconds':.01}
    study=optuna.create_study(storage='sqlite:///'+str(path),study_name='unit')
    trial=study.ask();trial.set_user_attr('phase4_wall_seconds',1.);study.tell(trial,1.)
    study.set_user_attr('identity',identity);study.set_user_attr('phase4_runtime_stop',stop)
    study.set_user_attr('phase4_runtime_seconds',3.)
    record={'plan_sha256':approvals.PLAN_SHA256,'study_identity_sha256':opt.fingerprint(identity),
        'stop_sha256':opt.fingerprint(stop),'decision':'clear_and_restart_budget','created_utc':'2026-11-06T00:00:00Z'}
    if selector_damage=='wrong_identity':record['study_identity_sha256']='0'*64
    if selector_damage=='wrong_stop':record['stop_sha256']='0'*64
    raw=policy.canonical_bytes(record);selector=hashlib.sha256(raw).hexdigest()
    target=approvals.approval_root()/('runtime-decision-'+selector+'.json');target.parent.mkdir(parents=True)
    if selector_damage!='missing':target.write_bytes(raw)
    if selector_damage=='tampered':target.write_bytes(raw+b' ')
    if selector_damage=='symlink':target.unlink();target.symlink_to(path)
    if selector_damage=='writable':target.chmod(0o666)
    call=lambda:campaign.clear_runtime_stop(path,'unit',selector)
    if selector_damage!='none':
        with pytest.raises(ValueError):call()
        assert study.user_attrs['phase4_runtime_stop']==stop
    else:
        call()
        assert study.user_attrs['phase4_runtime_stop'] is None
        audit=study.user_attrs['phase4_runtime_decisions']
        assert len(audit)==1 and audit[0]['decision_sha256']==selector and audit[0]['stop']==stop
        assert study.user_attrs['phase4_runtime_seconds']==3.
        monkeypatch.setattr(opt,'objective',lambda *a:1.)
        a=args_for(tmp_path,phase4=True,n_jobs=1,n_trials=2,max_trial_seconds=.1,max_study_seconds=.2)
        opt.optimize_loop(study,None,None,a)
        assert len(opt.complete_trials(study))==2 and study.user_attrs['phase4_runtime_stop'] is None
        a.n_trials=3
        monkeypatch.setattr(opt,'objective',lambda *a:time.sleep(.12) or 1.)
        opt.optimize_loop(study,None,None,a)
        assert study.user_attrs['phase4_runtime_stop']['reason']=='max_trial_seconds'
        with pytest.raises(ValueError):call()  # The old decision cannot clear a new stop.
        assert len(study.user_attrs['phase4_runtime_decisions'])==1


def registered(tmp_path,monkeypatch,trial=(),total=()):
    for module in (campaign,policy):monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(campaign,'source_state',lambda:{'base':'unit','sha256':'source','dirty':True})
    snapshots=[]
    for i in range(3):
        p=tmp_path/f'legacy-{i}';p.write_bytes(bytes([i]));snapshots.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    snapshot_file=tmp_path/'snapshots.json';write_json(snapshot_file,snapshots)
    manifest=tmp_path/'output/phase4/campaign.json'
    campaign.main(['register','--manifest',str(manifest),'--snapshots',str(snapshot_file),
        '--driver',str(tmp_path/'output/phase4/run_scripts/driver.py'),
        *[v for pair in trial for v in ('--max-trial-seconds',pair)],
        *[v for pair in total for v in ('--max-study-seconds',pair)]])
    return manifest


def test_4b_f2_register_limits_through_to_objective(tmp_path,monkeypatch):
    key='BTCUSDT:fixed_tick_imbalance:off'
    path=registered(tmp_path,monkeypatch,trial=[key+'=0.01'],total=[key+'=10'])
    manifest,_=p4.read_campaign(path);assert manifest['arms'][key]['max_trial_seconds']==.01
    result=campaign.run_driver(path,stage='B');command=next(c for c in result['commands'] if c[c.index('--bar-type')+1]=='fixed_tick_imbalance' and c[c.index('--trade-signals')+1]=='off')
    assert command[command.index('--max-trial-seconds')+1]=='0.01'
    assert command[command.index('--max-study-seconds')+1]=='10.0'
    a=opt.build_parser().parse_args(command[2:]);a.symbol,a.bar_type='BTCUSDT','fixed_tick_imbalance'
    p4.campaign_arm(a);assert a.max_trial_seconds==.01 and a.max_study_seconds==10.
    study=optuna.create_study()
    monkeypatch.setattr(opt,'objective',lambda *a:time.sleep(.025) or 1.)
    opt.optimize_loop(study,None,None,a)
    assert len(study.trials)==1 and study.user_attrs['phase4_runtime_stop']['reason']=='max_trial_seconds'


@pytest.mark.parametrize('value',['0','-1','nan','inf','bad'])
def test_4b_f2_limits_validate_before_write(tmp_path,monkeypatch,value):
    with pytest.raises(SystemExit):registered(tmp_path,monkeypatch,trial=['BTCUSDT:fixed_tick_imbalance:off='+value])
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('key',['unknown','BTCUSDT:fixed_tick_imbalance:framing_trades'])
def test_4b_f2_study_arms_only(tmp_path,monkeypatch,key):
    with pytest.raises(SystemExit):registered(tmp_path,monkeypatch,total=[key+'=1'])
    assert not (tmp_path/'output').exists()


def waiver(b,tmp_path,gate_path,*,legacy=False):
    artifact=tmp_path/'replacement.json';digest=write_json(artifact,{'seconds':1})
    record={'gate_name' if legacy else 'gate_path':gate_path,'benchmark_sha256':b.gates['benchmark_sha256'],
        'plan_sha256':approvals.PLAN_SHA256,'campaign_sha256':b.record['campaign_sha256'],
        'created_utc':'2026-11-06T00:00:00Z','replacement_gate':{'name':'replacement',
            'threshold_spec':{'metric':['seconds'],'operator':'<=','value':2},
            'measured_artifact_path':str(artifact),'measured_artifact_sha256':digest}}
    raw=policy.canonical_bytes(record);sha=hashlib.sha256(raw).hexdigest()
    (approvals.approval_root()/('waiver-'+sha+'.json')).write_bytes(raw)
    return sha


def test_4b_f3_one_waiver_per_full_gate_path(bound_gate,tmp_path):
    b=bound_gate;b.values['benchmark']['extrapolation']['resource_gates']['build_under_4h']=False
    b.values['benchmark']['extrapolation']['conservative_upper_bound']={'resource_gates':{'build_under_4h':False}};b.seal()
    b.gates['waivers']=[waiver(b,tmp_path,'extrapolation.build_under_4h')];b.seal()
    with pytest.raises(ValueError,match='extrapolation.conservative_upper_bound.build_under_4h'):b.admit()
    b.gates['waivers'].append(waiver(b,tmp_path,'extrapolation.conservative_upper_bound.build_under_4h'));b.seal()
    assert b.admit()['sha256']==b.selector


def test_4b_f3_legacy_name_only_refused(bound_gate,tmp_path):
    b=bound_gate;b.values['benchmark']['extrapolation']['resource_gates']['arms_under_3x']=False;b.seal()
    b.gates['waivers']=[waiver(b,tmp_path,'arms_under_3x',legacy=True)];b.seal()
    with pytest.raises(ValueError,match='waiver'):b.admit()

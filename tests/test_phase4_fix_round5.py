"""Round-5 failure injection; no real studies, approvals, cache builds or experiments."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

import optuna
import pytest

from scripts import optimize_orderflow as opt,phase4_resources as resources,phase4_campaign as campaign
from scripts import phase4_studies as p4,phase4_policy as policy
from test_optimize_orderflow import args_for

GiB=1024**3


@pytest.mark.parametrize('sample',[
    {'rss_bytes':int(4.5*GiB),'private_bytes':int(.3*GiB)},
    {'rss_bytes':3*GiB,'private_bytes':int(4.2*GiB)},
    {'rss_bytes':3*GiB,'private_bytes':None,'private_is_fallback':True},
])
def test_r5_w4_strict_memory_even_with_mapping(sample):
    with pytest.raises(ValueError):resources.enforce_memory(sample,3*GiB)


SCRIPT=r'''
import sys,time,json,threading
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import optuna
from scripts import optimize_orderflow as o,phase4_resources as r
root=Path(sys.argv[1]);mode=sys.argv[2];root.mkdir(parents=True,exist_ok=True)
db=root/'unit.db'
s=optuna.create_study(storage='sqlite:///'+str(db),study_name='unit',load_if_exists=True)
o.PHASE4_HEARTBEAT_SECONDS=.06;o.PHASE4_HEARTBEAT_STALE_SECONDS=.18
state={'calls':0,'finished':False,'failed':False};original=optuna.trial.Trial.set_user_attr

def injected(trial,key,value):
    if key=='phase4_timing':
        state['calls']+=1
        if mode=='blocked' and state['calls']>=2:time.sleep(2.)
        fail=(mode in ('persistent','sigkill') and state['calls']>=2 or mode=='transient' and state['calls']==2
              or mode=='final' and state['finished'])
        if fail:
            state['failed']=True
            raise OSError('injected checkpoint outage')
    return original(trial,key,value)
optuna.trial.Trial.set_user_attr=injected
original_timing=o.phase4_trial_timing
@contextmanager
def timed(*a,**kw):
    state['start']=time.perf_counter()
    with original_timing(*a,**kw):yield
o.phase4_trial_timing=timed
if mode=='memory':
    def memory(*a,**kw):
        value=5*r.GiB if time.perf_counter()-state.get('start',time.perf_counter())>.03 else 100000000
        return {'rss_bytes':value,'private_bytes':100000000,'peak_private_bytes':100000000,
                'private_metric':'injected','private_is_fallback':False}
    r.process_memory=memory

def objective(t,*unused):
    t.suggest_float('x',0.,1.)
    until=time.perf_counter()+(.075 if mode=='final' else .25 if mode=='transient' else 2.)
    while time.perf_counter()<until:
        marker=root/'work.tmp';marker.write_text(json.dumps({'seconds':time.perf_counter()-state['start']}))
        marker.replace(root/'work.json')
        if mode=='sigkill' and state['failed']:(root/'ready').touch()
        time.sleep(.003)
    state['finished']=True
    return 1.
o.objective=objective
a=SimpleNamespace(phase4=True,n_trials=2 if mode=='final' else 1,seed=42,n_jobs=1,
                  max_trial_seconds=.12 if mode=='live' else None)
try:o.optimize_loop(s,None,None,a)
except Exception as exc:
    print(type(exc).__name__,str(exc),file=sys.stderr,flush=True);sys.exit(3)
(root/'result.json').write_text(json.dumps({'attrs':s.user_attrs,'trials':[(t.state.name,t.user_attrs) for t in s.trials]}))
'''


def launch(tmp_path,mode):
    return subprocess.Popen([sys.executable,'-c',SCRIPT,str(tmp_path),mode],env={**os.environ,'PYTHONPATH':str(Path.cwd())},
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)


def finish(proc,timeout=8):
    try:return proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill();out,err=proc.communicate();pytest.fail('watchdog/ceiling did not terminate the process: '+err)


def load_unit(tmp_path):return optuna.load_study(storage='sqlite:///'+str(tmp_path/'unit.db'),study_name='unit')


def reconcile_and_assert_bound(tmp_path):
    study=load_unit(tmp_path);opt.reconcile_phase4_trials(study,SimpleNamespace(phase4=True))
    trial=study.trials[-1];record=trial.user_attrs['phase4_reconciled']
    work=json.loads((tmp_path/'work.json').read_text())['seconds'];bound=.18
    assert record['accounting']=='conservative_upper_bound'
    assert work<=record['accounted_seconds']<=work+bound
    first=study.user_attrs['phase4_runtime_seconds'];audit=study.user_attrs['phase4_reconciliation_audit']
    opt.reconcile_phase4_trials(study,SimpleNamespace(phase4=True))
    assert study.user_attrs['phase4_runtime_seconds']==first
    assert study.user_attrs['phase4_reconciliation_audit']==audit
    assert first>=record['accounted_seconds']
    return study


def test_r5_l1_transient_failure_recovers_and_keeps_trial(tmp_path):
    proc=launch(tmp_path,'transient');out,err=finish(proc)
    assert proc.returncode==0,(out,err)
    study=load_unit(tmp_path);trial=study.trials[0]
    assert trial.state.name=='COMPLETE'
    health=trial.user_attrs['phase4_heartbeat_health']
    assert health['total_failures']==1 and health['consecutive_failures']==0
    assert 'injected checkpoint outage' in health['last_error']
    assert 0.<health['longest_gap_seconds']<.18
    assert trial.user_attrs['phase4_memory']['peak_rss_bytes']>0
    assert study.user_attrs['phase4_runtime_policy']['stale_seconds']==.18


@pytest.mark.parametrize('mode',['persistent','blocked'])
def test_r5_l1_persistent_failure_watchdog_and_conservative_resume(tmp_path,mode):
    proc=launch(tmp_path,mode);out,err=finish(proc)
    assert proc.returncode!=0 and 'stale' in err.lower(),(out,err)
    study=load_unit(tmp_path);assert study.trials[-1].state.name=='RUNNING'
    reconcile_and_assert_bound(tmp_path)


def test_r5_l1_final_checkpoint_failure_stops_loop(tmp_path):
    proc=launch(tmp_path,'final');out,err=finish(proc)
    assert proc.returncode!=0,(out,err)
    study=load_unit(tmp_path);assert len(study.trials)==1 and study.trials[0].state.name=='FAIL'
    assert 'phase4_wall_seconds' not in study.trials[0].user_attrs
    assert study.trials[0].user_attrs['phase4_heartbeat_health']['total_failures']>=3
    reconcile_and_assert_bound(tmp_path)


def test_r5_l1_sigkill_after_failure_not_undercounted(tmp_path):
    proc=launch(tmp_path,'sigkill')
    try:
        deadline=time.monotonic()+8
        while not (tmp_path/'ready').exists():
            assert proc.poll() is None,proc.communicate()
            assert time.monotonic()<deadline
            time.sleep(.003)
        proc.send_signal(signal.SIGKILL);finish(proc)
        reconcile_and_assert_bound(tmp_path)
    finally:
        if proc.poll() is None:proc.kill();finish(proc)


@pytest.mark.parametrize('mode,reason',[('live','max_trial_seconds'),('memory','memory')])
def test_r5_live_stop_persisted_and_blocks_resume(tmp_path,mode,reason):
    proc=launch(tmp_path,mode);out,err=finish(proc)
    assert proc.returncode!=0,(out,err)
    study=load_unit(tmp_path);assert study.trials[0].state.name=='RUNNING'
    stop=study.user_attrs['phase4_runtime_stop'];assert stop['reason']==reason and stop['live'] is True
    if mode=='memory':assert study.trials[0].user_attrs['phase4_memory']['peak_rss_bytes']>4*GiB
    opt.reconcile_phase4_trials(study,SimpleNamespace(phase4=True))
    before=len(study.trials)
    opt.optimize_loop(study,None,None,SimpleNamespace(phase4=True,n_trials=2,seed=42,n_jobs=1))
    assert len(study.trials)==before


def test_r5_child_memory_refusal_before_cache_and_db(tmp_path,monkeypatch):
    from orderflow_system.config.settings import TradeSignalConfig
    def configured(a):a.phase4=True;a.trade_signal_config=TradeSignalConfig()
    monkeypatch.setattr(p4,'configure',configured)
    monkeypatch.setattr(opt,'require_source',lambda *a:None)
    monkeypatch.setattr(resources,'available_memory',lambda:(7*GiB,'injected'))
    monkeypatch.setattr(opt,'build_cache',lambda *a:pytest.fail('cache opened before child preflight'))
    monkeypatch.setattr(opt,'open_study',lambda *a:pytest.fail('DB opened before child preflight'))
    with pytest.raises(SystemExit):opt.main(['--phase4','--no-warm-start','--study-db',str(tmp_path/'unit.db')])
    assert not (tmp_path/'unit.db').exists()


def test_r5_driver_logs_memory_refusal_no_child(tmp_path,monkeypatch):
    manifest={'arms':{}};digest='unit'
    monkeypatch.setattr(campaign,'ROOT',tmp_path);monkeypatch.setattr(policy,'ROOT',tmp_path)
    monkeypatch.setattr(campaign,'read_campaign',lambda *a:(manifest,digest))
    monkeypatch.setattr(campaign,'commands',lambda *a:[])
    monkeypatch.setattr(p4,'require_gates',lambda *a,**kw:{'sha256':'owner','gates_sha256':'gates'})
    from scripts import phase4_approvals as approvals
    monkeypatch.setattr(approvals,'approval_for_command',lambda *a:'child')
    monkeypatch.setattr(campaign,'preflight',lambda *a:{})
    monkeypatch.setattr(campaign,'protected_state',lambda:{})
    monkeypatch.setattr(campaign,'source_state',lambda:{})
    monkeypatch.setattr(campaign,'verify_source',lambda *a:None)
    monkeypatch.setattr(resources,'available_memory',lambda:(7*GiB,'injected'))
    # Exercise the real memory planner with a bar-only command, no TFC metadata needed.
    original=campaign.command_memory_plan
    monkeypatch.setattr(campaign,'command_memory_plan',lambda *a,**kw:original(['--trade-signals','off'],**kw))
    from scripts import measure_rss
    monkeypatch.setattr(measure_rss,'run_measured',lambda *a,**kw:pytest.fail('child launched on refused preflight'))
    driver=tmp_path/'driver.py';driver.write_text('unit')
    with pytest.raises(ValueError,match='memory'):
        campaign.run_driver('unit',stage='A',owner_approved='owner',driver=driver,
            inspected_driver=hashlib.sha256(driver.read_bytes()).hexdigest(),dry_run=False)
    row=next(json.loads(line) for line in (tmp_path/'output/phase4/campaign-log.jsonl').read_text().splitlines()
             if json.loads(line).get('status')=='refused')
    assert row['status']=='refused' and row['memory_preflight']['available_bytes']==7*GiB
    assert row['measurement'] is None


def test_r5_preflight_append_and_policy_reporting(tmp_path,monkeypatch):
    a=args_for(tmp_path,phase4=True,approval_sha256='owner',gates_sha256='gates')
    monkeypatch.setattr(p4,'campaign_arm',lambda *a,**kw:{})
    study=opt.open_study(a,'unit',{'phase':'phase4'})
    assert study.user_attrs['phase4_memory_preflight'][-1]['available_bytes']>=8*GiB
    study=opt.open_study(a,'unit',{'phase':'phase4'})
    assert len(study.user_attrs['phase4_memory_preflight'])==2
    from scripts import phase4_report as report
    summary=report.runtime_diagnostics(study.user_attrs)
    assert summary['policy']['interval_seconds']==30. and summary['policy']['stale_seconds']==90.


def test_r5_legacy_has_no_heartbeat_or_reconciliation(tmp_path,monkeypatch):
    a=args_for(tmp_path,phase4=False,n_trials=2,n_jobs=1)
    study=optuna.create_study()
    monkeypatch.setattr(opt,'objective',lambda t,*a:t.suggest_float('x',0.,1.))
    opt.optimize_loop(study,None,None,a)
    assert not study.user_attrs and all(not t.user_attrs for t in study.trials)
    orphan=study.ask();opt.reconcile_phase4_trials(study,a)
    assert study.trials[-1].state.name=='RUNNING'


def test_r5_second_study_process_refused_by_execution_lock(tmp_path,monkeypatch):
    monkeypatch.setattr(opt,'ROOT',tmp_path);monkeypatch.setattr(policy,'ROOT',tmp_path)
    code='''from pathlib import Path
import sys
from scripts import optimize_orderflow as o,phase4_policy as p
o.ROOT=p.ROOT=Path(sys.argv[1])
with o.phase4_execution(True):raise AssertionError("second study admitted")
'''
    with opt.phase4_execution(True):
        proc=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True,timeout=10)
    assert proc.returncode!=0 and 'another Phase 4 campaign process' in proc.stderr


RECOVERY=r'''
import sys,time,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import optuna
from scripts import optimize_orderflow as o,phase4_resources as r,phase4_policy as policy,phase4_campaign as campaign,phase4_approvals as approvals
root=Path(sys.argv[1]);stage=sys.argv[2];folder=root/('whole' if stage=='whole' else 'resumed')
db=folder/'output/optimize/phase4/studies.db';db.parent.mkdir(parents=True,exist_ok=True)
s=optuna.create_study(storage='sqlite:///'+str(db),study_name='unit',direction='maximize',load_if_exists=True)
s.set_user_attr('identity',{'phase':'phase4','unit':'determinism'})
o.PHASE4_HEARTBEAT_SECONDS=.06;o.PHASE4_HEARTBEAT_STALE_SECONDS=.18
if stage in ('live','finish'):o.reconcile_phase4_trials(s,SimpleNamespace(phase4=True))
if stage=='finish':
    o.ROOT=policy.ROOT=campaign.ROOT=folder
    approvals.approval_root=lambda:folder/'approvals/phase4'
    record={'plan_sha256':approvals.PLAN_SHA256,'study_identity_sha256':o.fingerprint(s.user_attrs['identity']),
            'stop_sha256':o.fingerprint(s.user_attrs['phase4_runtime_stop']),
            'decision':'clear_and_restart_budget','created_utc':'2026-11-06T00:00:00Z'}
    raw=policy.canonical_bytes(record);sha=hashlib.sha256(raw).hexdigest()
    path=approvals.approval_root()/('runtime-decision-'+sha+'.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    campaign.clear_runtime_stop(db,'unit',sha)
original=optuna.trial.Trial.set_user_attr;counts={}
def checkpoint(t,key,value):
    if key=='phase4_timing' and stage=='watchdog' and len(o.complete_trials(s))==11:
        counts[t.number]=counts.get(t.number,0)+1
        if counts[t.number]>1:raise OSError('persistent deterministic outage')
    return original(t,key,value)
optuna.trial.Trial.set_user_attr=checkpoint

def objective(t,*unused):
    x=t.suggest_float('x',-3.,5.);y=t.suggest_int('y',1,8)
    if stage in ('watchdog','live') and len(o.complete_trials(s))==11:time.sleep(3.)
    return -(x-1.)**2+y
o.objective=objective
args=SimpleNamespace(phase4=True,n_trials=14,n_jobs=1,seed=42,max_trial_seconds=.25 if stage=='live' else None)
o.optimize_loop(s,None,None,args)
(folder/'complete.json').write_text(json.dumps([(t.params,t.value) for t in o.complete_trials(s)]))
'''


def test_r5_deterministic_watchdog_then_live_stop_then_host_clear(tmp_path):
    env={**os.environ,'PYTHONPATH':str(Path.cwd())}
    def run(stage):return subprocess.run([sys.executable,'-c',RECOVERY,str(tmp_path),stage],env=env,capture_output=True,text=True,timeout=15)
    first=run('watchdog');assert first.returncode!=0 and 'stale' in first.stderr.lower(),first.stderr
    db=tmp_path/'resumed/output/optimize/phase4/studies.db'
    def rows():
        with sqlite3.connect(db) as con:
            return {table:con.execute(f"SELECT * FROM {table} WHERE trial_id IN (SELECT trial_id FROM trials WHERE state='COMPLETE') ORDER BY trial_id").fetchall()
                for table in ('trials','trial_params','trial_values','trial_user_attributes')}
    before=rows();assert len(before['trials'])==11
    second=run('live');assert second.returncode!=0,second.stderr
    assert all(before[k]==rows()[k] for k in before)
    for stage in ('finish','whole'):
        got=run(stage);assert got.returncode==0,got.stderr
    resumed=json.loads((tmp_path/'resumed/complete.json').read_text())
    assert resumed==json.loads((tmp_path/'whole/complete.json').read_text()) and len(resumed)==14
    after=rows()
    assert all(all(row in after[key] for row in values) for key,values in before.items())

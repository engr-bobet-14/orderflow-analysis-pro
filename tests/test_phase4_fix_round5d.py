"""Final-review failure injection; no real studies, approvals or campaign execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import optuna
import pytest

from scripts import optimize_orderflow as opt,phase4_resources as resources
from scripts import phase4_campaign as campaign
from test_phase4_fix_round5 import finish
from test_phase4_fix_round5c import unit,host_clear
from test_phase4_fix_round5b import driver_unit,checkpoint

GiB=1024**3
SAFE={'rss_bytes':100000000,'private_bytes':100000000,'peak_private_bytes':100000000}
BREACH={**SAFE,'peak_rss_bytes':5*GiB}


TERMINAL=r'''
import sys,time,json
from pathlib import Path
from types import SimpleNamespace
import optuna
from scripts import optimize_orderflow as o
db=Path(sys.argv[1]);mode=sys.argv[2];ceiling=sys.argv[3]
s=optuna.create_study(storage='sqlite:///'+str(db),study_name='unit')
s.set_user_attr('identity',{'phase':'phase4','unit':'terminal'})
o.PHASE4_HEARTBEAT_SECONDS=.05;o.PHASE4_HEARTBEAT_STALE_SECONDS=.18
state={};backend=getattr(s._storage,'_backend',s._storage);original=backend.set_trial_state_values
def terminal(trial_id,state_value=None,values=None,**kw):
    value=kw.get('state',state_value)
    if value.is_finished():
        until=time.perf_counter()+(.6 if mode=='blocked' else .025)
        while time.perf_counter()<until:
            temp=db.parent/'elapsed.tmp';temp.write_text(str(time.perf_counter()-state['started']))
            temp.replace(db.parent/'elapsed.txt');time.sleep(.003)
        raise OSError('injected terminal-state write failure')
    return original(trial_id,state=value,values=values)
backend.set_trial_state_values=terminal
write=s.set_user_attr
def outer_write(key,value):
    if key=='phase4_runtime_stop':raise OSError('injected stop write outage')
    if key=='phase4_runtime_seconds' and s.trials and 'phase4_wall_seconds' in s.trials[-1].user_attrs:
        (db.parent/'outer-write-failed').touch()
        raise OSError('injected outer runtime write failure')
    return write(key,value)
s.set_user_attr=outer_write
def objective(*unused):
    state['started']=time.perf_counter();time.sleep(.025);return 1.
o.objective=objective
a=SimpleNamespace(phase4=True,n_trials=1,n_jobs=1,seed=42,
    max_study_seconds=.1 if ceiling=='study' else None,max_trial_seconds=.1 if ceiling=='trial' else None)
o.optimize_loop(s,None,None,a)
'''


@pytest.mark.parametrize('mode',['blocked','failed'])
@pytest.mark.parametrize('ceiling',['study','trial'])
def test_5d_terminal_commit_guard_and_conservative_stop(unit,monkeypatch,mode,ceiling):
    proc=subprocess.Popen([sys.executable,'-c',TERMINAL,str(unit),mode,ceiling],
        env={**os.environ,'PYTHONPATH':str(Path.cwd())},stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    out,err=finish(proc)
    assert proc.returncode==70 and any(reason in err for reason in ('stale','terminal state','max_trial_seconds')),(out,err)
    if mode=='failed':assert (unit.parent/'outer-write-failed').exists()
    study=optuna.load_study(storage='sqlite:///'+str(unit),study_name='unit')
    assert study.trials[0].state.name=='RUNNING'
    assert 'phase4_wall_seconds' in study.trials[0].user_attrs
    args=SimpleNamespace(phase4=True,n_jobs=1,n_trials=1,seed=42,
        max_study_seconds=.1 if ceiling=='study' else None,max_trial_seconds=.1 if ceiling=='trial' else None)
    opt.reconcile_phase4_trials(study,args)
    record=study.trials[0].user_attrs['phase4_reconciled']
    assert record['accounting']=='conservative_upper_bound'
    assert record['accounted_seconds']>=float((unit.parent/'elapsed.txt').read_text())
    stop=study.user_attrs['phase4_runtime_stop']
    assert stop['reason']=='max_'+ceiling+'_seconds' and stop['reconciled'] is True
    attrs=study.user_attrs;opt.reconcile_phase4_trials(study,args);assert study.user_attrs==attrs
    opt.optimize_loop(study,None,None,args);assert len(study.trials)==1
    host_clear(unit,study)
    monkeypatch.setattr(opt,'objective',lambda *a:1.)
    args.max_trial_seconds=args.max_study_seconds=None
    opt.optimize_loop(study,None,None,args);assert len(opt.complete_trials(study))==1


@pytest.mark.parametrize('state',['RUNNING','WAITING'])
def test_5d_pending_final_wall_charged_once_using_larger_wall(unit,state):
    study=optuna.create_study(storage='sqlite:///'+str(unit),study_name='unit')
    if state=='RUNNING':trial_id=study.ask()._trial_id
    else:study.enqueue_trial({});trial_id=study.trials[0]._trial_id
    for key,value in {'phase4_wall_seconds':8.,'phase4_timing':{
        'started_utc':'2026-11-06T00:00:00+00:00','heartbeat_utc':'2026-11-06T00:00:02+00:00',
        'wall_seconds':2.,'runtime_seconds':2.,'stale_seconds':90.}}.items():
        study._storage.set_trial_user_attr(trial_id,key,value)
    args=SimpleNamespace(phase4=True,max_study_seconds=.1)
    opt.reconcile_phase4_trials(study,args)
    record=study.trials[0].user_attrs['phase4_reconciled']
    assert record['accounted_seconds']==98. and record['runtime_seconds']>=98.
    attrs=study.user_attrs;opt.reconcile_phase4_trials(study,args);assert study.user_attrs==attrs
    assert study.user_attrs['phase4_runtime_stop']['reason']=='max_study_seconds'


RESULT=r'''
import sys,json
from pathlib import Path
from types import SimpleNamespace
import optuna
from scripts import optimize_orderflow as o,phase4_resources as r
db=Path(sys.argv[1]);step=sys.argv[2];fault=sys.argv[3]
s=optuna.create_study(storage='sqlite:///'+str(db),study_name='unit',load_if_exists=True)
s.set_user_attr('identity',{'phase':'phase4','unit':'result'});s.set_user_attr('versions',{'unit':'1'})
a=SimpleNamespace(phase4=True,n_trials=1,n_jobs=1,seed=42,symbol='BTCUSDT',bar_type='fixed_tick_imbalance')
o.objective=lambda *a:1.
o.optimize_loop(s,None,None,a)
if step=='first':
    if fault=='bound':r.current_memory=lambda:{'peak_rss_bytes':5*r.GiB,'private_bytes':100000000}
    else:
        def unavailable():raise OSError('injected '+fault+' measurement failure')
        setattr(r,fault,unavailable)
result=o.study_result(s,a,{},s.user_attrs['identity'])
(db.parent/'result.json').write_text(json.dumps(result))
'''


@pytest.mark.parametrize('fault',['bound','current_memory','process_memory'])
def test_5d_result_memory_stop_child_failure_and_retry_gate(unit,fault):
    def run(step):return subprocess.run([sys.executable,'-c',RESULT,str(unit),step,fault],
        env={**os.environ,'PYTHONPATH':str(Path.cwd())},capture_output=True,text=True,timeout=10)
    first=run('first');assert first.returncode!=0
    study=optuna.load_study(storage='sqlite:///'+str(unit),study_name='unit')
    stop=study.user_attrs['phase4_runtime_stop']
    assert stop['reason']=='memory' and stop['live'] is False and stop['phase']=='result' and stop['utc']
    if fault=='bound':assert stop['measurement']['peak_rss_bytes']==5*GiB
    else:assert 'injected '+fault in stop['error']
    retry=run('retry');assert retry.returncode!=0 and 'owner decision' in retry.stderr
    assert study.user_attrs['phase4_runtime_stop']==stop and len(study.trials)==1
    assert not (unit.parent/'result.json').exists()
    host_clear(unit,study)
    cleared=run('retry');assert cleared.returncode==0,cleared.stderr
    assert (unit.parent/'result.json').exists() and len(study.trials)==1


def test_5d_prior_trial_stop_retained_with_post_trial_memory_evidence(unit,monkeypatch):
    study=optuna.create_study(storage='sqlite:///'+str(unit),study_name='unit')
    earlier={'reason':'max_trial_seconds','trial':0,'measured_seconds':2.,'limit_seconds':1.,'utc':'earlier'}
    def objective(*a):study.set_user_attr('phase4_runtime_stop',earlier);return 1.
    monkeypatch.setattr(opt,'objective',objective)
    monkeypatch.setattr(resources,'enforce_current',lambda *a:resources.enforce_memory(BREACH))
    with pytest.raises(ValueError):opt.optimize_loop(study,None,None,SimpleNamespace(phase4=True,n_jobs=1,n_trials=1,seed=42))
    stop=study.user_attrs['phase4_runtime_stop']
    assert {k:stop[k] for k in earlier}==earlier
    assert stop['memory_breach']['measurement']==BREACH


@pytest.mark.parametrize('target',['current_memory','process_memory'])
def test_5d_post_trial_measurement_exception_persists_stop(unit,monkeypatch,target):
    study=optuna.create_study(storage='sqlite:///'+str(unit),study_name='unit')
    study.set_user_attr('identity',{'phase':'phase4','unit':'measurement'})
    original_check=resources.enforce_current;original_sample=getattr(resources,target)
    def unavailable():raise OSError('injected '+target+' failure')
    def check(*a):
        monkeypatch.setattr(resources,target,unavailable)
        return original_check(*a)
    monkeypatch.setattr(resources,'enforce_current',check)
    monkeypatch.setattr(opt,'objective',lambda *a:1.)
    args=SimpleNamespace(phase4=True,n_jobs=1,n_trials=2,seed=42)
    with pytest.raises(OSError,match=target):opt.optimize_loop(study,None,None,args)
    stop=study.user_attrs['phase4_runtime_stop']
    assert stop['reason']=='memory' and stop['live'] is False and target in stop['error']
    monkeypatch.setattr(resources,target,original_sample)
    monkeypatch.setattr(resources,'enforce_current',original_check)
    opt.optimize_loop(study,None,None,args);assert len(study.trials)==1


def test_5d_result_memory_failure_records_stopped_stage(driver_unit,monkeypatch):
    from scripts import measure_rss
    u=driver_unit;u.execute('A');original=measure_rss.run_measured
    monkeypatch.setattr(resources,'current_memory',lambda:dict(BREACH))
    def child(command,**kw):
        result=original(command,**kw)
        if Path(command[1]).name=='optimize_orderflow.py':
            name=opt.list_study_names(u.db)[0]
            study=optuna.load_study(storage='sqlite:///'+str(u.db),study_name=name)
            try:opt.study_result(study,SimpleNamespace(phase4=True),{},study.user_attrs['identity'])
            except ValueError:result['returncode']=2
        return result
    monkeypatch.setattr(measure_rss,'run_measured',child)
    with pytest.raises(ValueError):u.execute('B')
    path,stopped=checkpoint(u,'B',True)
    assert stopped['runtime_stops'][0]['stop']['phase']=='result'
    assert not path.with_name('stage-B.json').exists()

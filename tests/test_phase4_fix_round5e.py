"""Write-ahead failure injection; only temporary synthetic studies and host records."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import optuna
import pytest

from scripts import optimize_orderflow as opt,phase4_resources as resources
from scripts import phase4_campaign as campaign,phase4_studies as p4,phase4_report as report
from test_phase4_fix_round5c import unit,host_clear
from test_phase4_fix_round5d import BREACH
from test_optimize_orderflow import args_for
from test_warm_start import valid_params


def completed(db):
    study=optuna.create_study(storage='sqlite:///'+str(db),study_name='unit')
    study.set_user_attr('identity',{'phase':'phase4','campaign_sha256':'unit','arm':'unit','trade_signals':'disabled'})
    study.set_user_attr('versions',{})
    study.add_trial(optuna.trial.create_trial(value=1.,user_attrs={
        'feasible':True,'params':valid_params(),'replay_digest':'unit','n_scored_bars':1,'total_trades':1}))
    return study


SLOW_COMMIT=r'''
import sys,time
from pathlib import Path
from types import SimpleNamespace
import optuna
from scripts import optimize_orderflow as o
db=Path(sys.argv[1]);fault=sys.argv[2]
s=optuna.create_study(storage='sqlite:///'+str(db),study_name='unit')
s.set_user_attr('identity',{'phase':'phase4','unit':'write-ahead'})
o.PHASE4_HEARTBEAT_SECONDS=.1;o.PHASE4_HEARTBEAT_STALE_SECONDS=.4
backend=getattr(s._storage,'_backend',s._storage);original=backend.set_trial_state_values
def commit(trial_id,state=None,values=None):
    if state.is_finished():
        (db.parent/'terminal-attempt').touch()
        time.sleep(.12)
    result=original(trial_id,state=state,values=values)
    if state.is_finished():(db.parent/'elapsed').write_text(str(time.perf_counter()-begun))
    return result
backend.set_trial_state_values=commit
write=s.set_user_attr
def injected(key,value):
    if key=='phase4_accounting_pending' and value is not None and fault=='marker':raise OSError('marker write failed')
    if key=='phase4_runtime_seconds' and s.trials and 'phase4_wall_seconds' in s.trials[-1].user_attrs:
        raise OSError('runtime write failed after successful commit')
    return write(key,value)
s.set_user_attr=injected
def objective(*unused):
    global begun
    begun=time.perf_counter();time.sleep(.02);return 1.
o.objective=objective
o_args=SimpleNamespace(phase4=True,n_trials=1,n_jobs=1,seed=42,max_study_seconds=.1)
o.optimize_loop(s,None,None,o_args)
'''


def test_5e_complete_commit_runtime_loss_accounted_once_and_stopped(unit):
    child=subprocess.run([sys.executable,'-c',SLOW_COMMIT,str(unit),'runtime'],
        env={**os.environ,'PYTHONPATH':str(Path.cwd())},capture_output=True,text=True,timeout=10)
    assert child.returncode!=0
    study=optuna.load_study(storage='sqlite:///'+str(unit),study_name='unit')
    assert study.trials[0].state.name=='COMPLETE'
    pending=study.user_attrs['phase4_accounting_pending']
    assert {'trial','started_utc','heartbeat_utc','wall_seconds','runtime_seconds_before','stale_bound'}<=pending.keys()
    before=study.trials[0]
    args=SimpleNamespace(phase4=True,n_jobs=1,n_trials=2,seed=42,max_study_seconds=.1)
    opt.reconcile_phase4_trials(study,args)
    assert study.trials[0]==before
    assert study.user_attrs['phase4_runtime_seconds']>=float((unit.parent/'elapsed').read_text())
    assert study.user_attrs['phase4_runtime_stop']['reason']=='max_study_seconds'
    assert study.user_attrs.get('phase4_accounting_pending') is None
    attrs=study.user_attrs;opt.reconcile_phase4_trials(study,args);assert study.user_attrs==attrs
    opt.optimize_loop(study,None,None,args);assert len(study.trials)==1
    host_clear(unit,study)


def test_5e_accounting_marker_failure_prevents_terminal_commit(unit):
    child=subprocess.run([sys.executable,'-c',SLOW_COMMIT,str(unit),'marker'],
        env={**os.environ,'PYTHONPATH':str(Path.cwd())},capture_output=True,text=True,timeout=10)
    assert child.returncode!=0
    assert not (unit.parent/'terminal-attempt').exists()
    study=optuna.load_study(storage='sqlite:///'+str(unit),study_name='unit')
    assert study.trials[0].state.name=='RUNNING'


def accounting_marker():
    return {'trial':0,'started_utc':'2026-11-06T00:00:00+00:00','heartbeat_utc':'2026-11-06T00:00:02+00:00',
        'wall_seconds':3.,'runtime_seconds_before':0.,'stale_bound':90.,
        'limits':{'max_study_seconds':.1,'max_trial_seconds':None}}


@pytest.mark.parametrize('failed_key',['phase4_reconciliation_audit','phase4_runtime_seconds','phase4_runtime_stop','phase4_accounting_pending'])
def test_5e_accounting_repair_write_failure_is_idempotent(unit,monkeypatch,failed_key):
    study=completed(unit);study.set_user_attr('phase4_runtime_seconds',0.)
    study.set_user_attr('phase4_accounting_pending',accounting_marker())
    write=study.set_user_attr;failed=[]
    def injected(key,value):
        if key==failed_key and not failed:
            failed.append(key);raise OSError('repair write failed')
        return write(key,value)
    monkeypatch.setattr(study,'set_user_attr',injected)
    args=SimpleNamespace(phase4=True,max_study_seconds=.1)
    with pytest.raises(OSError,match='repair write'):opt.reconcile_phase4_trials(study,args)
    assert study.user_attrs.get('phase4_accounting_pending') is not None
    opt.reconcile_phase4_trials(study,args)
    assert study.user_attrs['phase4_runtime_seconds']==93.
    assert study.user_attrs['phase4_runtime_stop']['reason']=='max_study_seconds'
    assert study.user_attrs.get('phase4_accounting_pending') is None
    attrs=study.user_attrs;opt.reconcile_phase4_trials(study,args);assert study.user_attrs==attrs


@pytest.mark.parametrize('phase',['post_trial','result'])
def test_5e_memory_marker_failure_prevents_measurement(unit,monkeypatch,phase):
    study=completed(unit);write=study.set_user_attr;measured=[]
    def injected(key,value):
        if key=='phase4_memory_check_pending':raise OSError('memory marker write failed')
        return write(key,value)
    monkeypatch.setattr(study,'set_user_attr',injected)
    with pytest.raises(OSError,match='marker write'):
        with opt.phase4_memory_check(study,phase,0):measured.append(True)
    assert not measured


def admit(route,study,db,tmp_path,monkeypatch):
    if route=='open':
        monkeypatch.setattr(opt,'phase4_study_preflight',lambda *a:pytest.fail('preflight before pending repair'))
        return opt.open_study(args_for(tmp_path,phase4=True,study_db=db),'unit',study.user_attrs['identity'])
    if route=='loop':return opt.optimize_loop(study,None,None,SimpleNamespace(phase4=True,n_jobs=1,n_trials=2,seed=42))
    if route=='reconcile':return opt.reconcile_phase4_trials(study,SimpleNamespace(phase4=True))
    if route=='clear':return campaign.clear_runtime_stop(db,'unit','0'*64)
    if route=='result':return opt.study_result(study,SimpleNamespace(phase4=True),{},study.user_attrs['identity'])
    if route=='candidate':return p4.completed_candidate(db,'unit',0,1)
    if route=='freeze':return campaign.freeze_baseline(db,'unit',1,tmp_path/'output/phase4/frozen/unit.json')
    if route=='holdout':
        monkeypatch.setattr(p4,'campaign_arm',lambda *a:pytest.fail('holdout before pending repair'))
        return p4.holdout_look(SimpleNamespace(holdout_look='unit:0',study_db=db),[])
    if route=='report':
        snapshots=[]
        for n in range(3):
            path=tmp_path/f'legacy{n}';path.write_bytes(bytes([n]));snapshots.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        return report.report_snapshots({'snapshots':snapshots,'phase4_db':str(db),'arms':{'unit':{'budget':1}}},'unit',db)


@pytest.mark.parametrize('route',['open','loop','reconcile','clear','result','candidate','report','holdout','freeze'])
@pytest.mark.parametrize('phase',['post_trial','result'])
def test_5e_lost_memory_stop_repaired_before_every_admission(unit,tmp_path,monkeypatch,route,phase):
    study=completed(unit);write=study.set_user_attr
    def injected(key,value):
        if key=='phase4_runtime_stop':raise OSError('transient stop write failure')
        return write(key,value)
    monkeypatch.setattr(study,'set_user_attr',injected)
    with pytest.raises(OSError,match='stop write'):
        with opt.phase4_memory_check(study,phase,0):resources.enforce_memory(BREACH)
    assert study.user_attrs['phase4_memory_check_pending']['phase']==phase
    assert study.user_attrs.get('phase4_runtime_stop') is None
    monkeypatch.setattr(study,'set_user_attr',write)
    if route in ('loop','reconcile'):admit(route,study,unit,tmp_path,monkeypatch)
    else:
        with pytest.raises(ValueError):admit(route,study,unit,tmp_path,monkeypatch)
    stop=study.user_attrs['phase4_runtime_stop']
    assert stop['reason']=='memory' and stop['unresolved_check'] is True and stop['phase']==phase and stop['trial']==0
    assert study.user_attrs.get('phase4_memory_check_pending') is None
    assert len(study.trials)==1
    with pytest.raises(ValueError,match='owner decision'):opt.study_result(study,SimpleNamespace(phase4=True),{},study.user_attrs['identity'])
    host_clear(unit,study)
    assert study.user_attrs['phase4_runtime_stop'] is None


def test_5e_passed_memory_check_clear_failure_still_blocks(unit,monkeypatch):
    study=completed(unit);write=study.set_user_attr
    def injected(key,value):
        if key=='phase4_memory_check_pending' and value is None:raise OSError('clear failed')
        return write(key,value)
    monkeypatch.setattr(study,'set_user_attr',injected)
    with pytest.raises(OSError,match='clear failed'):
        with opt.phase4_memory_check(study,'result',0):pass
    monkeypatch.setattr(study,'set_user_attr',write)
    with pytest.raises(ValueError,match='owner decision'):opt.study_result(study,SimpleNamespace(phase4=True),{},study.user_attrs['identity'])
    assert study.user_attrs['phase4_runtime_stop']['unresolved_check'] is True


@pytest.mark.parametrize('route',['open','loop','reconcile','clear','result','candidate','report','holdout','freeze'])
def test_5e_accounting_marker_repaired_before_every_admission(unit,tmp_path,monkeypatch,route):
    study=completed(unit);study.set_user_attr('phase4_runtime_seconds',0.)
    study.set_user_attr('phase4_accounting_pending',accounting_marker())
    before=study.trials[0]
    if route in ('loop','reconcile'):admit(route,study,unit,tmp_path,monkeypatch)
    else:
        with pytest.raises(ValueError):admit(route,study,unit,tmp_path,monkeypatch)
    assert study.trials[0]==before
    assert study.user_attrs['phase4_runtime_seconds']==93.
    assert study.user_attrs['phase4_runtime_stop']['reason']=='max_study_seconds'
    assert study.user_attrs.get('phase4_accounting_pending') is None


@pytest.mark.parametrize('location',['trial','study'])
def test_5e_accounting_marker_reuses_existing_conservative_record(unit,location):
    study=optuna.create_study(storage='sqlite:///'+str(unit),study_name='unit')
    study.set_user_attr('identity',{'phase':'phase4'})
    record={'trial':0,'accounting':'conservative_upper_bound','accounted_seconds':93.,'runtime_seconds':93.}
    if location=='trial':
        study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL,user_attrs={'phase4_reconciled':record}))
    else:
        study.add_trial(optuna.trial.create_trial(value=1.))
        study.set_user_attr('phase4_reconciliation_audit',[record])
    study.set_user_attr('phase4_runtime_seconds',93.)
    study.set_user_attr('phase4_accounting_pending',accounting_marker())
    args=SimpleNamespace(phase4=True,max_study_seconds=.1)
    opt.reconcile_phase4_trials(study,args)
    assert study.user_attrs['phase4_runtime_seconds']==93.
    assert study.user_attrs['phase4_reconciliation_audit']==[record]
    attrs=study.user_attrs;opt.reconcile_phase4_trials(study,args);assert study.user_attrs==attrs


@pytest.mark.parametrize('failed_key',['phase4_runtime_stop','phase4_memory_check_pending'])
def test_5e_memory_repair_write_failure_keeps_evidence(unit,monkeypatch,failed_key):
    study=completed(unit)
    study.set_user_attr('phase4_memory_check_pending',{'phase':'result','trial':0,'utc':'2026-11-06T00:00:00Z'})
    write=study.set_user_attr;failed=[]
    def injected(key,value):
        if key==failed_key and not failed:
            failed.append(key);raise OSError('repair failed')
        return write(key,value)
    monkeypatch.setattr(study,'set_user_attr',injected)
    args=SimpleNamespace(phase4=True)
    with pytest.raises(OSError,match='repair failed'):opt.study_result(study,args,{},study.user_attrs['identity'])
    assert study.user_attrs['phase4_memory_check_pending'] is not None
    with pytest.raises(ValueError,match='owner decision'):opt.study_result(study,args,{},study.user_attrs['identity'])
    assert study.user_attrs['phase4_memory_check_pending'] is None
    stop=study.user_attrs['phase4_runtime_stop']
    with pytest.raises(ValueError,match='owner decision'):opt.study_result(study,args,{},study.user_attrs['identity'])
    assert study.user_attrs['phase4_runtime_stop']==stop


def test_5e_check_does_not_replace_an_unresolved_marker(unit):
    study=completed(unit);measured=[]
    study.set_user_attr('phase4_memory_check_pending',{'phase':'post_trial','trial':0,'utc':'2026-11-06T00:00:00Z'})
    with pytest.raises(ValueError,match='owner decision'):
        with opt.phase4_memory_check(study,'result',0):measured.append(True)
    assert not measured
    assert study.user_attrs['phase4_runtime_stop']['phase']=='post_trial'


def test_5e_repair_audit_takes_precedence_over_partial_trial_record(unit):
    study=optuna.create_study(storage='sqlite:///'+str(unit),study_name='unit')
    study.set_user_attr('identity',{'phase':'phase4'})
    trial=study.ask();trial.set_user_attr('phase4_reconciled',{'trial':0,'reason':'partial audit'})
    marker=accounting_marker();marker['limits']={'max_trial_seconds':.1,'max_study_seconds':None}
    study.set_user_attr('phase4_runtime_seconds',0.)
    study.set_user_attr('phase4_accounting_pending',marker)
    opt.reconcile_phase4_trials(study,SimpleNamespace(phase4=True))
    assert study.user_attrs['phase4_runtime_seconds']==93.
    assert study.user_attrs['phase4_runtime_stop']['reason']=='max_trial_seconds'

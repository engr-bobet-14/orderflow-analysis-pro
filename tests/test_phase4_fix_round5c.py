"""Inspection-6 regressions; all studies, records and spikes are synthetic."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import optuna
import pytest

from scripts import optimize_orderflow as opt,phase4_resources as resources
from scripts import phase4_campaign as campaign,phase4_policy as policy,phase4_approvals as approvals
from test_phase4_fix_round5 import finish

GiB=1024**3


def test_5c_current_memory_peak_includes_instantaneous_sample(monkeypatch):
    import resource
    monkeypatch.setattr(resources,'process_memory',lambda:{'rss_bytes':5*GiB,'private_bytes':100000000})
    monkeypatch.setattr(resource,'getrusage',lambda who:SimpleNamespace(ru_maxrss=0))
    sample=resources.current_memory()
    assert sample['peak_rss_bytes']==5*GiB
    assert not resources.memory_bounds(sample)['pass']


@pytest.fixture
def unit(tmp_path,monkeypatch):
    for module in (opt,campaign,policy):monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(approvals,'approval_root',lambda:tmp_path/'approvals/phase4')
    db=tmp_path/'output/optimize/phase4/unit.db';db.parent.mkdir(parents=True)
    return db


def host_clear(db,study):
    record={'plan_sha256':approvals.PLAN_SHA256,'study_identity_sha256':opt.fingerprint(study.user_attrs['identity']),
        'stop_sha256':opt.fingerprint(study.user_attrs['phase4_runtime_stop']),
        'decision':'clear_and_restart_budget','created_utc':'2026-11-06T00:00:00Z'}
    raw=policy.canonical_bytes(record);sha=hashlib.sha256(raw).hexdigest()
    path=approvals.approval_root()/('runtime-decision-'+sha+'.json');path.parent.mkdir(parents=True);path.write_bytes(raw)
    return campaign.clear_runtime_stop(db,'unit',sha)


PEAK_SCRIPT=r'''
import sys,time,resource
from pathlib import Path
from types import SimpleNamespace
import optuna
from scripts import optimize_orderflow as o,phase4_resources as r
db=Path(sys.argv[1]);mode=sys.argv[2]
s=optuna.create_study(storage='sqlite:///'+str(db),study_name='unit')
s.set_user_attr('identity',{'phase':'phase4','unit':'peaks'})
o.PHASE4_HEARTBEAT_SECONDS=.05;o.PHASE4_HEARTBEAT_STALE_SECONDS=1.
state={'peak':5*r.GiB if mode=='first_lost' else 100000000}
def memory():
    return {'rss_bytes':state['peak'] if mode=='first_lost' else 100000000,'private_bytes':100000000,
        'peak_private_bytes':state['peak'] if mode=='private' else 100000000,
        'private_metric':'synthetic lifetime peak','private_is_fallback':False}
r.process_memory=memory
def usage(who):
    peak=100000000 if mode=='private' else state['peak']
    return SimpleNamespace(ru_maxrss=peak if sys.platform=='darwin' else peak/1024)
resource.getrusage=usage
original=optuna.study.Study.set_user_attr
def write(self,key,value):
    if key=='phase4_runtime_stop' and mode=='first_lost':raise OSError('injected stop write outage')
    return original(self,key,value)
optuna.study.Study.set_user_attr=write
def objective(*unused):
    state['peak']=5*r.GiB  # Instantaneous RSS/private have already returned to normal at the next pulse.
    time.sleep(.2)
    return 1.
o.objective=objective
o.optimize_loop(s,None,None,SimpleNamespace(phase4=True,n_trials=2,n_jobs=1,seed=42))
'''


@pytest.mark.parametrize('metric',['rss','private'])
def test_5c_peak_between_pulses_stops_and_requires_host_clear(unit,monkeypatch,metric):
    proc=subprocess.Popen([sys.executable,'-c',PEAK_SCRIPT,str(unit),metric],
        env={**os.environ,'PYTHONPATH':str(Path.cwd())},stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    out,err=finish(proc)
    assert proc.returncode==70 and 'memory' in err,(out,err)
    study=optuna.load_study(storage='sqlite:///'+str(unit),study_name='unit')
    stop=study.user_attrs['phase4_runtime_stop']
    assert stop['reason']=='memory' and stop['live'] is True and stop['trial']==0
    assert stop['measurement'][metric+'_bytes']==100000000
    assert stop['bounds'][metric+'_bytes']==5*GiB
    if metric=='private':assert stop['bounds']['rss_bytes']<4*GiB
    assert study.trials[0].user_attrs['phase4_memory']['peak_'+metric+'_bytes']==5*GiB
    args=SimpleNamespace(phase4=True,n_jobs=1,n_trials=1,seed=42)
    opt.reconcile_phase4_trials(study,args)
    opt.optimize_loop(study,None,None,args)
    assert len(study.trials)==1 and not opt.complete_trials(study)
    host_clear(unit,study)
    monkeypatch.setattr(opt,'objective',lambda *a:1.)
    opt.optimize_loop(study,None,None,args)
    assert len(opt.complete_trials(study))==1 and study.user_attrs['phase4_runtime_stop'] is None


@pytest.mark.parametrize('where',['post_trial','objective'])
def test_5c_memory_exception_persists_stop_before_reraise(unit,monkeypatch,where):
    study=optuna.create_study(storage='sqlite:///'+str(unit),study_name='unit')
    study.set_user_attr('identity',{'phase':'phase4','unit':'memory-error'})
    sample={'rss_bytes':100000000,'peak_rss_bytes':5*GiB,'private_bytes':100000000,'peak_private_bytes':100000000}
    def breach(*unused):return resources.enforce_memory(sample)
    monkeypatch.setattr(opt,'objective',breach if where=='objective' else lambda *a:1.)
    if where=='post_trial':monkeypatch.setattr(resources,'enforce_current',breach)
    args=SimpleNamespace(phase4=True,n_jobs=1,n_trials=2,seed=42)
    with pytest.raises(ValueError,match='memory'):opt.optimize_loop(study,None,None,args)
    stop=study.user_attrs['phase4_runtime_stop']
    assert stop['reason']=='memory' and stop['live'] is False and stop['trial']==0
    assert stop['measurement']==sample and stop['bounds']['rss_bytes']==5*GiB
    assert stop['bounds']['rss_limit_bytes']==stop['bounds']['private_limit_bytes']==4*GiB
    assert stop['utc']
    monkeypatch.setattr(resources,'enforce_current',lambda *a:{'pass':True})
    monkeypatch.setattr(opt,'objective',lambda *a:1.)
    opt.optimize_loop(study,None,None,args)
    assert len(study.trials)==1
    host_clear(unit,study)
    opt.optimize_loop(study,None,None,args)
    assert len(opt.complete_trials(study))==2


def test_5c_first_checkpoint_lost_stop_reconstructed_without_timing(unit):
    proc=subprocess.Popen([sys.executable,'-c',PEAK_SCRIPT,str(unit),'first_lost'],
        env={**os.environ,'PYTHONPATH':str(Path.cwd())},stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    out,err=finish(proc)
    assert proc.returncode==70 and 'Stop persistence failed' in err,(out,err)
    study=optuna.load_study(storage='sqlite:///'+str(unit),study_name='unit')
    trial=study.trials[0]
    assert 'phase4_timing' not in trial.user_attrs
    assert trial.user_attrs['phase4_heartbeat_health']['memory_breach']['bounds']['rss_bytes']==5*GiB
    assert study.user_attrs.get('phase4_runtime_stop') is None
    args=SimpleNamespace(phase4=True,n_jobs=1,n_trials=1,seed=42)
    opt.reconcile_phase4_trials(study,args)
    stop=study.user_attrs['phase4_runtime_stop']
    assert stop['reason']=='memory' and stop['reconciled'] is True and stop['live'] is False
    assert stop['measurement']['peak_rss_bytes']==5*GiB
    opt.optimize_loop(study,None,None,args)
    assert len(study.trials)==1 and not opt.complete_trials(study)


@pytest.mark.parametrize('valid_record',[False,True])
def test_5c_clear_live_stop_reconciles_pending_only_after_valid_record(unit,valid_record):
    study=optuna.create_study(storage='sqlite:///'+str(unit),study_name='unit')
    study.set_user_attr('identity',{'phase':'phase4','unit':'clear-live'})
    trial=study.ask()
    trial.set_user_attr('phase4_timing',{'started_utc':'2026-11-06T00:00:00+00:00',
        'heartbeat_utc':'2026-11-06T00:00:02+00:00','wall_seconds':2.,'runtime_seconds':2.,'stale_seconds':90.})
    study.enqueue_trial({})
    study.set_user_attr('phase4_runtime_stop',{'reason':'memory','live':True,'trial':0})
    before_attrs=study.user_attrs;before_trials=study.trials
    if not valid_record:
        before_bytes=unit.read_bytes()
        with pytest.raises(ValueError):campaign.clear_runtime_stop(unit,'unit','0'*64)
        assert study.user_attrs==before_attrs and study.trials==before_trials
        assert [t.state.name for t in study.trials]==['RUNNING','WAITING']
        assert unit.read_bytes()==before_bytes
    else:
        entry=host_clear(unit,study)
        assert study.user_attrs['phase4_runtime_stop'] is None
        assert [t.state.name for t in study.trials]==['FAIL','FAIL']
        assert study.trials[0].user_attrs['phase4_reconciled']['accounted_seconds']==92.
        assert len(study.user_attrs['phase4_reconciliation_audit'])==2
        assert study.trials[1].user_attrs['phase4_reconciled']['accounted_seconds']==90.
        assert entry['reset']=={'runtime_seconds':182.,'last_trial_number':1}
        assert entry['stop']==before_attrs['phase4_runtime_stop']
        assert study.user_attrs['phase4_runtime_decisions']==[entry]

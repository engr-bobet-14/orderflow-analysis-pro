"""5f regressions: temporary stage attempts/studies, no real approvals or experiments."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import optuna
import pytest

from scripts import optimize_orderflow as opt,phase4_campaign as campaign
from scripts import phase4_report as report,phase4_studies as p4,phase4_policy as policy
from test_phase4_fix_round5b import driver_unit
from test_phase4_fix_round5c import unit,host_clear


@pytest.mark.parametrize('evidence',['log','archive'])
def test_5f_parent_selector_cannot_cycle_after_two_stopped_attempts(driver_unit,evidence):
    u=driver_unit
    for number,selector in enumerate(('approval-A','approval-B'),1):
        u.fault['child']=number
        with pytest.raises(ValueError,match='child'):u.execute('A',selector)
    root=u.root/'output/phase4/checkpoints'/u.digest
    log=u.root/'output/phase4/campaign-log.jsonl'
    if evidence=='log':
        for path in [*root.glob('stage-*.json'),*(root/'records').glob('*.json')]:path.unlink()
    else:
        events=[json.loads(line) for line in log.read_text().splitlines()]
        log.write_bytes(b''.join(policy.canonical_bytes(e) for e in events if e.get('owner_approval_sha256')!='approval-A'))
    before=log.read_bytes();u.fault.clear()
    with pytest.raises(ValueError,match='new.*approval|already used'):u.execute('A','approval-A')
    assert len(u.calls)==2 and log.read_bytes()==before
    u.execute('A','approval-C');assert len(u.calls)==3


@pytest.mark.parametrize('evidence',['log','checkpoint'])
def test_5f_parent_selector_cannot_move_to_another_stage(driver_unit,evidence):
    u=driver_unit;u.execute('A','one-use-selector')
    if evidence=='checkpoint':
        (u.root/'output/phase4/campaign-log.jsonl').unlink()
        for path in (u.root/'output/phase4/checkpoints'/u.digest/'records').glob('*.json'):path.unlink()
    with pytest.raises(ValueError,match='new.*approval|already used'):u.execute('B','one-use-selector')
    assert len(u.calls)==1


COMMIT=r'''
import sys,time,threading
from pathlib import Path
from types import SimpleNamespace
import optuna
from scripts import optimize_orderflow as o
db=Path(sys.argv[1])
s=optuna.create_study(storage='sqlite:///'+str(db),study_name='unit')
s.set_user_attr('identity',{'phase':'phase4','unit':'terminal-duration'})
# Delay the live watchdog's poll so the post-commit ceiling must catch this breach.
Event=threading.Event
class DelayedPoll(Event):
    def wait(self,timeout=None):return super().wait(1. if timeout==.25 else timeout)
threading.Event=DelayedPoll
backend=getattr(s._storage,'_backend',s._storage);commit=backend.set_trial_state_values
def slow(trial_id,state=None,values=None):
    if state.is_finished():time.sleep(.15)
    return commit(trial_id,state=state,values=values)
backend.set_trial_state_values=slow
o.objective=lambda *a:1.
o.optimize_loop(s,None,None,SimpleNamespace(phase4=True,n_jobs=1,n_trials=2,seed=42,max_trial_seconds=.1))
'''


def test_5f_trial_ceiling_includes_durable_terminal_commit(unit,monkeypatch):
    child=subprocess.run([sys.executable,'-c',COMMIT,str(unit)],env={**os.environ,'PYTHONPATH':str(Path.cwd())},
        capture_output=True,text=True,timeout=10)
    assert child.returncode==0,child.stderr
    study=optuna.load_study(storage='sqlite:///'+str(unit),study_name='unit')
    assert len(study.trials)==1 and study.trials[0].state.name=='COMPLETE'
    duration=study.user_attrs['phase4_terminal_wall_seconds']['0']
    assert duration>=.15 and study.trials[0].user_attrs['phase4_wall_seconds']<.1
    stop=study.user_attrs['phase4_runtime_stop']
    assert stop['reason']=='max_trial_seconds' and stop['measured_seconds']==duration
    assert study.user_attrs['phase4_accounting_pending'] is None
    args=SimpleNamespace(phase4=True,n_jobs=1,n_trials=2,seed=42,max_trial_seconds=.1)
    opt.optimize_loop(study,None,None,args);assert len(study.trials)==1
    with pytest.raises(ValueError,match='owner decision'):opt.study_result(study,args,{},study.user_attrs['identity'])
    host_clear(unit,study)
    monkeypatch.setattr(opt,'objective',lambda *a:1.)
    opt.optimize_loop(study,None,None,args)
    assert len(opt.complete_trials(study))==2 and study.user_attrs['phase4_runtime_stop'] is None


def test_5f_failed_terminal_duration_write_retains_accounting_marker(unit,monkeypatch):
    study=optuna.create_study(storage='sqlite:///'+str(unit),study_name='unit')
    study.set_user_attr('identity',{'phase':'phase4','unit':'duration-write-failure'})
    write=study.set_user_attr
    def injected(key,value):
        if key=='phase4_terminal_wall_seconds':raise OSError('duration persistence failed')
        return write(key,value)
    monkeypatch.setattr(study,'set_user_attr',injected)
    monkeypatch.setattr(opt,'objective',lambda *a:1.)
    args=SimpleNamespace(phase4=True,n_jobs=1,n_trials=1,seed=42,max_trial_seconds=.1)
    with pytest.raises(OSError,match='duration persistence'):opt.optimize_loop(study,None,None,args)
    assert study.trials[0].state.name=='COMPLETE'
    assert study.user_attrs.get('phase4_accounting_pending') is not None
    monkeypatch.setattr(study,'set_user_attr',write)
    opt.reconcile_phase4_trials(study,args)
    assert study.user_attrs['phase4_runtime_stop']['reason']=='max_trial_seconds'
    assert study.user_attrs['phase4_accounting_pending'] is None


def test_5f_usage_requires_rereading_and_resigning_repaired_stop():
    usage=(Path(__file__).resolve().parents[1]/'USAGE.md').read_text()
    recovery=usage.split('The command requires an existing stopped Phase 4 study',1)[1].split('### Fix round 5b',1)[0]
    assert 're-read the stop' in recovery and 're-sign the decision' in recovery


def test_5f_report_lock_refusal_is_parser_error(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(report,'ROOT',tmp_path)
    monkeypatch.setattr(report,'checked_write',lambda p:Path(p))
    monkeypatch.setattr(p4,'read_campaign',lambda *a:({},'unit'))
    def locked(*a,**kw):raise RuntimeError('study lock is held')
    monkeypatch.setattr(report,'report_snapshots',locked)
    with pytest.raises(SystemExit) as error:
        report.main(['--study-db',str(tmp_path/'unit.db'),'--study-name','unit','--campaign-manifest','unit'])
    assert error.value.code==2 and 'error: study lock is held' in capsys.readouterr().err

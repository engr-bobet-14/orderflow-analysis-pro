"""Synthetic C7 end-to-end stage driver; no Optuna study is opened."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import phase4_campaign as campaign, phase4_policy as policy, phase4_report as report
from tests.test_phase4b_operations import fixture_inputs, sha, intervals, benchmark_lane


def test_c7_synthetic_register_s1_s2t_two_trials_s3_isolated(tmp_path,monkeypatch):
    from scripts import phase4b_campaign as driver
    layout,snapshots,comparators,origin=fixture_inputs(tmp_path,monkeypatch)
    monkeypatch.setattr(campaign,'source_state',lambda:{'base':'base','sha256':'source','dirty':False})
    monkeypatch.setattr(driver,'source_state',campaign.source_state)
    monkeypatch.setattr(driver,'require_phase4b_approval',lambda *a,**kw:{'sha256':'approval','gates_sha256':'gates'})
    monkeypatch.setattr(driver,'preflight',lambda *a:None)
    state={str(p):sha(p) for p in (tmp_path/'output/phase4').rglob('*') if p.is_file()}
    s1layout=policy.PhaseLayout('phase4b-s1',root=tmp_path,approvals_base=tmp_path/'approvals')
    campaign.register(s1layout.manifest,snapshots,phase='phase4b-s1',layout=s1layout,
        comparators=comparators,phase4_origin=origin)
    def synthetic_operation(stage,manifest,digest,layout):
        if stage=='S1':return {'lanes':{bt:benchmark_lane() for bt in comparators}}
        if stage=='S2t':
            path=layout.db_path(layout.studies_db);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'synthetic fixture: two complete trial rows')
            return {'trials':[{'number':0,'state':'COMPLETE'},{'number':1,'state':'COMPLETE'}],
                    'net_expectancy_r':float('nan')}
        return {'reports':{'fixed_tick_imbalance':{'G4b_N':False}},'looks':{}}
    first=driver.run_stage(s1layout,'S1','approval1',operation=synthetic_operation)
    assert first['status']=='complete'
    campaign.register(layout.manifest,snapshots,phase='phase4b',layout=layout,
        comparators=comparators,phase4_origin=origin,benchmark=layout.output/'s1/benchmark.json',holdout_intervals=intervals())
    second=driver.run_stage(layout,'S2t','approval2',operation=synthetic_operation)
    assert len(second['result']['trials'])==2
    assert second['result']['net_expectancy_r'] is None
    # S3 requires both studies. Install the synthetic S2v checkpoint through the same path.
    driver.run_stage(layout,'S2v','approval3',operation=lambda *a:{'trials':[{},{}]})
    third=driver.run_stage(layout,'S3','approval4',operation=synthetic_operation)
    assert third['result']['looks']=={}
    with pytest.raises(ValueError,match='already|single'):
        driver.run_stage(layout,'S2t','approval2',operation=synthetic_operation)
    assert {str(p):sha(p) for p in (tmp_path/'output/phase4').rglob('*') if p.is_file()}==state
    for path in tmp_path.rglob('*'):
        if path.is_file() and ('phase4b' in str(path)):
            assert path.is_relative_to(layout.output) or path.is_relative_to(layout.optimize)


def test_c7_driver_durable_stop_and_consumed_attempt(tmp_path,monkeypatch):
    from scripts import phase4b_campaign as driver
    layout,snapshots,comparators,origin=fixture_inputs(tmp_path,monkeypatch)
    s1=policy.PhaseLayout('phase4b-s1',root=tmp_path,approvals_base=tmp_path/'approvals')
    monkeypatch.setattr(campaign,'source_state',lambda:{'base':'base','sha256':'source','dirty':False})
    monkeypatch.setattr(driver,'source_state',campaign.source_state)
    monkeypatch.setattr(driver,'preflight',lambda *a:None)
    monkeypatch.setattr(driver,'require_phase4b_approval',lambda *a,**kw:{'sha256':'approval','gates_sha256':'gates'})
    campaign.register(s1.manifest,snapshots,phase='phase4b-s1',layout=s1,comparators=comparators,phase4_origin=origin)
    def fail(*a):raise RuntimeError('ceiling breach')
    with pytest.raises(RuntimeError):driver.run_stage(s1,'S1','approval',operation=fail)
    stopped=list(s1.checkpoints.rglob('*.stopped.json'))
    assert len(stopped)==1 and json.loads(stopped[0].read_text())['error']=='ceiling breach'
    # rev 13: a stage that raised is an unclassified end -> unknown termination; a retry needs an owner
    # recovery approval, and the consumed stage approval is never reused.
    with pytest.raises(ValueError,match='used|consumed|recovery'):
        driver.run_stage(s1,'S1','approval',operation=lambda *a:pytest.fail('approval reused'))
    digest=hashlib.sha256(s1.manifest.read_bytes()).hexdigest()
    assert driver.recover_termination_audit(s1,digest,'S1')['kind']=='external_or_unknown_termination'


def test_c8_snapshots_phase4b_backup_only_in_own_root(tmp_path,monkeypatch):
    import sqlite3
    layout,snapshots,comparators,origin=fixture_inputs(tmp_path)
    layout.studies_db.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(layout.studies_db) as con:
        con.execute('CREATE TABLE synthetic(value INTEGER)');con.execute('INSERT INTO synthetic VALUES(1)')
        con.execute('CREATE TABLE studies(study_id INTEGER,study_name TEXT)')
        con.execute('CREATE TABLE study_user_attributes(study_id INTEGER,key TEXT,value_json TEXT)')
    manifest={'phase':'phase4b','phase4b_db':str(layout.studies_db),'snapshots':snapshots,'arms':{}}
    result=report.phase4b_report_snapshots(layout,manifest,'c'*64)
    assert len(result)==5 and Path(result[-1]['path']).is_relative_to(layout.snapshots)
    assert all(sha(s['path'])==s['sha256'] for s in snapshots)
    Path(snapshots[0]['path']).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='snapshot'):
        report.phase4b_report_snapshots(layout,manifest,'c'*64)


def test_c7_file_heartbeat_reuses_existing_runtime_watchdog(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from scripts import phase4b_campaign as driver, optimize_orderflow as opt
    layout=policy.PhaseLayout('phase4b-s1',root=tmp_path,approvals_base=tmp_path/'approvals')
    calls=[]
    @contextmanager
    def timing(trial,study,prior,begun,args=None):
        assert args.max_trial_seconds==600
        trial.set_user_attr('heartbeat','ok')
        study.set_user_attr('phase4_runtime_seconds',1.)
        calls.append(1)
        yield
    monkeypatch.setattr(opt,'phase4_trial_timing',timing)
    with driver.file_timing(layout,'campaign','S1-defaults',600):pass
    assert calls==[1]
    paths=list(layout.checkpoints.rglob('*.json'))
    assert paths and all(p.is_relative_to(layout.checkpoints) for p in paths)


def test_c7_campaign_cli_routes_phase4b_before_phase4_policy(monkeypatch):
    from scripts import phase4b_campaign as driver
    monkeypatch.setattr(driver,'main',lambda argv:argv)
    result=campaign.main(['register','--phase','phase4b-s1','--registration','synthetic.json'])
    assert result==['register','--registration','synthetic.json','--phase','phase4b-s1']


def test_c7_process_preflight_detects_other_phase4b_checkout(monkeypatch):
    import os
    import psutil
    from types import SimpleNamespace
    process=SimpleNamespace(pid=os.getpid()+100000,info={'cmdline':['python','scripts/phase4b_campaign.py','worker']})
    monkeypatch.setattr(psutil,'Process',lambda:SimpleNamespace(parents=lambda:[]))
    monkeypatch.setattr(psutil,'process_iter',lambda *a:[process])
    with pytest.raises(ValueError,match='conflicting process'):policy.assert_no_conflicts()


def test_c7_runtime_file_is_fsynced_before_atomic_publish(tmp_path,monkeypatch):
    from scripts import phase4b_campaign as driver
    layout=policy.PhaseLayout('phase4b',root=tmp_path,approvals_base=tmp_path/'approvals')
    events=[];replace=campaign.os.replace
    monkeypatch.setattr(campaign.os,'fsync',lambda fd:events.append('fsync'))
    def publish(source,destination):
        events.append('publish');return replace(source,destination)
    monkeypatch.setattr(campaign.os,'replace',publish)
    driver.write_json(layout,layout.checkpoints/'runtime/study.json',{'phase4_runtime_stop':{'reason':'memory'}})
    assert events==['fsync','publish']


@pytest.mark.parametrize('exclusive',[False,True])
def test_c9_nonfinite_diagnostic_can_be_persisted(tmp_path,exclusive):
    from scripts import phase4b_campaign as driver
    layout=policy.PhaseLayout('phase4b-s1',root=tmp_path,approvals_base=tmp_path/'approvals')
    path=driver.write_json(layout,layout.output/'s1/replay.json',{'net_expectancy_r':float('nan')},exclusive=exclusive)
    assert json.loads(path.read_text())=={'net_expectancy_r':None,'net_expectancy_r_status':'non_finite'}

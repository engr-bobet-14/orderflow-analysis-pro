"""F1–F5: pinned origin, executable calibration, independent arm reporting."""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts import phase4_approvals as approvals, phase4_campaign as campaign
from scripts import phase4_policy as policy, phase4_studies as p4, phase4_report as report
from scripts import phase4b_campaign as driver, optimize_orderflow as opt
from tests.test_phase4b_operations import fixture_inputs, sha, intervals, benchmark_lane


PIN='b54076c64820ed75cfbaff35a311b1ffecd294944c4f5c1d1189c8c60df32434'


def pinned_inputs(tmp_path,monkeypatch):
    values=fixture_inputs(tmp_path)
    origin=values[-1]
    Path(origin['manifest_path']+'.sha256').write_text(origin['sha256']+'\n')
    monkeypatch.setattr(approvals,'PHASE4_CAMPAIGN_SHA256',origin['sha256'],raising=False)
    return values


def passing_report(bt,digest):
    return {'G4b_N':True,'gates':{str(i):True for i in range(1,11)},'replay_digest':'d',
        'campaign_sha256':digest,'bar_type':bt}


def checkpoint(layout,digest,stage,status='complete'):
    path=layout.checkpoints/digest/(stage+'.json' if status=='complete' else stage+'.stopped.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(policy.canonical_bytes({'stage':stage,'status':status}))


def test_f1_production_pin_is_full_canonical_campaign_hash():
    assert getattr(approvals,'PHASE4_CAMPAIGN_SHA256',None)==PIN


@pytest.mark.parametrize('mutation',['substitution','wrong_seal','missing_seal','manifest_link','seal_link'])
@pytest.mark.parametrize('entry',['registration','allocation','look'])
def test_f1_canonical_origin_refused_on_every_path(tmp_path,monkeypatch,mutation,entry):
    layout,snapshots,comparators,origin=pinned_inputs(tmp_path,monkeypatch)
    bt=driver.BAR_TYPES[0];digest='b'*64;interval=intervals()[bt]
    checkpoint(layout,digest,'S2t')
    approvals.install_holdout_allocation(layout,interval,digest,origin)
    manifest=Path(origin['manifest_path']);seal=Path(str(manifest)+'.sha256')
    if mutation=='substitution':
        manifest.write_bytes(policy.canonical_bytes({'phase':'phase4','substituted':True}))
        origin['sha256']=sha(manifest);seal.write_text(origin['sha256']+'\n')
    elif mutation=='wrong_seal':seal.write_text('a'*64+'\n')
    elif mutation=='missing_seal':seal.unlink()
    else:
        path=manifest if mutation=='manifest_link' else seal
        target=tmp_path/(mutation+'.target');target.write_bytes(path.read_bytes())
        path.unlink();path.symlink_to(target)
    assert approvals.phase4_exposure_state(origin,bt)=='unavailable'
    with pytest.raises(ValueError,match='origin|exposure|allocation'):
        if entry=='registration':
            s1=policy.PhaseLayout('phase4b-s1',root=tmp_path,approvals_base=tmp_path/'approvals')
            campaign.register(s1.manifest,snapshots,phase=s1.phase,layout=s1,
                comparators=comparators,phase4_origin=origin)
        elif entry=='allocation':approvals.install_holdout_allocation(layout,interval,digest,origin)
        else:p4.phase4b_reserved_look(layout,{'phase':'phase4b','phase4_origin':origin},digest,
            interval,{'replay_digest':'d'},passing_report(bt,digest),lambda:pytest.fail('holdout opened'))


def test_f1_pinned_fixture_accepted(tmp_path,monkeypatch):
    layout,snapshots,comparators,origin=pinned_inputs(tmp_path,monkeypatch)
    bt=driver.BAR_TYPES[0];digest='b'*64;checkpoint(layout,digest,'S2t')
    assert approvals.phase4_exposure_state(origin,bt)=='verified_absent'
    approvals.install_holdout_allocation(layout,intervals()[bt],digest,origin)
    s1=policy.PhaseLayout('phase4b-s1',root=tmp_path,approvals_base=tmp_path/'approvals')
    campaign.register(s1.manifest,snapshots,phase=s1.phase,layout=s1,comparators=comparators,phase4_origin=origin)
    assert p4.phase4b_reserved_look(layout,{'phase':'phase4b','phase4_origin':origin},digest,
        intervals()[bt],{'replay_digest':'d'},passing_report(bt,digest),lambda:{'synthetic':True})=={'synthetic':True}


def calibrated(tmp_path,monkeypatch):
    layout,snapshots,comparators,origin=pinned_inputs(tmp_path,monkeypatch)
    state={'base':'base','sha256':'source','dirty':True}
    for module in (campaign,driver):monkeypatch.setattr(module,'source_state',lambda:state)
    monkeypatch.setattr(driver,'preflight',lambda *a:None)
    monkeypatch.setattr(driver,'require_phase4b_approval',lambda *a,**kw:{'sha256':'approval','gates_sha256':'gates'})
    s1=policy.PhaseLayout('phase4b-s1',root=tmp_path,approvals_base=tmp_path/'approvals')
    campaign.register(s1.manifest,snapshots,phase=s1.phase,layout=s1,comparators=comparators,phase4_origin=origin)
    completed=driver.run_stage(s1,'S1','s1-approval',operation=lambda *a:{
        'lanes':{bt:benchmark_lane() for bt in driver.BAR_TYPES}})
    def register(**kwargs):
        return campaign.register(layout.manifest,snapshots,phase=layout.phase,layout=layout,
            comparators=comparators,phase4_origin=origin,benchmark=layout.output/'s1/benchmark.json',
            holdout_intervals=intervals(),**kwargs)
    return layout,completed,register


@pytest.mark.parametrize('change',['module','defaults','version','source','frozen_builder','search_space'])
def test_f2_stale_runtime_calibration_refused(tmp_path,monkeypatch,change):
    from scripts import phase4b_runtime as runtime
    # An imported strategy module may change on disk without being reloaded in memory.
    module=ModuleType('orderflow_system.patterns._fingerprint_fixture')
    module.__file__=str(tmp_path/'strategy.py');Path(module.__file__).write_text('before = 1\n')
    monkeypatch.setitem(sys.modules,module.__name__,module)
    layout,completed,register=calibrated(tmp_path,monkeypatch)
    assert completed['runtime_fingerprint']==completed['result']['runtime_fingerprint']
    if change=='module':Path(module.__file__).write_text('after = 2\n')
    elif change=='defaults':
        original=opt.instrument_config
        monkeypatch.setattr(opt,'instrument_config',lambda symbol:replace(original(symbol),
            risk=replace(original(symbol).risk,min_composite_score=17.)))
    elif change=='version':monkeypatch.setattr(runtime.metadata,'version',lambda name:'changed-version')
    elif change=='source':monkeypatch.setattr(runtime,'source_state',lambda:{'base':'base','sha256':'changed','dirty':True})
    elif change=='search_space':monkeypatch.setitem(p4.FULL_STRATEGY_SPACE,'ts_stacked_min_levels',(1,3,'int'))
    else:
        original=Path.read_bytes
        monkeypatch.setattr(Path,'read_bytes',lambda path:original(path)+(b'changed' if str(path).endswith(
            'orderflow_system/data/trade_alignment.py') else b''))
    with pytest.raises(ValueError,match='runtime fingerprint|stale calibration'):register()
    assert not layout.manifest.exists()


def test_f2_identical_runtime_recorded_and_admitted(tmp_path,monkeypatch):
    from scripts import phase4b_runtime as runtime
    layout,completed,register=calibrated(tmp_path,monkeypatch)
    manifest=register()
    fingerprint=manifest['runtime_fingerprint']
    assert fingerprint==completed['runtime_fingerprint']==runtime.runtime_fingerprint()
    assert set(fingerprint['versions'])=={'python','numpy','scipy','pandas','optuna','RiskLabAI'}
    assert 'scripts/phase4_studies.py' in fingerprint['module_sha256']
    assert 'orderflow_system/patterns/initiative.py' in fingerprint['module_sha256']
    assert len(fingerprint['frozen_builder_sha256'])==4
    configuration=json.loads(fingerprint['configuration'])
    assert configuration['search_space']['ts_stacked_min_levels']==[1,4,'int']
    assert configuration['trade_flow_defaults']['footprint_bucket']==1.
    assert driver.run_stage(layout,'S2t','s2-approval',operation=lambda *a:{'synthetic':True})['status']=='complete'


def test_f2_fingerprint_stable_across_registration_and_worker_processes():
    import subprocess
    fingerprints=[]
    for module in ('scripts.phase4_campaign','scripts.phase4b_campaign'):
        code=('import '+module+'; from scripts.phase4b_runtime import runtime_fingerprint; '
            'from scripts.phase4_policy import canonical_bytes; import hashlib; '
            'print(hashlib.sha256(canonical_bytes(runtime_fingerprint())).hexdigest())')
        fingerprints.append(subprocess.check_output([sys.executable,'-c',code],text=True).strip())
    assert fingerprints[0]==fingerprints[1]


@pytest.mark.parametrize('stage',['S2t','S2v','S3'])
def test_f2_runtime_change_refused_before_stage_admission(tmp_path,monkeypatch,stage):
    from scripts import phase4b_runtime as runtime
    layout,_,register=calibrated(tmp_path,monkeypatch);register()
    if stage=='S3':checkpoint(layout,sha(layout.manifest),'S2t')
    monkeypatch.setattr(runtime.metadata,'version',lambda name:'changed-version')
    with pytest.raises(ValueError,match='runtime fingerprint'):
        driver.run_stage(layout,stage,'unused',operation=lambda *a:pytest.fail('stale runtime executed'))
    assert not list(layout.checkpoints.glob('*/attempts/unused.json'))


def test_f3_s2_stage_allowance_is_registered_and_hand_limits_checked(tmp_path,monkeypatch):
    layout,_,register=calibrated(tmp_path,monkeypatch)
    derived=campaign.derive_ceilings({'lanes':{bt:benchmark_lane() for bt in driver.BAR_TYPES}})
    for bt,stage in zip(driver.BAR_TYPES,('S2t','S2v')):
        assert derived['max_stage_seconds'][stage]==derived['arms'][bt]['max_study_seconds']+1200
    with pytest.raises(ValueError,match='limit'):
        register(stage_limits=[('S2t',derived['arms'][driver.BAR_TYPES[0]]['max_study_seconds'])])
    manifest=register(stage_limits=list(derived['max_stage_seconds'].items()))
    assert manifest['max_stage_seconds']==derived['max_stage_seconds']
    assert manifest['runtime_ceilings']['inputs']['S2_overhead_seconds']==1200


@pytest.mark.parametrize('stage',['S2t','S2v'])
def test_f3_live_stage_clock_keeps_separate_study_and_trial_limits(tmp_path,monkeypatch,stage):
    layout,_,register=calibrated(tmp_path,monkeypatch);manifest=register();digest=sha(layout.manifest)
    seen=[]
    @contextmanager
    def timing(layout,digest,label,ceiling,**kwargs):
        seen.append((label,ceiling));yield
    monkeypatch.setattr(driver,'file_timing',timing)
    monkeypatch.setattr(driver,'protected_hashes',lambda *a:{})
    watched=[]
    @contextmanager
    def watchdog(layout,manifest,digest,stage):
        watched.append(driver.stage_ceiling(manifest,stage));yield
    monkeypatch.setattr(driver,'stage_watchdog',watchdog)
    monkeypatch.setattr(driver,'execute_stage',lambda *a:{'synthetic':True})
    monkeypatch.setenv(driver.LAUNCH_ENV,json.dumps({'utc':'u','monotonic':driver._now(),'attempt':'production'}))
    monkeypatch.setattr(driver,'require_inherited_locks',lambda *a:None)
    driver.run_stage(layout,stage,'s2-approval',finalize=False)
    bt=driver.BAR_TYPES[stage=='S2v'];arm=manifest['arms'][f'BTCUSDT:{bt}:full']
    # rev 13: the S2 stage allowance is enforced from launch L by stage_watchdog; file_timing has no ceiling.
    assert watched==[arm['max_study_seconds']+1200] and seen==[(stage,None)]
    class Study:
        user_attrs={}
        trials=[]
    study=Study();captured=[]
    monkeypatch.setattr(opt,'study_identity',lambda args,meta:('synthetic',{}))
    monkeypatch.setattr(opt,'open_study',lambda args,*a:(captured.append(args) or study))
    monkeypatch.setattr(opt,'reconcile_phase4_trials',lambda *a:None)
    monkeypatch.setattr(opt,'optimize_loop',lambda *a:None)
    monkeypatch.setattr(opt,'complete_trials',lambda *a:[None]*300)
    driver.run_study(layout,{**manifest,'_approval_sha256':'approval','_gates_sha256':'gates'},digest,bt,
        SimpleNamespace(trade_updates=[],input_paths={'trade':tmp_path/'synthetic'},trade_cache_identity={},metadata={}),[])
    assert captured[0].max_study_seconds==arm['max_study_seconds']
    assert captured[0].max_trial_seconds==arm['max_trial_seconds']


@pytest.mark.parametrize('stage',['S2t','S2v'])
def test_f3_external_timeout_uses_s2_stage_allowance(tmp_path,monkeypatch,stage):
    from scripts import measure_rss,phase4_resources
    layout,_,register=calibrated(tmp_path,monkeypatch);manifest=register()
    monkeypatch.setattr(driver,'PhaseLayout',lambda *a:layout)
    monkeypatch.setattr(driver.shutil,'which',lambda *a:'/synthetic/timeout')
    layout.approval_root.mkdir(parents=True);(layout.approval_root/('a'*64+'.json')).write_text('{}')
    monkeypatch.setattr(approvals,'verify_phase4b_approval',lambda *a:None)
    monkeypatch.setattr(phase4_resources,'enforce_memory',lambda *a:None)
    @contextmanager
    def locks(layout,manifest):yield {}
    monkeypatch.setattr(driver,'stage_locks',locks)
    commands=[]
    monkeypatch.setattr(measure_rss,'run_measured',lambda command,**kw:(commands.append(command) or {'returncode':0}))
    driver.main(['run','--phase','phase4b','--stage',stage,'--owner-approved','parent'])
    bt=driver.BAR_TYPES[stage=='S2v']
    assert commands[0][1]=='--foreground' and int(commands[0][4])==manifest['arms'][f'BTCUSDT:{bt}:full']['max_study_seconds']+1200


def reporting_fixture(tmp_path,monkeypatch,statuses=('complete','breached')):
    import optuna
    layout=policy.PhaseLayout('phase4b',root=tmp_path,approvals_base=tmp_path/'approvals');digest='c'*64
    snapshots=[]
    for index in range(4):
        path=tmp_path/f'historical{index}.db'
        study=optuna.create_study(storage='sqlite:///'+str(path),study_name=f'legacy{index}')
        study.add_trial(optuna.trial.create_trial(value=0.,user_attrs={'replay_digest':f'legacy{index}'}))
        snapshots.append({'path':str(path),'sha256':sha(path)})
    layout.studies_db.parent.mkdir(parents=True)
    arms={}
    for index,(bt,status) in enumerate(zip(driver.BAR_TYPES,statuses)):
        arm=f'BTCUSDT:{bt}:full';arms[arm]={'budget':2}
        study=optuna.create_study(storage='sqlite:///'+str(layout.studies_db),study_name=bt)
        study.set_user_attr('identity',{'phase':'phase4b','arm':arm,'campaign_sha256':digest})
        for number in range(2 if status=='complete' else 1):
            study.add_trial(optuna.trial.create_trial(value=float(number),user_attrs={'replay_digest':f'{bt}-{number}'}))
        if status!='complete':study.set_user_attr('phase4_runtime_stop',{'reason':'max_trial_seconds'})
        checkpoint(layout,digest,'S2t' if index==0 else 'S2v',status)
    manifest={'phase':'phase4b','phase4b_db':str(layout.studies_db),'snapshots':snapshots,'arms':arms,
        'phase4_origin':{},'holdout_intervals':intervals(),'stages':['S3'],
        'max_stage_seconds':{'S3':10000},'runtime_ceilings':{'inputs':{}}}
    return layout,manifest,digest


def test_f4_independent_report_keeps_stopped_trials_in_multiplicity(tmp_path,monkeypatch):
    layout,manifest,digest=reporting_fixture(tmp_path,monkeypatch)
    seen=[]
    monkeypatch.setattr(driver,'open_prefix',lambda layout,manifest,bt:(seen.append(bt) or (object(),object(),[])))
    def build(prefix,trials,comparator,snapshots,folds,**kwargs):
        union=report.campaign_union(snapshots)
        assert union['n_symbol_all_studies']==7  # 4 historical + 2 tick + 1 stopped volume.
        assert len(trials)==2
        return {'G4b_N':False,'gates':{str(i):False for i in range(1,11)},'multiplicity':union}
    monkeypatch.setattr(report,'build_full_strategy_report',build)
    monkeypatch.setattr(p4,'phase4b_reserved_look',lambda *a:pytest.fail('failed/stopped arm looked at'))
    result=driver.execute_stage('S3',manifest,digest,layout)
    assert seen==[driver.BAR_TYPES[0]]
    assert set(result['reports'])=={driver.BAR_TYPES[0]}
    assert result['not_reported']=={driver.BAR_TYPES[1]:'S2 breached'}
    assert result['looks']=={}


def test_f4_both_stopped_refuses_s3_before_consumption(tmp_path,monkeypatch):
    layout,manifest,digest=reporting_fixture(tmp_path,monkeypatch,('breached','stopped'))
    monkeypatch.setattr(driver,'read_manifest',lambda *a:(manifest,digest))
    monkeypatch.setattr(driver,'preflight',lambda *a:None)
    consumed=[]
    monkeypatch.setattr(driver,'require_phase4b_approval',lambda *a,**kw:(consumed.append(kw['consume']) or {}))
    with pytest.raises(ValueError,match='at least one|completed S2'):
        driver.run_stage(layout,'S3','approval',operation=lambda *a:pytest.fail('no completed S2'))
    assert True not in consumed


def test_f4_s3_admits_completed_arm_after_other_arm_timeout(tmp_path,monkeypatch):
    layout,_,register=calibrated(tmp_path,monkeypatch);manifest=register();digest=sha(layout.manifest)
    checkpoint(layout,digest,'S2t');checkpoint(layout,digest,'S2v')
    # Even a stale complete checkpoint cannot overrule a durable ceiling-breach audit.
    driver.record_termination(layout,manifest,digest,'S2v',stop={'reason':'max_study_seconds',
        'measured_seconds':80000.,'limit_seconds':75900.})
    completed,omitted=driver.completed_s2(layout,digest)
    assert completed==[driver.BAR_TYPES[0]] and omitted=={driver.BAR_TYPES[1]:'S2 breached'}
    assert driver.run_stage(layout,'S3','s3-approval',operation=lambda *a:{
        'reports':{driver.BAR_TYPES[0]:{'G4b_N':False}},'not_reported':omitted,'looks':{}})['status']=='complete'


def test_f4_stopped_arm_refused_even_with_fabricated_passing_report(tmp_path,monkeypatch):
    layout,manifest,digest=reporting_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(approvals,'require_holdout_allocation',lambda *a:None)
    bt=driver.BAR_TYPES[1]
    with pytest.raises(ValueError,match='completed S2|S2.*complete'):
        p4.phase4b_reserved_look(layout,manifest,digest,intervals()[bt],{'replay_digest':'d'},
            passing_report(bt,digest),lambda:pytest.fail('stopped arm looked at'))


@pytest.mark.parametrize('mutation',['incomplete','running','waiting','stopped','historical_tamper'])
def test_f4_reported_arm_snapshot_checks_still_fail(tmp_path,monkeypatch,mutation):
    import sqlite3
    layout,manifest,digest=reporting_fixture(tmp_path,monkeypatch)
    if mutation=='historical_tamper':Path(manifest['snapshots'][0]['path']).write_bytes(b'tampered')
    else:
        with sqlite3.connect(layout.studies_db) as con:
            sid=con.execute('SELECT study_id FROM studies WHERE study_name=?',(driver.BAR_TYPES[0],)).fetchone()[0]
            if mutation=='stopped':
                con.execute('INSERT INTO study_user_attributes(study_id,key,value_json) VALUES(?,?,?)',
                    (sid,'phase4_runtime_stop','{}'))
            else:
                state={'incomplete':'FAIL','running':'RUNNING','waiting':'WAITING'}[mutation]
                con.execute('UPDATE trials SET state=? WHERE study_id=? AND number=1',(state,sid))
    with pytest.raises(ValueError,match='snapshot|budget|unstopped'):
        report.phase4b_report_snapshots(layout,manifest,digest,reported_arms={f'BTCUSDT:{driver.BAR_TYPES[0]}:full'})


def test_f4_existing_gate_fail_still_fails():
    from tests.test_phase4b_statistics import passing_gate_args
    values=passing_gate_args();values['stress']=-1.
    result=report.net_profitability_gates(**values)
    assert result['G4b_N'] is False and result['gates']['9'] is False


def test_f5_script_namespace_has_phase4b_functions_before_main(monkeypatch,capsys):
    expected={name for name,value in vars(report).items() if inspect.isfunction(value)
        and value.__module__==report.__name__}
    observed={}
    # Stop just after parsing: this observes the namespace at actual script entry, without I/O.
    def stop(*args,**kwargs):
        observed.update(inspect.currentframe().f_back.f_globals)
        raise RuntimeError('synthetic script stop')
    monkeypatch.setattr(report.argparse.ArgumentParser,'parse_args',stop)
    with pytest.raises(RuntimeError,match='synthetic script stop'):
        runpy.run_path('scripts/phase4_report.py',run_name='__main__')
    assert expected<=observed.keys()


def test_rev12_preserved_and_rev13_bound():
    assert sha(Path('docs/plans/phase4b-full-strategy-plan-rev12.md'))=='dbb407fb611720e42d70ac6c051d82077350247ef83fb433f13e9d97d8849099'
    expected='75b22452c0b168f43fd91704a7aee23a4cb1c84c637f4a8f775b87be13857b7f'
    assert approvals.PHASE4B_PLAN_SHA256==expected
    assert sha(approvals.PHASE4B_PLAN_PATH)==expected

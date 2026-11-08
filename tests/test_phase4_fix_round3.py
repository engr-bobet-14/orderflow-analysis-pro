"""Owner-authorized release readiness regressions; red evidence under fix-round3."""
import importlib
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import trade_feature_cache as tfc, phase4_policy as policy, phase4_studies as p4
from scripts import optimize_orderflow as opt, phase4_approvals as approvals
from tests.test_phase4_cache import paired,BT

GiB=1024**3
ALL=sorted(b.value for b in tfc.BarType)


def production_unit(tmp_path,monkeypatch,two=False):
    if two:
        from tests.paired_fixture import make_trades,write_paired
        from orderflow_system.data.paired_dataset import PairedDataset
        directory,root=write_paired(tmp_path,make_trades(days=3,per_day=80),{b:[4,41,90,160,200] for b in ALL})
        ds=PairedDataset.open(directory,root)
    else:
        return production_unit(tmp_path,monkeypatch,two=True)
    monkeypatch.setattr(tfc,'open_registered',lambda *a:ds)
    monkeypatch.setattr(tfc,'pinned_root',lambda _:ds.root_hash)
    monkeypatch.setattr(tfc,'checked_write',lambda p:Path(p))
    monkeypatch.setattr(tfc,'validate_build_range',lambda *a:'production')
    state=policy.source_state()|{'dirty':False}
    monkeypatch.setattr(tfc,'require_source_mode',lambda *a:state)
    monkeypatch.setattr(tfc,'verify_source',lambda *a:None)
    monkeypatch.setattr(tfc,'protected_state',lambda:{'owner':'same'},raising=False)
    monkeypatch.setattr(tfc,'production_preflight',lambda *a,**kw:{'protected_state':{'owner':'same'},'source_state':state},raising=False)
    monkeypatch.setattr(approvals,'verify_approval',lambda *a,**kw:{'sha256':'unit','record':{}})
    return ds,state


def test_r3_1_preflight_refusal_leaves_destination_and_approval_untouched(tmp_path,monkeypatch):
    ds,state=production_unit(tmp_path,monkeypatch)
    consumed=[];dest=tmp_path/'destination';dest.mkdir();(dest/'keep').write_bytes(b'old')
    before={p.name:p.read_bytes() for p in dest.iterdir()}
    monkeypatch.setattr(approvals,'consume_approval',lambda *a:consumed.append('used'))
    monkeypatch.setattr(tfc,'production_preflight',lambda *a,**kw:(_ for _ in ()).throw(ValueError('preflight refused')),raising=False)
    with pytest.raises(ValueError,match='preflight refused'):
        tfc.build(ds,dest,full_history=True,owner_approved='unit')
    assert consumed==[] and {p.name:p.read_bytes() for p in dest.iterdir()}==before


@pytest.mark.parametrize('failure',['mount','paired','disk','conflict','enumeration','protected','source','memory'])
def test_r3_1_each_production_preflight_check(tmp_path,monkeypatch,failure):
    ds,state=production_unit(tmp_path,monkeypatch)
    monkeypatch.setattr(tfc,'production_preflight',policy.production_preflight)
    used=[];monkeypatch.setattr(approvals,'consume_approval',lambda *a:used.append(1))
    monkeypatch.setattr(policy,'mount_ready',lambda:failure!='mount',raising=False)
    monkeypatch.setattr(policy,'paired_readable',lambda ds:failure!='paired',raising=False)
    monkeypatch.setattr(policy,'destination_free_bytes',lambda p:50*GiB if failure=='disk' else 200*GiB,raising=False)
    def conflicts(**kw):
        if failure=='enumeration':raise ValueError('process enumeration unavailable')
        if failure=='conflict':raise ValueError('conflicting process unit')
        return {'status':'clear'}
    monkeypatch.setattr(policy,'assert_no_conflicts',conflicts)
    monkeypatch.setattr(policy,'protected_state',lambda:{'a':'wrong'} if failure=='protected' else {'a':'b'})
    monkeypatch.setattr(policy,'owner_protected_state',lambda:{'a':'b'},raising=False)
    monkeypatch.setattr(policy,'source_state',lambda *a:state|({'sha256':'changed'} if failure=='source' else {}))
    from scripts import phase4_resources as resources
    def memory(*a):
        if failure=='memory':raise ValueError('memory admission refused')
        return {'pass':True}
    monkeypatch.setattr(resources,'require_memory_plan',memory)
    match={'mount':'mount','paired':'paired','disk':'disk','conflict':'conflicting','enumeration':'enumeration unavailable',
           'protected':'protected','source':'source_state','memory':'memory'}[failure]
    with pytest.raises(ValueError,match=match):
        tfc.build(ds,tmp_path/'new',full_history=True,owner_approved='unit')
    assert used==[] and not (tmp_path/'new').exists()


def test_r3_1_changed_protected_set_never_publishes(tmp_path,monkeypatch):
    ds,_=production_unit(tmp_path,monkeypatch);calls=[]
    monkeypatch.setattr(approvals,'consume_approval',lambda *a:calls.append('used'))
    monkeypatch.setattr(tfc,'protected_state',lambda:{'owner':'changed'},raising=False)
    dest=tmp_path/'out'
    with pytest.raises(ValueError,match='protected'):
        tfc.build(ds,dest,full_history=True,owner_approved='unit')
    assert calls==['used'] and not (dest/'COMPLETE.json').exists()


def test_r3_2_reconcile_preserves_complete_and_audits_fail(tmp_path):
    import optuna
    from optuna.trial import TrialState,create_trial
    study=optuna.create_study(storage='sqlite:///'+str(tmp_path/'unit.db'),study_name='unit')
    study.add_trial(create_trial(value=2.,user_attrs={'keep':{'literal':'bytes'}}))
    pending=study.ask();pending.set_user_attr('phase4_worker',{'pid':123,'host':'unit'})
    before=study.get_trials()[0]
    opt.reconcile_phase4_trials(study,SimpleNamespace(phase4=True))
    after=study.get_trials()
    assert after[0]==before and after[1].state==TrialState.FAIL
    record=after[1].user_attrs['phase4_reconciled']
    assert record['from_state']=='RUNNING' and record['reason']=='orphan after interruption'
    assert record['pid']==123 and record['host']=='unit'
    assert len(study.user_attrs['phase4_reconciliation_audit'])==1
    opt.reconcile_phase4_trials(study,SimpleNamespace(phase4=True))
    assert len(study.user_attrs['phase4_reconciliation_audit'])==1
    legacy=study.ask()
    opt.reconcile_phase4_trials(study,SimpleNamespace(phase4=False))
    assert study.get_trials()[-1].state==TrialState.RUNNING


def test_r3_3_rename_failure_retry_uses_only_complete_generation(tmp_path,monkeypatch):
    ds,_=production_unit(tmp_path,monkeypatch,two=True)
    monkeypatch.setattr(approvals,'consume_approval',lambda *a:None)
    original=Path.rename;count=[];dest=tmp_path/'out'
    def fail_second(self,target):
        count.append(target)
        if len(count)==2:raise OSError('injected rename failure')
        return original(self,target)
    monkeypatch.setattr(Path,'rename',fail_second)
    with pytest.raises(OSError):tfc.build(ds,dest,full_history=True,owner_approved='unit')
    orphan=next(dest.glob('*/manifest.json')).parent
    monkeypatch.setattr(Path,'rename',original)
    paths,result=tfc.build(ds,dest,full_history=True,owner_approved='different-unit')
    generation=tfc.production_generation(dest)
    assert set(generation['lanes'])==set(ALL) and str(orphan) in generation['orphans']
    assert generation['lanes']=={k:str(v) for k,v in paths.items()}
    # Scanning orphan paths is allowed; opening their manifests or arrays is not.
    import builtins,io
    def intercept(original_open):
        def checked(path,*a,**kw):
            if not isinstance(path,int):assert not Path(path).is_relative_to(orphan)
            return original_open(path,*a,**kw)
        return checked
    for module in (builtins,io):monkeypatch.setattr(module,'open',intercept(module.open))
    tfc.production_generation(dest)
    for path in paths.values():tfc.CachedTradeUpdates(path,production=True)
    verified=tfc.main(['verify','--full-history','--owner-approved','verify-unit','--out',str(dest),'--dataset-id',ds.dataset_id])
    assert len(verified['lanes'])==6 and str(orphan) in verified['orphans']
    with pytest.raises(ValueError,match='COMPLETE|complete generation'):
        tfc.build(ds,dest,full_history=True,owner_approved='third-unit')


def test_r3_4_round_is_not_called_in_lazy_reader():
    from test_pipeline import synth_bars
    bar=synth_bars(1)[0]
    import inspect
    assert 'round(' not in inspect.getsource(tfc._CachedFootprint.levels.fget)
    arrays=(np.array([1,10001,20_000_000,2**40],dtype='i8'),np.array([3,7,11,19]),np.array([5,9,13,17]))
    from orderflow_system.analytics.footprint import FootprintEngine
    engine=FootprintEngine()
    eager=engine.build_from_levels(bar,arrays)
    lazy=tfc._CachedFootprint(bar,tfc.CachedFootprintLevels(arrays))
    assert list(lazy.levels)==list(eager.levels) and lazy.levels==eager.levels


def test_r3_5_memory_bounds_and_total_budget():
    r=importlib.import_module('scripts.phase4_resources')
    assert not r.memory_bounds({'rss_bytes':3*GiB,'private_bytes':5*GiB},2*GiB)['pass']
    assert not r.memory_bounds({'rss_bytes':7*GiB,'private_bytes':3*GiB},2*GiB)['pass']
    assert not r.memory_bounds({'rss_bytes':5*GiB,'private_bytes':3*GiB},2*GiB)['pass']
    baseline=r.memory_bounds({'rss_bytes':5*GiB,'private_bytes':3*GiB},0)
    assert not baseline['pass'] and baseline['private_limit_bytes']==4*GiB
    plan=[{'private_budget_bytes':4*GiB,'mapped_files':{'same-cache':2*GiB}}]*2
    value=r.memory_plan(plan,available_bytes=8*GiB)
    assert value['planned_bytes']==10*GiB and value['unique_mapped_bytes']==2*GiB and value['pass']
    assert not r.memory_plan(plan*3,available_bytes=32*GiB)['pass']
    assert not r.memory_plan(plan,available_bytes=7*GiB)['pass']


def test_r3_5_vm_stat_formula():
    r=importlib.import_module('scripts.phase4_resources')
    text='Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages inactive: 20.\nPages speculative: 30.\nPages active: 999.\n'
    assert r.parse_vm_stat(text)==60*16384


def test_r3_2_sigkill_resume_exact_budget_and_concurrent_refusal(tmp_path,monkeypatch):
    import subprocess,sys,time,signal,sqlite3,hashlib
    code=r'''
import json,sys,time
from pathlib import Path
from scripts import optimize_orderflow as o,phase4_studies as p4
import os
p4.campaign_arm=lambda *a,**kw:None
if os.environ.get('P4_TEST_DISABLE_ORPHAN_RECONCILE'):o.reconcile_phase4_trials=lambda *a:None
from test_pipeline import synth_bars
root=Path(sys.argv[1]);stage=sys.argv[2]
a=o.build_parser().parse_args(['--phase4','--no-warm-start','--n-trials','3','--n-jobs','1'])
a.symbol='BTCUSDT';a.bar_type='fixed_dollar_imbalance';a.study_db=root/'unit.db';a.output=root/'results.json'
a.cache_dir=root/'cache';a.space=dict(o.SEARCH_SPACE_BOUNDS);a.approval_sha256='synthetic';a.gates_sha256='synthetic'
binfile=root/'bars.bin'
if not binfile.exists():o.write_bar_stream(binfile,synth_bars(100,step_ms=86400000))
def objective(t,*unused):
    t.suggest_float('probe',.1,.9)
    if (stage=='kill' and t.number==1) or (stage=='resume' and t.number==2):
        (root/(stage+'.ready')).write_text(str(t.number))
        while not (root/(stage+'.release')).exists():time.sleep(.02)
    t.set_user_attr('params',{'price_proximity_pct':.002});t.set_user_attr('feasible',True)
    t.set_user_attr('n_scored_bars',10);t.set_user_attr('total_trades',20);t.set_user_attr('replay_digest',str(t.number)*64)
    return 100.-t.number
o.objective=objective
o.optimize_study(a,binfile,{'fingerprint':'f'*64})
'''
    env=os.environ.copy();env['PYTHONPATH']=str(Path.cwd())+os.pathsep+str(Path.cwd()/'tests')
    def launch(stage):return subprocess.Popen([sys.executable,'-c',code,str(tmp_path),stage],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    def ready(proc,stage):
        deadline=time.monotonic()+25
        while not (tmp_path/(stage+'.ready')).exists():
            if proc.poll() is not None:raise AssertionError(proc.communicate())
            if time.monotonic()>deadline:raise AssertionError('trial did not start')
            time.sleep(.03)
    def complete_rows():
        with sqlite3.connect(tmp_path/'unit.db') as con:
            return {'trials':con.execute("SELECT * FROM trials WHERE state='COMPLETE' ORDER BY trial_id").fetchall(),
                **{table:con.execute(f"SELECT * FROM {table} WHERE trial_id IN (SELECT trial_id FROM trials WHERE state='COMPLETE') ORDER BY trial_id").fetchall()
                   for table in ('trial_values','trial_params','trial_user_attributes')}}
    first=launch('kill');resume=None;other=None
    try:
        ready(first,'kill');before=complete_rows();assert len(before['trials'])==1
        first.send_signal(signal.SIGKILL);first.communicate(timeout=10)
        resume=launch('resume');ready(resume,'resume')
        other=launch('other');stdout,stderr=other.communicate(timeout=15)
        assert other.returncode!=0 and 'Same study already running' in stderr
        (tmp_path/'resume.release').touch()
        stdout,stderr=resume.communicate(timeout=25);assert resume.returncode==0,(stdout,stderr)
        result=json.loads((tmp_path/'results.json').read_text())
        # Compare original COMPLETE database rows byte-for-byte, including serialized attrs.
        after=complete_rows()
        for table,rows in before.items():assert all(row in after[table] for row in rows)
        names=opt.list_study_names(tmp_path/'unit.db');assert len(names)==1
        trials=opt.read_study_trials(tmp_path/'unit.db',names[0])
        counts={state:sum(t['state']==state for t in trials) for state in ('COMPLETE','FAIL','RUNNING','WAITING')}
        assert counts=={'COMPLETE':3,'FAIL':1,'RUNNING':0,'WAITING':0}
        failed=next(t for t in trials if t['state']=='FAIL')
        assert failed['attrs']['phase4_reconciled']['from_state']=='RUNNING'
        best,identity,reservation=p4.completed_candidate(tmp_path/'unit.db',names[0],0,3)
        assert reservation['failed']==reservation['reconciled']==1
        assert opt.read_study_attrs(tmp_path/'unit.db',names[0])['phase4_reconciliation_audit']
        from scripts import phase4_report as report
        import sqlite3
        monkeypatch.setattr(policy,'ROOT',tmp_path)
        target=tmp_path/'output/optimize/phase4/studies.db';target.parent.mkdir(parents=True)
        with sqlite3.connect((tmp_path/'unit.db').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(target) as copy:
            source.backup(copy)
        legacy=[]
        for i in range(3):
            path=tmp_path/f'legacy-{i}';path.write_bytes(bytes([i]))
            legacy.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        campaign={'phase4_db':str(target),'snapshots':legacy,'arms':{identity['arm']:{'kind':'study','budget':3}}}
        snapshots=report.report_snapshots(campaign,None,target)
        assert snapshots[-1]['trial_recovery'][names[0]]=={'failed':1,'reconciled':1}
        exposure=tmp_path/'output/phase4/holdout/exposures.json'
        got=p4.reserved_look(exposure,identity['arm'],reservation,lambda:{'synthetic_admission':True})
        assert got['synthetic_admission'] is True
        stored=json.loads(exposure.read_text())[identity['arm']]
        assert stored['candidate']['failed']==stored['candidate']['reconciled']==1

    finally:
        for proc in (first,resume,other):
            if proc is not None and proc.poll() is None:proc.kill();proc.communicate(timeout=10)


def test_r3_4_exhaustive_tick_float_bits_and_random_sample():
    count=0
    for start in range(0,20_000_001,500_000):
        k=np.arange(start,min(start+500_000,20_000_001),dtype='i8')
        direct=k/100
        rounded=np.fromiter((round(int(n)/100,10) for n in k),dtype='f8',count=len(k))
        assert np.array_equal(direct.view('u8'),rounded.view('u8'));count+=len(k)
    assert count==20_000_001
    k=np.random.default_rng(20261106).integers(0,2**40,size=100000,dtype='i8')
    assert np.array_equal((k/100).view('u8'),np.fromiter((round(int(n)/100,10) for n in k),dtype='f8').view('u8'))


def test_r3_1_enumeration_denied_is_fail_closed(monkeypatch):
    import psutil
    monkeypatch.setattr(psutil,'process_iter',lambda *a:(_ for _ in ()).throw(psutil.AccessDenied(123)))
    with pytest.raises(ValueError,match='enumeration unavailable'):policy.assert_no_conflicts(strict=True)
    assert policy.assert_no_conflicts(strict=False)['status']=='unavailable'


def test_r3_3_interrupted_capture_consumes_once_but_never_publishes(tmp_path,monkeypatch):
    ds,_=production_unit(tmp_path,monkeypatch);events=[]
    monkeypatch.setattr(tfc,'production_preflight',lambda *a,**kw:events.append('preflight') or {'protected_state':{'owner':'same'}})
    monkeypatch.setattr(approvals,'consume_approval',lambda *a:events.append('consume'))
    monkeypatch.setattr(tfc._Writer,'capture',lambda *a:(_ for _ in ()).throw(RuntimeError('injected interrupted capture')))
    dest=tmp_path/'out'
    with pytest.raises(RuntimeError,match='capture'):tfc.build(ds,dest,full_history=True,owner_approved='unit')
    assert events==['preflight','consume']
    assert not (dest/'COMPLETE.json').exists() and not list(dest.iterdir())
    with pytest.raises(ValueError,match='COMPLETE'):p4.attach_cache(SimpleNamespace(trade_cache=dest,bar_type=BT,dataset_id=ds.dataset_id),None)


@pytest.mark.parametrize('damage',['missing','altered','missing_lane_entry'])
def test_r3_3_marker_damage_refuses_verify_and_attach(tmp_path,monkeypatch,damage,capsys):
    ds,_=production_unit(tmp_path,monkeypatch);dest=tmp_path/'out'
    monkeypatch.setattr(approvals,'consume_approval',lambda *a:None)
    paths,_=tfc.build(ds,dest,full_history=True,owner_approved='unit')
    marker=dest/'COMPLETE.json';entry=paths[BT]/'manifest.json'
    if damage=='missing':entry.unlink()
    elif damage=='altered':entry.write_bytes(entry.read_bytes()+b' ')
    else:
        data=json.loads(marker.read_text());data['manifests'].pop(paths[BT].name+'/manifest.json');marker.write_text(json.dumps(data))
    monkeypatch.setattr(tfc,'open_registered',lambda *a:ds)
    with pytest.raises(ValueError,match='COMPLETE|six-lane'):p4.attach_cache(SimpleNamespace(trade_cache=dest,bar_type=BT,dataset_id=ds.dataset_id),None)
    with pytest.raises(SystemExit) as exc:tfc.main(['verify','--full-history','--owner-approved','unit','--out',str(dest),'--dataset-id',ds.dataset_id])
    assert exc.value.code==2 and 'COMPLETE' in capsys.readouterr().err


def test_r3_1_verify_preflight_fails_without_consuming(tmp_path,monkeypatch):
    ds,_=production_unit(tmp_path,monkeypatch);used=[]
    monkeypatch.setattr(tfc,'generation_directory',lambda *a:tmp_path/'generation')
    monkeypatch.setattr(tfc,'open_registered',lambda *a:ds)
    monkeypatch.setattr(approvals,'consume_approval',lambda *a:used.append(1))
    monkeypatch.setattr(tfc,'production_preflight',lambda *a,**kw:(_ for _ in ()).throw(ValueError('preflight refused')))
    with pytest.raises(SystemExit) as error:tfc.main(['verify','--full-history','--owner-approved','unit','--out',str(tmp_path/'generation')])
    assert error.value.code==2 and used==[] and not (tmp_path/'generation').exists()


def test_r3_5_native_measurement_or_documented_fallback():
    from scripts.phase4_resources import process_memory,memory_bounds
    value=process_memory()
    assert value['rss_bytes']>0
    if not value['private_is_fallback']:assert value['peak_private_bytes']>=value['private_bytes']>0
    else:assert not memory_bounds(value,GiB)['pass']


def test_r3_5_private_peaks_are_measured_in_child():
    import sys
    from scripts.measure_rss import run_measured
    value=run_measured([sys.executable,'-c','import time; x=bytearray(16*1024**2); time.sleep(.15)'],memory=True)
    assert value['returncode']==0 and value['memory_samples']>0
    assert value['peak_private_bytes']>0 and value['peak_rss_bytes']>0


@pytest.mark.parametrize('dtype,upper',[('i4',20_000_001),('i8',2**40)])
def test_r3_4_random_lazy_levels_equal_eager_dict_and_order(dtype,upper):
    from test_pipeline import synth_bars
    from orderflow_system.analytics.footprint import FootprintEngine
    rng=np.random.default_rng(20261106)
    ticks=np.unique(rng.integers(0,upper,size=1000,dtype=dtype));rng.shuffle(ticks)
    arrays=(ticks,rng.integers(0,2**40,size=len(ticks)),rng.integers(0,2**40,size=len(ticks)))
    bar=synth_bars(1)[0]
    eager=FootprintEngine().build_from_levels(bar,arrays).levels
    lazy=tfc._CachedFootprint(bar,tfc.CachedFootprintLevels(arrays)).levels
    assert list(eager)==list(lazy) and eager==lazy
    for a,b in zip(eager,lazy):assert struct.pack('d',a)==struct.pack('d',b)


def test_r3_5_release_anchors_do_not_use_detector_experiment():
    from scripts.phase4_benchmark import release_anchors
    data=release_anchors(Path('output/phase4/inputs/release-readiness'))
    assert data['phase3_full_replay_seconds']==8063
    assert data['stored_bar_only_trial_durations']['read_mode']=='sqlite3 mode=ro'
    assert data['per_level_a2_model']['approved_reader_no_round']['rank']==2
    assert 'variant_prescreen_detector_EXPERIMENT' not in data['per_level_a2_model']


def test_r3_2_waiting_and_interrupted_audit_are_repaired(tmp_path):
    import optuna
    from optuna.trial import TrialState
    study=optuna.create_study(storage='sqlite:///'+str(tmp_path/'unit.db'))
    study.enqueue_trial({'x':1})
    opt.reconcile_phase4_trials(study,SimpleNamespace(phase4=True))
    assert study.trials[0].state==TrialState.FAIL
    first=study.user_attrs['phase4_reconciliation_audit']
    study.ask()
    # Simulate a crash after persisting FAIL, before appending the study audit.
    trial=study.trials[-1]
    record={'trial':trial.number,'from_state':'RUNNING','reason':'orphan after interruption','reconciled_utc':'2026-11-06T00:00:00Z'}
    study._storage.set_trial_user_attr(trial._trial_id,'phase4_reconciled',record)
    study._storage.set_trial_state_values(trial._trial_id,TrialState.FAIL)
    opt.reconcile_phase4_trials(study,SimpleNamespace(phase4=True))
    audit=study.user_attrs['phase4_reconciliation_audit']
    assert audit[:len(first)]==first and audit[-1]==record


def test_r3_1_successful_preflight_records_all_values_without_writes(tmp_path,monkeypatch):
    state={'base':'unit','sha256':'source','dirty':False}
    monkeypatch.setattr(policy,'mount_ready',lambda:True)
    monkeypatch.setattr(policy,'paired_readable',lambda ds:True)
    monkeypatch.setattr(policy,'destination_free_bytes',lambda p:200*GiB)
    monkeypatch.setattr(policy,'assert_no_conflicts',lambda **kw:{'status':'clear'})
    monkeypatch.setattr(policy,'protected_state',lambda:{'owner':'baseline'})
    monkeypatch.setattr(policy,'owner_protected_state',lambda:{'owner':'baseline'})
    monkeypatch.setattr(policy,'source_state',lambda *a:state)
    from scripts import phase4_resources as r
    monkeypatch.setattr(r,'available_memory',lambda:(20*GiB,'unit'))
    result=policy.production_preflight(SimpleNamespace(directory=tmp_path),tmp_path/'new',state,60*GiB)
    assert result['required_disk_bytes']==120*GiB and result['free_disk_bytes']==200*GiB
    assert result['memory']['available_bytes']==20*GiB and result['memory']['pass']
    assert result['source_state']==state and result['protected_state']=={'owner':'baseline'}
    assert not (tmp_path/'new').exists()


def test_r3_1_source_change_during_publish_leaves_no_marker(tmp_path,monkeypatch):
    ds,_=production_unit(tmp_path,monkeypatch)
    monkeypatch.setattr(approvals,'consume_approval',lambda *a:None)
    calls=[]
    def pin(state):
        calls.append(1)
        if len(calls)==2:raise ValueError('source_state changed during publish')
    monkeypatch.setattr(tfc,'verify_source',pin)
    dest=tmp_path/'out'
    with pytest.raises(ValueError,match='source_state'):
        tfc.build(ds,dest,full_history=True,owner_approved='unit')
    assert not (dest/'COMPLETE.json').exists()
    with pytest.raises(ValueError,match='complete'):p4.attach_cache(SimpleNamespace(trade_cache=dest,bar_type=BT,dataset_id=ds.dataset_id),None)


def test_r3_5_campaign_dry_run_reports_memory_plan(tmp_path,monkeypatch):
    from scripts import phase4_campaign as campaign,phase4_resources as r
    monkeypatch.setattr(campaign,'checked_write',lambda p:Path(p))
    monkeypatch.setattr(r,'available_memory',lambda:(20*GiB,'unit'))
    snapshots=[]
    import hashlib
    for i in range(3):
        p=tmp_path/f'legacy-{i}';p.write_bytes(bytes([i]));snapshots.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest=tmp_path/'campaign.json';campaign.register(manifest,snapshots)
    result=campaign.run_driver(manifest)
    assert not result['executed'] and len(result['memory_plans'])==13
    assert all(plan['planned_concurrency']==1 and plan['planned_private_bytes']==4*GiB for plan in result['memory_plans'])

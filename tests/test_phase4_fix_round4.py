"""Synthetic campaign-readiness regressions. No real approvals or studies."""
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

from scripts import optimize_orderflow as opt, phase4_approvals as approvals
from scripts import phase4_campaign as campaign, phase4_studies as p4, phase4_policy as policy
from scripts import trade_feature_cache as tfc
from test_optimize_orderflow import args_for
from test_phase4_fix_round1 import approval_record

GATE_NAMES = ('arms_under_3x','build_under_4h','disk_under_40GiB','worker_under_4GiB','sum_under_24GiB','campaign_total_memory')


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=policy.canonical_bytes(value);path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def bound_gate(tmp_path,monkeypatch):
    import pwd
    monkeypatch.setattr(pwd,'getpwuid',lambda uid:SimpleNamespace(pw_dir=str(tmp_path/'home')))
    monkeypatch.setattr(approvals,'ROOT',tmp_path)
    for module in (campaign,policy,opt,p4):monkeypatch.setattr(module,'ROOT',tmp_path)
    _,record,selector=approval_record(monkeypatch)
    generation=tmp_path/'output/optimize/trade_cache/20261103T011635Z-d3a7b9603b37'
    refs={'full_off':{'lanes':{}},'full_all':{'lanes':{}}};marker={'manifests':{}};lanes={};paths={};verify={}
    for bt in tfc.PRODUCTION_LANES:
        entry={'mode':'production','bar_type':bt,'dataset_id':generation.name,'root_hash':'root',
               'builder_fp':tfc.builder_fingerprint(),'trade_digest':'trade-'+bt,'n_bars':10,'start':None,'end':None}
        raw=policy.canonical_bytes(entry);digest=hashlib.sha256(raw).hexdigest()
        lane=generation/(bt+'-'+digest);lane.mkdir(parents=True);(lane/'manifest.json').write_bytes(raw)
        marker['manifests'][lane.name+'/manifest.json']=digest
        off={'trade_digest':entry['trade_digest'],'bars':10,'replay_digest':'off-'+bt}
        all_on={**off,'replay_digest':'all-'+bt,'raw_signal_digest':'raw-'+bt}
        refs['full_off']['lanes'][bt]=off;refs['full_all']['lanes'][bt]=all_on
        lanes[bt]=off;paths[bt]=str(lane)
        verify[str(lane)]={'off':{'replay_digest':off['replay_digest']},'all':{'replay_digest':all_on['replay_digest'],'raw_signal_digest':all_on['raw_signal_digest']}}
    artifacts={'benchmark':tmp_path/'output/phase4/benchmark/benchmark.json',
               'complete':generation/'COMPLETE.json','build_report':tmp_path/'output/phase4/cache-build.json',
               'verify_report':tmp_path/'output/phase4/fullhistory/verify.stdout'}
    values={'benchmark':{'extrapolation':{'resource_gates':dict.fromkeys(GATE_NAMES,True)}},
            'complete':marker,'build_report':{'full_history':True,'mode':'production','dataset_id':generation.name,
                'root_hash':'root','cache_paths':paths,'lanes':lanes},'verify_report':{'lanes':verify}}
    write_json(tmp_path/'tests/fixtures/phase4_reference_digests.json',refs)
    gates={key:record[key] for key in ('plan_sha256','campaign_sha256','base_commit','source_state','created_utc')}
    gates.update(dict.fromkeys(approvals.GATES,'pass'))
    def seal():
        for name,path in artifacts.items():gates[name+'_sha256']=write_json(path,values[name])
        write_json(approvals.approval_root()/('gates-'+record['campaign_sha256']+'.json'),gates)
    seal()
    def admit():return approvals.require_campaign_approval(selector,record['campaign_sha256'],record['command_argv'])
    return SimpleNamespace(artifacts=artifacts,values=values,gates=gates,seal=seal,admit=admit,record=record,selector=selector)


def test_r4_1_valid_bound_artifacts(bound_gate):
    assert bound_gate.admit()['sha256']==bound_gate.selector


@pytest.mark.parametrize('entry',['require','driver','ablation','child'])
def test_r4_1_failed_speed_gate_before_any_open(bound_gate,monkeypatch,entry):
    b=bound_gate;b.values['benchmark']['extrapolation']['resource_gates']['arms_under_3x']=False;b.seal()
    def forbidden(*a,**kw):pytest.fail('study/cache opened before refusal')
    monkeypatch.setattr(opt,'build_cache',forbidden);monkeypatch.setattr(opt,'open_study',forbidden)
    monkeypatch.setattr(opt,'CachedBars',forbidden);monkeypatch.setattr(tfc,'CachedTradeUpdates',forbidden)
    monkeypatch.setattr(p4,'read_campaign',lambda *a:({'arms':{}},b.record['campaign_sha256']))
    monkeypatch.setattr(campaign,'read_campaign',p4.read_campaign)
    monkeypatch.setattr(approvals,'command_argv',lambda argv=None:b.record['command_argv'])
    monkeypatch.setattr(campaign,'preflight',forbidden)
    if entry=='require':call=b.admit
    elif entry=='driver':
        driver=b.artifacts['benchmark']
        assert campaign.ROOT==policy.ROOT==driver.parents[3], 'campaign tests must use a temporary checkpoint root'
        call=lambda:campaign.run_driver('unit',stage='A',owner_approved=b.selector,driver=driver,inspected_driver=hashlib.sha256(driver.read_bytes()).hexdigest(),dry_run=False)
    elif entry=='ablation':call=lambda:campaign.ablation(b.selector,'unit')
    else:
        a=opt.build_parser().parse_args(['--phase4','--no-warm-start','--owner-approved',b.selector,'--campaign-manifest','unit',
            '--study-db',str(policy.ROOT/'output/optimize/phase4/studies.db'),'--output',str(policy.ROOT/'output/optimize/phase4/results.json'),
            '--cache-dir',str(policy.ROOT/'output/optimize/phase4/cache')])
        call=lambda:p4.configure(a)
    with pytest.raises(ValueError,match='arms_under_3x'):call()
    assert not (approvals.approval_root()/(b.selector+'.consumed')).exists()


@pytest.mark.parametrize('name',['benchmark','verify_report','complete','build_report'])
def test_r4_1_tampered_bound_artifact(bound_gate,name):
    path=bound_gate.artifacts[name];path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='hash'):bound_gate.admit()


@pytest.mark.parametrize('damage',['missing_lane','wrong_digest','foreign_generation','wrong_trade_digest'])
def test_r4_1_verify_must_match_current_six_lanes(bound_gate,damage):
    b=bound_gate;lanes=b.values['verify_report']['lanes'];path=next(iter(lanes))
    if damage=='missing_lane':lanes.pop(path)
    elif damage=='foreign_generation':lanes[path+'-other']=lanes.pop(path)
    elif damage=='wrong_digest':lanes[path]['all']['raw_signal_digest']='wrong'
    else:next(iter(b.values['build_report']['lanes'].values()))['trade_digest']='wrong'
    b.seal()
    with pytest.raises(ValueError,match='certified|generation|digest'):b.admit()


@pytest.mark.parametrize('damage',['none','stale','failed','unknown','symlink','writable','tampered'])
def test_r4_1_host_waiver(bound_gate,tmp_path,damage):
    b=bound_gate;b.values['benchmark']['extrapolation']['resource_gates']['arms_under_3x']=False;b.seal()
    measured=tmp_path/'replacement.json';measured_hash=write_json(measured,{'trial_seconds':20. if damage=='failed' else 2.})
    waiver={'gate_path':'unknown' if damage=='unknown' else 'extrapolation.arms_under_3x',
        'benchmark_sha256':'0'*64 if damage=='stale' else b.gates['benchmark_sha256'],
        'plan_sha256':approvals.PLAN_SHA256,'campaign_sha256':b.record['campaign_sha256'],
        'replacement_gate':{'name':'owner absolute runtime','threshold_spec':{'metric':['trial_seconds'],'operator':'<=','value':3.},
            'measured_artifact_path':str(measured),'measured_artifact_sha256':measured_hash},'created_utc':b.record['created_utc']}
    raw=policy.canonical_bytes(waiver);digest=hashlib.sha256(raw).hexdigest()
    path=approvals.approval_root()/('waiver-'+digest+'.json');path.write_bytes(raw)
    b.gates['waivers']=[digest];b.seal()
    if damage=='symlink':path.unlink();path.symlink_to(measured)
    if damage=='writable':path.chmod(0o666)
    if damage=='tampered':path.write_bytes(raw+b' ')
    if damage=='none':assert b.admit()['sha256']==b.selector
    else:
        with pytest.raises(ValueError,match='waiver|writable|symlink'):b.admit()


@pytest.mark.parametrize('field',['concurrency','n_jobs'])
def test_r4_3_parallel_configuration_refused(tmp_path,monkeypatch,field):
    manifest={'phase':'phase4','arms':{},field:2};path=tmp_path/'campaign.json'
    digest=write_json(path,manifest);Path(str(path)+'.sha256').write_text(digest)
    with pytest.raises(ValueError,match='sequential|concurrency|n_jobs'):campaign.run_driver(path)


def test_r4_3_dry_run_is_sequential(tmp_path,monkeypatch):
    path=tmp_path/'campaign.json';digest=write_json(path,{'phase':'phase4','arms':{}})
    Path(str(path)+'.sha256').write_text(digest)
    assert campaign.run_driver(path)['concurrency']==1


@pytest.mark.parametrize('limit',['trial','study','absent'])
def test_r4_2_slow_objective_runtime_stop(tmp_path,monkeypatch,limit):
    a=args_for(tmp_path,phase4=True,n_trials=3,n_jobs=1)
    if limit!='absent':setattr(a,'max_'+limit+'_seconds',.01)
    study=optuna.create_study()
    def objective(trial,*args):time.sleep(.025);return trial.suggest_float('x',0.,1.)
    monkeypatch.setattr(opt,'objective',objective)
    opt.optimize_loop(study,None,None,a)
    assert len(opt.complete_trials(study))==(3 if limit=='absent' else 1)
    assert all(t.user_attrs['phase4_wall_seconds']>=.025 for t in study.trials)
    if limit!='absent':
        assert study.user_attrs['phase4_runtime_stop']['reason']=='max_'+limit+'_seconds'
        opt.optimize_loop(study,None,None,a)
        assert len(opt.complete_trials(study))==1
    else:assert 'phase4_runtime_stop' not in study.user_attrs


def test_r4_2_runtime_limits_bound_to_campaign_arm(tmp_path,monkeypatch):
    a=args_for(tmp_path,phase4=True);a.campaign_manifest='unit'
    data={k:getattr(a,k) for k in ('seed','fee_bps','slippage_bps','min_total_trades','min_trades_per_fold','n_splits','embargo','holdout_fraction')}
    data['arms']={p4.arm_key(a):{'budget':a.n_trials,'max_trial_seconds':3.,'max_study_seconds':10.}}
    monkeypatch.setattr(p4,'read_campaign',lambda *a:(data,'campaign'))
    p4.campaign_arm(a)
    assert a.max_trial_seconds==3. and a.max_study_seconds==10.


def test_r4_2_runtime_stopped_candidate_and_report_refused(tmp_path,monkeypatch):
    monkeypatch.setattr(opt,'read_study_attrs',lambda *a:{'identity':{'phase':'phase4'},'phase4_runtime_stop':{'reason':'max_trial_seconds'}})
    monkeypatch.setattr(opt,'read_study_trials',lambda *a:pytest.fail('read trials after runtime stop'))
    with pytest.raises(ValueError,match='runtime'):p4.completed_candidate('db','study',0,1)
    from scripts import phase4_report as report
    monkeypatch.setattr(policy,'phase4_db_path',lambda p:Path(p));db=tmp_path/'unit.db'
    # Snapshot admission must fail before backup/replay (real synthetic SQLite schema).
    study=optuna.create_study(storage='sqlite:///'+str(db),study_name='unit')
    study.set_user_attr('identity',{'phase':'phase4','campaign_sha256':'campaign','arm':'unit'})
    study.set_user_attr('phase4_runtime_stop',{'reason':'max_trial_seconds'})
    snapshots=[]
    for i in range(3):
        path=tmp_path/f'legacy-{i}';path.write_bytes(bytes([i]))
        snapshots.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    monkeypatch.setattr(policy,'ROOT',tmp_path)
    manifest={'phase4_db':str(db),'snapshots':snapshots,'arms':{'unit':{'budget':1}}}
    with pytest.raises(ValueError,match='runtime'):report.report_snapshots(manifest,'campaign',db)


def test_r4_4_approval_history_retains_first(tmp_path,monkeypatch):
    a=args_for(tmp_path,phase4=True,approval_sha256='first',gates_sha256='gates1')
    monkeypatch.setattr(p4,'campaign_arm',lambda *a,**kw:{})
    study=opt.open_study(a,'unit',{'unit':True})
    a.approval_sha256='second';a.gates_sha256='gates2'
    study=opt.open_study(a,'unit',{'unit':True})
    attrs=study.user_attrs
    assert attrs['owner_approval_sha256']=='first' and attrs['gates_sha256']=='gates1'
    assert [(r['approval_sha256'],r['gates_sha256']) for r in attrs['phase4_approvals']]==[('first','gates1'),('second','gates2')]
    assert all(r['utc'] for r in attrs['phase4_approvals'])


def test_r4_5_sigkill_resume_matches_uninterrupted(tmp_path):
    code=r'''
import sys,time,json
from pathlib import Path
from types import SimpleNamespace
import optuna
from scripts import optimize_orderflow as o
root=Path(sys.argv[1]); stage=sys.argv[2];db=root/("resumed.db" if stage!="whole" else "whole.db")
s=optuna.create_study(storage="sqlite:///"+str(db),study_name="unit",direction="maximize",load_if_exists=True)
a=SimpleNamespace(phase4=True,n_trials=14,seed=42,n_jobs=1)
if stage=="resume":o.reconcile_phase4_trials(s,a)
def objective(t,*unused):
    x=t.suggest_float("x",-3.,5.); y=t.suggest_int("y",1,8)
    if stage=="kill" and len(o.complete_trials(s))==11:
        (root/"ready").touch();time.sleep(60)
    return -(x-1.)**2+y
o.objective=objective
o.optimize_loop(s,None,None,a)
(root/(stage+".json")).write_text(json.dumps([(t.params,t.value) for t in o.complete_trials(s)]))
'''
    env=os.environ.copy();env['PYTHONPATH']=str(Path.cwd())
    first=subprocess.Popen([sys.executable,'-c',code,str(tmp_path),'kill'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+40
        while not (tmp_path/'ready').exists():
            assert first.poll() is None,first.communicate()[1].decode()
            assert time.monotonic()<deadline,'synthetic study did not reach interruption'
            time.sleep(.03)
        first.send_signal(signal.SIGKILL);first.communicate(timeout=10)
        for stage in ('resume','whole'):
            result=subprocess.run([sys.executable,'-c',code,str(tmp_path),stage],env=env,capture_output=True,timeout=60)
            assert result.returncode==0,result.stderr.decode()
        assert json.loads((tmp_path/'resume.json').read_text())==json.loads((tmp_path/'whole.json').read_text())
    finally:
        if first.poll() is None:first.kill();first.communicate(timeout=10)


@pytest.mark.parametrize('name',GATE_NAMES)
def test_r4_1_every_resource_gate_fails_closed(bound_gate,name):
    b=bound_gate;b.values['benchmark']['extrapolation']['resource_gates'][name]=False;b.seal()
    with pytest.raises(ValueError,match=name):b.admit()


def test_r4_1_conservative_gate_cannot_be_hidden(bound_gate):
    b=bound_gate
    b.values['benchmark']['extrapolation']['conservative_upper_bound']={'resource_gates':{'build_under_4h':False}}
    b.seal()
    with pytest.raises(ValueError,match='build_under_4h'):b.admit()


def test_r4_3_execution_lock_refuses_another_process(tmp_path,monkeypatch):
    monkeypatch.setattr(opt,'ROOT',tmp_path);monkeypatch.setattr(policy,'ROOT',tmp_path)
    code='''import sys
from pathlib import Path
from scripts import optimize_orderflow as opt,phase4_policy as policy
opt.ROOT=policy.ROOT=Path(sys.argv[1])
with opt.phase4_execution(True):raise AssertionError("concurrent admission")
'''
    with opt.phase4_execution(True):
        result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True,timeout=15)
    assert result.returncode!=0 and 'another Phase 4 campaign process' in result.stderr
    assert 'concurrent admission' not in result.stderr

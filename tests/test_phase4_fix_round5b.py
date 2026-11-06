"""Stage-scoped driver regressions with synthetic artifacts, studies and approvals."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import optuna
import pytest

from scripts import phase4_campaign as campaign,phase4_policy as policy,phase4_studies as p4
from scripts import optimize_orderflow as opt,phase4_approvals as approvals
from test_phase4_fix_round5 import SCRIPT,finish
from test_warm_start import make_source_study,valid_params
from test_pipeline import synth_bars

GiB=1024**3
BARS=('fixed_tick_imbalance','fixed_volume_imbalance')
MODES=('off','absorption_levels','exhaustion_bonus','exhaustion_required','framing_trades')


@pytest.fixture
def driver_unit(tmp_path,monkeypatch):
    from scripts import phase4_resources as resources,measure_rss,diagnose_trades as diagnose
    from scripts.phase4_benchmark import MODES as ALL_MODES
    for module in (campaign,policy,opt,p4):monkeypatch.setattr(module,'ROOT',tmp_path)
    state={'base':'unit','sha256':'source','dirty':True}
    monkeypatch.setattr(campaign,'source_state',lambda:dict(state))
    monkeypatch.setattr(resources,'enforce_current',lambda *a:{'pass':True})
    protected=tmp_path/'protected';protected.write_text('same')
    protection=lambda:{str(protected):campaign.sha256_file(protected)}
    monkeypatch.setattr(campaign,'protected_state',protection)
    monkeypatch.setattr(campaign,'preflight',lambda *a:protection())
    fault={};calls=[];events=[]
    def gates(selector,digest,argv=None):
        if fault.get('gates') and calls:raise ValueError('injected gate recheck failure')
        return {'sha256':selector,'gates_sha256':'gates'}
    monkeypatch.setattr(p4,'require_gates',gates)
    monkeypatch.setattr(approvals,'approval_for_command',lambda digest,cmd:hashlib.sha256(policy.canonical_bytes(cmd)).hexdigest())
    monkeypatch.setattr(campaign,'command_memory_plan',lambda *a,**kw:{'pass':True,'unique_mapped_bytes':0,'available_bytes':20*GiB,'planned_concurrency':1})
    db=tmp_path/'output/optimize/phase4/studies.db'
    manifest={'phase':'phase4','version':1,'phase4_db':str(db),'source_state':dict(state),'arms':{},
        'seed':42,'fee_bps':5.,'slippage_bps':1.,'min_total_trades':105,'min_trades_per_fold':11,
        'n_splits':5,'embargo':.02,'holdout_fraction':.2}
    for bt in BARS:
        for mode in MODES:
            manifest['arms']['BTCUSDT:'+bt+':'+mode]={'kind':'candidate' if mode=='framing_trades' else 'study','budget':1 if mode=='framing_trades' else 2}
    path=tmp_path/'output/phase4/unit.json';path.parent.mkdir(parents=True)
    raw=policy.canonical_bytes(manifest);path.write_bytes(raw);digest=hashlib.sha256(raw).hexdigest();Path(str(path)+'.sha256').write_text(digest)
    driver=tmp_path/'driver.py';driver.write_text('unit')
    def field(cmd,name):return cmd[cmd.index(name)+1]
    def name(bt,mode):return 'unit__'+bt+'__'+mode
    def run(cmd,**kw):
        child=field(cmd,'--owner-approved');cmd=cmd[:-2];calls.append(cmd);events.append('child')
        if fault.get('child')==len(calls):return {'returncode':2,'wall_seconds':.01,'peak_rss_bytes':1000000,'peak_private_bytes':500000}
        if 'ablation' in cmd:
            for bt in campaign_bar_types():
                output=tmp_path/'output/phase4/ablation'/f'{bt}.json';output.parent.mkdir(parents=True,exist_ok=True)
                opt.atomic_json(output,{m:{'owner_approval_sha256':child,'gates_sha256':'gates','replay_digest':'unit'} for m in ALL_MODES})
        elif 'freeze' in cmd:
            bt=field(cmd,'--bar-type')
            campaign.freeze_baseline(db,name(bt,'off'),2,Path(field(cmd,'--out')))
        else:
            bt,mode=field(cmd,'--bar-type'),field(cmd,'--trade-signals');arm='BTCUSDT:'+bt+':'+mode
            if mode=='framing_trades':
                artifact={'arm':arm,'campaign_sha256':digest,'mode':mode,'replay_digest':'unit','raw_signal_digest':'unit'}
                encoded=policy.canonical_bytes(artifact);sha=hashlib.sha256(encoded).hexdigest()
                output=tmp_path/'output/phase4/candidates'/f'{sha}.json';output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(encoded)
            else:
                db.parent.mkdir(parents=True,exist_ok=True)
                if not db.exists() or name(bt,mode) not in opt.list_study_names(db):
                    identity={'phase':'phase4','campaign_sha256':digest,'arm':arm,'trade_signals':'disabled' if mode=='off' else {},'data_fp':bt}
                    make_source_study(db,name(bt,mode),[{'value':float(i),'attrs':{'params':valid_params(),'replay_digest':'unit'}} for i in range(2)],study_attrs={'identity':identity})
                    opt.study_lock_path(db,name(bt,mode)).touch()
                if fault.get('runtime'):
                    study=optuna.load_study(storage='sqlite:///'+str(db),study_name=name(bt,mode));study.set_user_attr('phase4_runtime_stop',{'reason':'max_trial_seconds'})
        if fault.get('source'):state['sha256']='changed'
        if fault.get('protected'):protected.write_text('changed')
        return {'returncode':0,'wall_seconds':.01,'peak_rss_bytes':5*GiB if fault.get('memory') else 1000000,
                'peak_private_bytes':500000,'private_is_fallback':False,'private_metric':'unit'}
    monkeypatch.setattr(measure_rss,'run_measured',run)
    monkeypatch.setattr(diagnose,'find_cached_bars',lambda cache,fp:(tmp_path/'unused.bin',{'start':'2023-01-01','end':'2023-01-02','symbol':'BTCUSDT','bar_type':fp}))
    class UnitBars(list):
        def __init__(self,*a,**kw):super().__init__(synth_bars(120))
    monkeypatch.setattr(opt,'CachedBars',UnitBars)
    def replay(*a,**kw):events.append('verify');return {'replay_digest':'wrong' if fault.get('digest') else 'unit'}
    monkeypatch.setattr(opt,'run_backtest',replay)
    def execute(stage,owner=None):
        return campaign.run_driver(path,stage=stage,owner_approved=owner or 'owner-'+stage,
            inspected_driver=campaign.sha256_file(driver),driver=driver,dry_run=False)
    return SimpleNamespace(path=path,manifest=manifest,digest=digest,driver=driver,calls=calls,events=events,fault=fault,
        execute=execute,root=tmp_path,db=db,state=state)


def campaign_bar_types():
    from orderflow_system.data.models import BarType
    return [bt.value for bt in BarType]


def checkpoint(unit,stage,stopped=False):
    path=unit.root/'output/phase4/checkpoints'/unit.digest/f'stage-{stage}{".stopped" if stopped else ""}.json'
    return path,json.loads(path.read_text())


def test_5b_execute_requires_explicit_stage(driver_unit):
    u=driver_unit
    with pytest.raises(ValueError,match='stage'):campaign.run_driver(u.path,owner_approved='owner',driver=u.driver,
        inspected_driver=campaign.sha256_file(u.driver),dry_run=False)
    assert not u.calls


def test_5b_dry_run_groups_and_filters_stages(driver_unit):
    u=driver_unit;all_plan=campaign.run_driver(u.path)
    assert [len(all_plan['stages'][s]) for s in ('A','B','C')]==[1,4,8]
    for stage,count in (('A',1),('B',4),('C',8)):
        result=campaign.run_driver(u.path,stage=stage)
        assert len(result['commands'])==count and not result['executed']
    c=all_plan['stages']['C'];assert all('framing_trades' not in cmd for cmd in c[:6])
    assert all('framing_trades' in cmd for cmd in c[6:])
    assert not u.calls and not u.db.exists()


def test_5b_stage_approval_cannot_admit_another_stage(driver_unit,monkeypatch):
    u=driver_unit
    from test_phase4_fix_round1 import approval_record
    monkeypatch.setattr(approvals,'approval_root',lambda:u.root/'approvals/phase4')
    argv_a=['python','phase4_campaign.py','driver','--execute','--stage','A']
    _,record,selector=approval_record(monkeypatch,campaign=u.digest,argv=argv_a)
    monkeypatch.setattr(approvals,'command_argv',lambda argv=None:[*argv_a[:-1],'B'])
    monkeypatch.setattr(p4,'require_gates',approvals.require_campaign_approval)
    with pytest.raises(ValueError,match='argv'):u.execute('B',selector)
    assert not u.calls


def test_5b_A_checkpoint_then_B_then_C_and_stop_each_time(driver_unit):
    u=driver_unit
    u.execute('A');assert len(u.calls)==1
    path,a=checkpoint(u,'A');assert a['next_stage_requires_owner_approval'] is True and a['stage']=='A'
    assert len(a['commands'])==1 and len(a['artifacts'])==6
    u.execute('B');assert len(u.calls)==5
    _,b=checkpoint(u,'B');assert b['protected_before']==b['protected_after']
    assert len(b['commands'])==4 and sum('freeze' in c for c in u.calls)==2
    assert len(opt.list_study_names(u.db))==2
    for bt in BARS:assert (u.root/'output/phase4/frozen'/f'{bt}.json').exists()
    before=len(u.events);u.execute('C');assert u.events[before:before+2]==['verify','verify']
    _,c=checkpoint(u,'C');assert len(c['commands'])==8
    assert len(opt.list_study_names(u.db))==8  # P2 creates artifacts, no study.
    assert len(list((u.root/'output/phase4/candidates').glob('*.json')))==2
    log=[json.loads(line) for line in (u.root/'output/phase4/campaign-log.jsonl').read_text().splitlines()]
    events=[r for r in log if r.get('checkpoint_sha256') and r.get('status')=='stage_complete']
    assert len(events)==3
    for row in events:
        file=u.root/'output/phase4/checkpoints'/u.digest/f'stage-{row["stage"]}.json'
        assert campaign.sha256_file(file)==row['checkpoint_sha256']


@pytest.mark.parametrize('stage',['B','C'])
def test_5b_missing_predecessor_refuses_before_child(driver_unit,stage):
    u=driver_unit
    with pytest.raises(ValueError,match='checkpoint'):u.execute(stage)
    assert not u.calls


def test_5b_valid_stage_cannot_run_twice(driver_unit):
    u=driver_unit;u.execute('A')
    with pytest.raises(ValueError,match='already'):u.execute('A','new-owner')
    assert len(u.calls)==1


@pytest.mark.parametrize('fault',['child','memory','protected','source','gates'])
def test_5b_failed_stage_writes_only_stopped_checkpoint(driver_unit,fault):
    u=driver_unit;u.fault[fault]=1 if fault=='child' else True
    with pytest.raises(ValueError):u.execute('A')
    path,stopped=checkpoint(u,'A',True)
    assert stopped['status']=='stopped' and stopped['commands']
    assert not path.with_name('stage-A.json').exists()


def test_5b_runtime_stop_prevents_stage_complete(driver_unit):
    u=driver_unit;u.execute('A');u.fault['runtime']=True
    with pytest.raises(ValueError,match='runtime'):u.execute('B')
    _,stopped=checkpoint(u,'B',True)
    assert stopped['runtime_stops']
    assert not (u.root/'output/phase4/checkpoints'/u.digest/'stage-B.json').exists()


def test_5b_resume_stage_new_owner_no_duplicate_complete_trials(driver_unit):
    u=driver_unit;u.execute('A');u.fault['child']=3
    with pytest.raises(ValueError):u.execute('B','owner-B-first')
    assert len(opt.list_study_names(u.db))==1
    u.fault.clear()
    with pytest.raises(ValueError,match='new.*approval'):u.execute('B','owner-B-first')
    u.execute('B','owner-B-second')
    names=opt.list_study_names(u.db);assert len(names)==2
    assert all(sum(t['state']=='COMPLETE' for t in opt.read_study_trials(u.db,n))==2 for n in names)


def test_5b_checkpoint_artifact_tamper_refuses_next_stage(driver_unit):
    u=driver_unit;u.execute('A');_,a=checkpoint(u,'A')
    path=Path(a['artifacts'][0]['path']);path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='hash|artifact'):u.execute('B')
    assert len(u.calls)==1


def test_5b_C_checks_the_checkpoint_chain(driver_unit):
    u=driver_unit;u.execute('A');u.execute('B');_,a=checkpoint(u,'A');before=len(u.calls)
    path=Path(a['artifacts'][0]['path']);path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='hash|artifact'):u.execute('C')
    assert len(u.calls)==before


def test_5b_started_driver_without_stop_record_needs_new_approval(driver_unit):
    u=driver_unit
    log=u.root/'output/phase4/campaign-log.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
    log.write_bytes(policy.canonical_bytes({'status':'stage_started','stage':'A','campaign_sha256':u.digest,'owner_approval_sha256':'interrupted-owner'}))
    with pytest.raises(ValueError,match='new.*approval'):u.execute('A','interrupted-owner')
    assert not u.calls
    u.execute('A','fresh-owner');assert len(u.calls)==1


@pytest.mark.parametrize('damage',['missing','params','digest','count','pending'])
def test_5b_C_revalidates_both_frozen_B0_before_children(driver_unit,damage):
    u=driver_unit;u.execute('A');u.execute('B');before=len(u.calls)
    path=u.root/'output/phase4/frozen'/f'{BARS[0]}.json'
    if damage=='missing':path.unlink()
    elif damage=='params':
        data=json.loads(path.read_text());data['params']['price_proximity_pct']=.99;opt.atomic_json(path,data)
    elif damage=='digest':u.fault['digest']=True
    else:
        name='unit__'+BARS[0]+'__off';study=optuna.load_study(storage='sqlite:///'+str(u.db),study_name=name)
        if damage=='count':study.add_trial(optuna.trial.create_trial(value=0.))
        else:study.ask()
    with pytest.raises((ValueError,OSError)):u.execute('C')
    assert len(u.calls)==before


@pytest.mark.parametrize('kind,reason',[('live','max_trial_seconds'),('memory','memory')])
def test_5b_lost_live_stop_is_reconstructed_and_requires_host_clear(tmp_path,monkeypatch,kind,reason):
    mode=kind+'_lost'
    code=SCRIPT.replace("db=root/'unit.db'", "db=root/'output/optimize/phase4/unit.db';db.parent.mkdir(parents=True,exist_ok=True)")
    code=code.replace("if mode=='memory':", "if mode in ('memory','memory_lost'):")
    code=code.replace("if mode=='live' else None", "if mode in ('live','live_lost') else None")
    code=code.replace('original_timing=o.phase4_trial_timing', '''s.set_user_attr('identity',{'phase':'phase4','unit':'lost-stop'})
original_study_attr=optuna.study.Study.set_user_attr
def lose_stop(self,key,value):
    if key=='phase4_runtime_stop':raise OSError('injected stop write outage')
    return original_study_attr(self,key,value)
optuna.study.Study.set_user_attr=lose_stop
original_timing=o.phase4_trial_timing''')
    proc=subprocess.Popen([sys.executable,'-c',code,str(tmp_path),mode],env={**os.environ,'PYTHONPATH':str(Path.cwd())},stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    out,err=finish(proc);assert proc.returncode!=0 and 'Stop persistence failed' in err
    db=tmp_path/'output/optimize/phase4/unit.db';study=optuna.load_study(storage='sqlite:///'+str(db),study_name='unit')
    assert study.user_attrs.get('phase4_runtime_stop') is None
    args=SimpleNamespace(phase4=True,n_jobs=1,n_trials=1,seed=42,max_trial_seconds=.12 if kind=='live' else 1.)
    opt.reconcile_phase4_trials(study,args)
    stop=study.user_attrs['phase4_runtime_stop'];assert stop['reason']==reason and stop['reconciled'] is True and stop['live'] is False
    opt.optimize_loop(study,None,None,args);assert len(study.trials)==1 and not opt.complete_trials(study)
    for module in (campaign,policy,opt):monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(approvals,'approval_root',lambda:tmp_path/'approvals/phase4')
    record={'plan_sha256':approvals.PLAN_SHA256,'study_identity_sha256':opt.fingerprint(study.user_attrs['identity']),
        'stop_sha256':opt.fingerprint(stop),'decision':'clear_and_restart_budget','created_utc':'2026-11-06T00:00:00Z'}
    raw=policy.canonical_bytes(record);sha=hashlib.sha256(raw).hexdigest();path=approvals.approval_root()/('runtime-decision-'+sha+'.json');path.parent.mkdir(parents=True);path.write_bytes(raw)
    campaign.clear_runtime_stop(db,'unit',sha)
    monkeypatch.setattr(opt,'objective',lambda *a:1.)
    opt.optimize_loop(study,None,None,args);assert len(opt.complete_trials(study))==1

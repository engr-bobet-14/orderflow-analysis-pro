"""Host fix round 1 contract. Each numbered item has recorded red evidence."""
import hashlib
import importlib
import io
import builtins
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import optimize_orderflow as opt,phase4_studies as p4,trade_feature_cache as tfc
from scripts import phase4_policy as policy,phase4_report as report,phase4_benchmark as bench
from orderflow_system.data.trade_stream import TradeStream
from tests.test_phase4_cache import paired,BT
from test_phase4_studies import args as cli_args


@pytest.fixture(autouse=True)
def isolated_home(tmp_path,monkeypatch):
    import pwd
    monkeypatch.setattr(pwd,'getpwuid',lambda uid:SimpleNamespace(pw_dir=str(tmp_path/'home')))


@pytest.mark.parametrize('wrong', ['campaign','budget'])
def test_b1_frozen_b0_bound_to_campaign(tmp_path,monkeypatch,wrong):
    from test_warm_start import valid_params
    params=valid_params()
    spec={'study_db':str(p4.ROOT/'output/optimize/phase4/unit.db'),'study':'B0','trial':0,
          'budget':3 if wrong=='budget' else 200,'params':params,'replay_digest':'digest'}
    path=tmp_path/'frozen.json';path.write_text(json.dumps(spec))
    a=SimpleNamespace(freeze_params=path,symbol='BTCUSDT',bar_type=BT,campaign_manifest=tmp_path/'campaign',campaign_identity='campaign')
    monkeypatch.setattr(p4,'read_campaign',lambda *a:({'arms':{'BTCUSDT:'+BT+':off':{'budget':200}}},'campaign'))
    identity={'trade_signals':'disabled','campaign_sha256':'other' if wrong=='campaign' else 'campaign'}
    monkeypatch.setattr(opt,'read_study_attrs',lambda *a:{'identity':identity})
    monkeypatch.setattr(p4,'completed_candidate',lambda *a:({'attrs':{'params':params}},identity,{'replay_digest':'digest'}))
    monkeypatch.setattr(opt,'split_timeline',lambda *a:(1,[]))
    monkeypatch.setattr(opt,'run_backtest',lambda *a,**kw:{'replay_digest':'digest'})
    with pytest.raises(ValueError,match='campaign|budget'):
        p4.load_frozen(a,[SimpleNamespace(bar_type=BT)])


def test_b2_bounded_build_refuses_production_root_before_open(monkeypatch,capsys):
    monkeypatch.setattr(tfc,'open_registered',lambda *a:pytest.fail('opened dataset before root refusal'))
    with pytest.raises(SystemExit) as error:
        tfc.main(['build','--start','2021-05-19','--end','2021-05-20','--out',str(tfc.ROOT/'output/optimize/trade_cache')])
    assert error.value.code==2 and 'bounded' in capsys.readouterr().err


def sqlite_campaign(path,campaign='campaign',state='COMPLETE',budget=2):
    path.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(path) as con:
        con.executescript('CREATE TABLE studies(study_id INTEGER,study_name TEXT);'
            'CREATE TABLE study_user_attributes(study_id INTEGER,key TEXT,value_json TEXT);'
            'CREATE TABLE trials(trial_id INTEGER,study_id INTEGER,number INTEGER,state TEXT);'
            'CREATE TABLE trial_user_attributes(trial_id INTEGER,key TEXT,value_json TEXT);')
        con.execute('INSERT INTO studies VALUES(1,?)',('B0',))
        con.execute('INSERT INTO study_user_attributes VALUES(1,?,?)',('identity',json.dumps(
            {'phase':'phase4','campaign_sha256':campaign,'arm':'BTCUSDT:'+BT+':off'})))
        con.executemany('INSERT INTO trials VALUES(?,1,?,?)',[(i,i,state) for i in range(budget)])


@pytest.mark.parametrize('state,budget',[('COMPLETE',2),('RUNNING',2),('WAITING',2),('COMPLETE',1)])
def test_b3_report_snapshots_read_only_complete_campaign(tmp_path,monkeypatch,state,budget):
    monkeypatch.setattr(policy,'ROOT',tmp_path)
    db=tmp_path/'output/optimize/phase4/studies.db';sqlite_campaign(db,state=state,budget=budget)
    snapshots=[]
    for i in range(3):
        path=tmp_path/f'legacy{i}.db';path.write_bytes(b'legacy'+bytes([i]))
        snapshots.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    campaign={'snapshots':snapshots,'phase4_db':str(db),'arms':{'BTCUSDT:'+BT+':off':{'kind':'study','budget':2}}}
    original=sqlite3.connect;opens=[]
    out=tmp_path/'output/phase4/reports/snapshots'
    def connect(path,*args,**kw):
        opens.append(str(path))
        assert '?mode=ro' in str(path) or Path(str(path)).resolve().is_relative_to(out)
        return original(path,*args,**kw)
    monkeypatch.setattr(sqlite3,'connect',connect)
    if state!='COMPLETE' or budget!=2:
        with pytest.raises(ValueError,match='budget|RUNNING|WAITING'):
            report.report_snapshots(campaign,'campaign',db,out)
        assert not out.exists()
    else:
        result=report.report_snapshots(campaign,'campaign',db,out)
        assert len(result)==4 and result[-1]['sha256']==report.sha256_file(Path(result[-1]['path']))
        assert any('?mode=ro' in p for p in opens)
        with original(result[-1]['path']) as con:
            assert con.execute('SELECT count(*) FROM trials').fetchone()[0]==2


def test_b4_arbitrary_approval_cannot_unlock_full_history():
    with pytest.raises(ValueError,match='approval'):
        tfc.validate_build_range(None,None,full_history=True,owner_approved='arbitrary')


def test_b4_old_local_gates_cannot_unlock(tmp_path,monkeypatch):
    monkeypatch.setattr(p4,'ROOT',tmp_path)
    path=tmp_path/'output/phase4/gates.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps({k:'pass' for k in ('U','FH1','FH2','FH3','FH4','R')}))
    with pytest.raises(ValueError,match='approval|host'):
        p4.require_gates()


def test_b5_straddling_file_refused_before_any_open(tmp_path,monkeypatch):
    ds=paired(tmp_path)
    ds.manifest['files'][0]['max_time']=1704067200000
    forbidden=str(Path(ds.manifest['source_root'])/ds.manifest['files'][0]['path'])
    def intercept(original):
        def checked(path,*args,**kw):
            assert isinstance(path,int) or str(path)!=forbidden
            return original(path,*args,**kw)
        return checked
    for module in (builtins,io,os):
        monkeypatch.setattr(module,'open',intercept(module.open))
    with pytest.raises(ValueError,match='straddl'):
        TradeStream(ds,development_only=True,temp_dir=tmp_path/'spill')


def test_b6_all_bar_fields_including_gap_must_match(tmp_path,monkeypatch):
    ds=paired(tmp_path)
    monkeypatch.setattr(tfc,'pinned_root',lambda _:ds.root_hash)
    monkeypatch.setattr(tfc,'checked_write',lambda p:Path(p))
    paths,_=tfc.build(ds,tmp_path/'cache',start='2023-01-01',end='2023-01-04')
    cache=tfc.CachedTradeUpdates(paths[BT])
    for field in opt.BAR_DTYPE.names:
        bars=cache.bars();bars.array=bars.array.copy()
        bars.array[field][0]=not bars.array[field][0] if field=='data_gap' else bars.array[field][0]+1
        with pytest.raises(ValueError,match='alignment'):
            cache.validate_bars(bars)


def test_b7_implied_activation_does_not_change_identity(tmp_path):
    from test_optimize_orderflow import args_for
    a=args_for(tmp_path,phase4=True,phase4_implied=False)
    first=opt.study_identity(a,{'fingerprint':'f'*64})
    a.phase4_implied=True
    assert opt.study_identity(a,{'fingerprint':'f'*64})==first
    assert 'phase4_implied' not in first[1]


def test_b7_second_study_same_arm_refused(tmp_path,monkeypatch):
    a=cli_args('--dry-run');a.symbol,a.bar_type='BTCUSDT',BT
    a.space=dict(opt.SEARCH_SPACE_BOUNDS)
    name,identity=opt.study_identity(a,{'fingerprint':'f'*64})
    campaign={k:getattr(a,k) for k in ('seed','fee_bps','slippage_bps','min_total_trades','min_trades_per_fold','n_splits','embargo','holdout_fraction')}
    campaign['arms']={p4.arm_key(a):{'budget':a.n_trials}}
    monkeypatch.setattr(p4,'read_campaign',lambda *a:(campaign,'campaign'))
    a.campaign_manifest=tmp_path/'campaign';a.campaign_identity='campaign'
    identity.update(campaign_sha256='campaign',arm=p4.arm_key(a))
    monkeypatch.setattr(opt,'list_study_names',lambda *a:['other'])
    monkeypatch.setattr(opt,'_study_names',lambda *a:['other'])
    monkeypatch.setattr(opt,'read_study_attrs',lambda *a:{'identity':identity})
    a.study_db=tmp_path/'unit.db';a.study_db.write_bytes(b'unit')
    a.study_name=name
    with pytest.raises(ValueError,match='second study|already.*arm'):
        p4.campaign_arm(a)


def test_b8_candidate_requires_freeze_before_io():
    with pytest.raises(ValueError,match='freeze-params'):
        p4.configure(cli_args('--dry-run','--holdout-look','candidate:'+'a'*64,'--bar-type',BT))


def test_b8_freeze_source_db_outside_phase4_refused(tmp_path,monkeypatch):
    spec=tmp_path/'frozen.json';spec.write_text(json.dumps({'study_db':str(tmp_path/'legacy.db'),'budget':200,'study':'B0','trial':0}))
    a=SimpleNamespace(freeze_params=spec)
    monkeypatch.setattr(p4,'completed_candidate',lambda *a:pytest.fail('read outside Phase 4 DB'))
    with pytest.raises(ValueError,match='Phase 4|phase4'):
        p4.load_frozen(a,[])


def test_host9_component_fits_not_whole_run_ratios():
    rows=[]
    for trades,bars,days in [(100,10,1),(200,20,1),(400,30,1)]:
        rows.append({'trade_count':trades,'verified_trade_count':trades,'bar_counts':{BT:bars},'build_seconds':2+.01*trades,
            'cache_bytes':100+10*bars+50*days,'disk_components':{'fixed_bytes':100,'bar_bytes':10*bars,'day_bytes':50*days,'finalised_days':days},
            'peak_rss_bytes':100,'replay_seconds':{BT:{m:bars*.1 for m in bench.MODES}},
            'cache_open':{BT:{'hash_seconds':0.,'hash_bytes':100.,'digest_seconds':0.,'bars':bars,'other_seconds':0.}}})
    r=bench.extrapolate(rows,10000,{BT:1000},full_days=10)
    assert r['build_wall_seconds']==pytest.approx(102)
    assert r['disk_bytes']==pytest.approx(10600)
    assert r['conservative_upper_bound']['build_wall_seconds']>r['build_wall_seconds']
    assert r['phase3_empirical_anchor']['wall_seconds']==[8063,8804]


def test_host10_no_memmap_scalar_dispatch_in_update_materialisation(tmp_path,monkeypatch):
    ds=paired(tmp_path)
    monkeypatch.setattr(tfc,'pinned_root',lambda _:ds.root_hash)
    monkeypatch.setattr(tfc,'checked_write',lambda p:Path(p))
    paths,_=tfc.build(ds,tmp_path/'cache',start='2023-01-01',end='2023-01-04')
    cache=tfc.CachedTradeUpdates(paths[BT])
    original=np.memmap.__getitem__;calls=[]
    def access(self,index):
        calls.append(index);return original(self,index)
    monkeypatch.setattr(np.memmap,'__getitem__',access)
    update=cache[0]
    for array in update.footprint_levels:
        list(array)
        assert np.shares_memory(array,cache.arrays['fp_0']) or len(array)==0 or np.shares_memory(array,cache.arrays['fp_1']) or np.shares_memory(array,cache.arrays['fp_2'])
        assert not array.flags.writeable
    assert not calls


def approval_record(monkeypatch,scope='campaign',campaign='c'*64,argv=None,changes=None):
    approvals=importlib.import_module('scripts.phase4_approvals')
    state={'base':'b'*40,'sha256':'s'*64,'dirty':False}
    monkeypatch.setattr(approvals,'source_state',lambda:state)
    record={'scope':scope,'plan_sha256':approvals.PLAN_SHA256,'command_argv':argv or ['python','command.py','--budget','200'],
            'base_commit':state['base'],'source_state':state,'created_utc':'2026-11-05T00:00:00Z'}
    if scope=='campaign':
        record['campaign_sha256']=campaign
    record.update(changes or {})
    raw=policy.canonical_bytes(record);digest=hashlib.sha256(raw).hexdigest()
    root=approvals.approval_root();root.mkdir(parents=True,exist_ok=True)
    (root/(digest+'.json')).write_bytes(raw)
    return approvals,record,digest


def test_b4_valid_bound_record_and_host_gates(monkeypatch):
    approvals,record,digest=approval_record(monkeypatch)
    # Artifact validation is exercised with complete synthetic evidence in round 4.
    monkeypatch.setattr(approvals,'verify_campaign_artifacts',lambda *a:None)
    gates={key:record[key] for key in ('plan_sha256','campaign_sha256','base_commit','source_state','created_utc')}
    gates.update({key:'pass' for key in approvals.GATES})
    gatefile=approvals.approval_root()/('gates-'+record['campaign_sha256']+'.json')
    gatefile.write_bytes(policy.canonical_bytes(gates))
    argv=[*record['command_argv'],'--owner-approved',digest]
    result=p4.require_gates(digest,record['campaign_sha256'],argv)
    assert result['sha256']==digest and len(result['gates_sha256'])==64
    gates['source_state']={'changed':True};gatefile.write_bytes(policy.canonical_bytes(gates))
    with pytest.raises(ValueError,match='source state'):
        p4.require_gates(digest,record['campaign_sha256'],argv)


@pytest.mark.parametrize('field,value,match',[
    ('plan_sha256','x'*64,'plan'),('campaign_sha256','d'*64,'campaign'),('source_state',{},'source'),
    ('base_commit','x'*40,'source'),('scope','full_history_tfc','scope'),
    ('command_argv',['python','different.py'],'argv'),('created_utc','bad','UTC')])
def test_b4_binding_mismatches_refused(monkeypatch,field,value,match):
    approvals,record,digest=approval_record(monkeypatch,changes={field:value})
    with pytest.raises(ValueError,match=match):
        approvals.verify_approval('campaign',digest,'c'*64,['python','command.py','--budget','200'])


def test_b4_content_hash_and_full_build_binding(monkeypatch):
    argv=['python','scripts/trade_feature_cache.py','build','--full-history']
    approvals,record,digest=approval_record(monkeypatch,scope='full_history_tfc',argv=argv)
    assert tfc.validate_build_range(None,None,True,digest,[*argv,'--owner-approved',digest])=='production'
    path=approvals.approval_root()/(digest+'.json');path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='hash'):
        tfc.validate_build_range(None,None,True,digest,argv)


def test_b2_default_roots_and_window_reference():
    assert tfc.cache_destination(None,'bounded_verification')==tfc.ROOT/'output/phase4/bounded_cache'
    assert tfc.cache_destination(None,'production')==tfc.ROOT/'output/optimize/trade_cache'
    with pytest.raises(ValueError,match='production'):
        tfc.cache_destination(tfc.ROOT/'output/phase4/bounded_cache','production')
    refs=json.loads(Path('tests/fixtures/phase4_reference_digests.json').read_text())
    assert tfc.reference_for({'start':'2021-05-19','end':'2021-05-20'},refs,'all')==refs['bounded_all']
    with pytest.raises(ValueError,match='no unique matching'):
        tfc.reference_for({'start':'2023-10-16','end':'2023-10-17'},refs,'all')


def test_b2_attach_ignores_bounded_and_wrong_dataset(tmp_path,monkeypatch):
    from scripts.phase4_benchmark import DATASET_ID
    paths={}
    for folder,mode,dataset in [('bounded','bounded_verification',DATASET_ID),('wrong','production','other'),('right','production',DATASET_ID)]:
        paths[folder]={};entries={}
        for bt in tfc.PRODUCTION_LANES:
            raw=policy.canonical_bytes({'bar_type':bt,'mode':mode,'dataset_id':dataset,'root_hash':'root','files':{}})
            key=hashlib.sha256(raw).hexdigest()
            path=tmp_path/folder/(bt+'-'+key);path.mkdir(parents=True)
            (path/'manifest.json').write_bytes(raw);entries[path.name+'/manifest.json']=key;paths[folder][bt]=path
        if mode=='production':
            (tmp_path/folder/'COMPLETE.json').write_text(json.dumps({'manifests':entries}))
    seen=[]
    class Cache:
        def __init__(self,path,*a,**kw):
            seen.append(path);self.identity={}
    monkeypatch.setattr(tfc,'CachedTradeUpdates',Cache)
    a=SimpleNamespace(trade_cache=tmp_path,bar_type=BT)
    p4.attach_cache(a,[])
    assert seen==[paths['right'][BT]]


def test_b1_report_checks_baseline_campaign_before_trials(tmp_path,monkeypatch,capsys):
    from test_optimize_orderflow import args_for
    a=args_for(tmp_path,phase4=True,campaign_identity='campaign')
    _,identity=opt.study_identity(a,{'fingerprint':'f'*64})
    baseline={**identity,'campaign_sha256':'foreign'}
    campaign={key:identity[key] for key in ('seed','fee_bps','slippage_bps','min_total_trades','min_trades_per_fold','n_splits','embargo','holdout_fraction')}
    campaign['arms']={f'BTCUSDT:{a.bar_type}:off':{'budget':a.n_trials,'kind':'study'}}
    db=p4.ROOT/'output/optimize/phase4/unit-never-created.db'
    monkeypatch.setattr(p4,'read_campaign',lambda *args:(campaign,'campaign'))
    monkeypatch.setattr(report,'report_snapshots',lambda *args:[{'path':str(db)}])
    monkeypatch.setattr(opt,'read_study_attrs',lambda path,name:{'identity':identity if name=='s' else baseline})
    monkeypatch.setattr(report.diagnose,'find_cached_bars',lambda *args:(None,{'start':'2023-01-01','end':'2023-02-01','symbol':'BTCUSDT','bar_type':a.bar_type}))
    monkeypatch.setattr(opt,'CachedBars',lambda *args:[])
    monkeypatch.setattr(p4,'attach_cache',lambda *args:None)
    monkeypatch.setattr(opt,'read_study_trials',lambda *args:pytest.fail('baseline trials read before campaign binding'))
    with pytest.raises(SystemExit) as error:
        report.main(['--study-db',str(db),'--study-name','s','--baseline-study','b','--campaign-manifest','unit'])
    assert error.value.code==2 and 'campaign' in capsys.readouterr().err


@pytest.mark.parametrize('wrong',['campaign','budget'])
def test_b1_candidate_holdout_binds_source_b0(tmp_path,monkeypatch,wrong):
    spec=tmp_path/'freeze.json';spec.write_text(json.dumps({'study_db':str(p4.ROOT/'output/optimize/phase4/unit-never-created.db'),
        'study':'B0','trial':0,'budget':3 if wrong=='budget' else 200}))
    a=cli_args('--trade-signals','framing_trades','--freeze-params',str(spec),'--holdout-look','candidate:'+'a'*64)
    a.symbol,a.bar_type='BTCUSDT',BT;a.campaign_manifest=tmp_path/'campaign';a.n_trials=1;a.trade_cache_identity={}
    campaign={key:getattr(a,key) for key in ('seed','fee_bps','slippage_bps','min_total_trades','min_trades_per_fold','n_splits','embargo','holdout_fraction')}
    campaign['arms']={p4.arm_key(a):{'kind':'candidate','budget':1},'BTCUSDT:'+BT+':off':{'kind':'study','budget':200}}
    monkeypatch.setattr(p4,'read_campaign',lambda *args:(campaign,'campaign'))
    monkeypatch.setattr(p4,'read_candidate',lambda *args:{'arm':p4.arm_key(a),'campaign_sha256':'campaign','trade_cache':{},'source_b0':{'trial':0,'complete':200}})
    identity={'trade_signals':'disabled','campaign_sha256':'foreign' if wrong=='campaign' else 'campaign'}
    monkeypatch.setattr(opt,'read_study_attrs',lambda *args:{'identity':identity})
    monkeypatch.setattr(p4,'completed_candidate',lambda *args:pytest.fail('source trials read before campaign/budget binding'))
    with pytest.raises(ValueError,match='campaign|budget'):
        p4.holdout_look(a,[])


def test_b4_full_verify_requires_host_record_before_cache_reads(monkeypatch,capsys):
    monkeypatch.setattr(tfc,'CachedTradeUpdates',lambda *a,**kw:pytest.fail('full verification opened cache'))
    with pytest.raises(SystemExit) as error:
        tfc.main(['verify','--full-history','--owner-approved','arbitrary'])
    assert error.value.code==2 and 'approval' in capsys.readouterr().err


def test_host10_framing_replay_does_not_materialise_unused_levels(tmp_path,monkeypatch):
    from orderflow_system.config.settings import TradeFlowConfig,parse_trade_signals,get_btcusd_config
    from orderflow_system.data.models import FootprintLevel
    ds=paired(tmp_path)
    monkeypatch.setattr(tfc,'pinned_root',lambda _:ds.root_hash)
    monkeypatch.setattr(tfc,'checked_write',lambda p:Path(p))
    paths,_=tfc.build(ds,tmp_path/'cache',start='2023-01-01',end='2023-01-04')
    cache=tfc.CachedTradeUpdates(paths[BT]);bars=cache.bars()
    cfg=TradeFlowConfig(signals=parse_trade_signals('framing_trades'))
    original=FootprintLevel.__init__;calls=[]
    def initialise(self,*args,**kw):
        calls.append(1);original(self,*args,**kw)
    monkeypatch.setattr(FootprintLevel,'__init__',initialise)
    opt.run_backtest(bars,get_btcusd_config(),
                     .002,bars[0].bar_type,trade_updates=cache,trade_config=cfg)
    assert not calls


def test_host10_lazy_footprint_matches_eager_for_extreme_integer_levels():
    from dataclasses import fields
    from orderflow_system.analytics.footprint import FootprintBar,FootprintEngine
    from test_pipeline import synth_bars
    bar=synth_bars(1)[0]
    maximum=np.iinfo(np.int64).max
    arrays=(np.array([1,100001,maximum],dtype='<i8'),np.array([0,maximum,13],dtype='<i8'),np.array([19,1,maximum],dtype='<i8'))
    lazy=tfc.CachedFootprintLevels(arrays).to_footprint(bar,FootprintEngine())
    eager=FootprintEngine().build_from_levels(bar,arrays)
    assert all(getattr(lazy,field.name)==getattr(eager,field.name) for field in fields(FootprintBar))


def test_b7_atomic_guard_and_activation_attr(tmp_path,monkeypatch):
    import sys
    a=cli_args('--dry-run');a.symbol,a.bar_type='BTCUSDT',BT
    a.study_db=tmp_path/'unit.db';a.phase4_implied=True;a.approval_sha256='approval';a.gates_sha256='gates'
    calls=[]
    study=SimpleNamespace(user_attrs={},trials=[])
    study.set_user_attr=lambda k,v:study.user_attrs.__setitem__(k,v)
    fake=SimpleNamespace(storages=SimpleNamespace(RDBStorage=lambda *a,**kw:calls.append('schema')),
        create_study=lambda **kw:calls.append('create') or study,
        samplers=SimpleNamespace(TPESampler=lambda **kw:None),pruners=SimpleNamespace(NopPruner=lambda:None))
    monkeypatch.setitem(sys.modules,'optuna',fake)
    monkeypatch.setattr(opt,'library_versions',lambda *a:{})
    def admission(args,name,**kw):
        assert kw['under_init_lock'] is True
        calls.append('guard')
    monkeypatch.setattr(p4,'campaign_arm',admission)
    identity={'phase':'phase4'}
    opt.open_study(a,'unit',identity)
    assert calls==['schema','guard','create']
    assert study.user_attrs['phase4_implied'] is True
    assert 'phase4_implied' not in study.user_attrs['identity']

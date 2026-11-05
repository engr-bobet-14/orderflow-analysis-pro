"""Last host fix round: targeted regressions, recorded red before fixes."""
import base64
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import phase4_report as report, phase4_approvals as approvals
from scripts import phase4_campaign as campaign, phase4_studies as studies
from scripts import trade_feature_cache as tfc, phase4_benchmark as bench
from scripts import optimize_orderflow as opt, phase4_policy as policy
from tests.test_phase4_cache import paired, BT


@pytest.fixture(autouse=True)
def account_home(tmp_path,monkeypatch):
    import pwd
    monkeypatch.setattr(pwd,'getpwuid',lambda uid:SimpleNamespace(pw_dir=str(tmp_path/'account')))


def payload(returns):
    return {'attrs':{'daily_returns_search':{'start':str(returns.index[0].date()),'n':len(returns),
        'blown':bool(returns.attrs.get('blown',False)),
        'base64':base64.b64encode(returns.to_numpy(dtype='<f4').tobytes()).decode()}}}


def test_rev01_blown_trial_in_cscv_and_genuine_calendar_mismatch(monkeypatch):
    days=pd.date_range('2023-01-01',periods=16,tz='UTC')
    bars=[SimpleNamespace(timestamp_ms=int(d.timestamp()*1000)) for d in days]
    full=pd.DataFrame({'exit_time':days,'net_pnl':[-100.,-6000.]+[0.]*14})
    monkeypatch.setattr(report.diagnose,'scored_only',lambda *a:full)
    returns=report.search_returns(full,bars,[])
    assert returns.iloc[0]==pytest.approx(-100/opt.INITIAL_CAPITAL_USD)
    assert returns.iloc[1]==-1. and (returns.iloc[2:]==0).all()
    assert returns.attrs['blown'] is True
    trials=[payload(returns),payload(pd.Series(np.tile([.01,.02],8),index=days)),
            payload(pd.Series(np.tile([-.01,.01],8),index=days))]
    matrix=report.daily_matrix(trials)
    assert np.isfinite(matrix).all() and report.cscv_pbo(matrix,blocks=2)['n_candidates']==3
    trials[-1]['attrs']['daily_returns_search']['start']='2023-01-02'
    with pytest.raises(ValueError,match='calendars differ'):report.daily_matrix(trials)


def test_rev01_nonfinite_payload_is_not_a_calendar_error():
    trial=payload(pd.Series([np.nan,0.],index=pd.date_range('2023-01-01',periods=2)))
    with pytest.raises(ValueError,match='finite'):
        report.daily_matrix([trial,trial])


def record(monkeypatch,scope='full_history_tfc'):
    state={'base':'b'*40,'sha256':'s'*64,'dirty':False}
    monkeypatch.setattr(approvals,'source_state',lambda:state)
    value={'scope':scope,'plan_sha256':approvals.PLAN_SHA256,'base_commit':state['base'],
           'source_state':state,'created_utc':'2026-11-05T00:00:00Z','command_argv':['python','build']}
    raw=policy.canonical_bytes(value);key=hashlib.sha256(raw).hexdigest()
    import pwd
    root=approvals.approval_root()
    assert root==Path(pwd.getpwuid(os.getuid()).pw_dir)/'.orderflow-approvals/phase4'
    root.mkdir(parents=True,mode=0o700)
    path=root/(key+'.json');path.write_bytes(raw);path.chmod(0o600)
    return root,path,key,value


def test_rev02_ignores_home(monkeypatch,tmp_path):
    monkeypatch.setenv('HOME',str(tmp_path/'forged'))
    assert approvals.approval_root()==tmp_path/'account/.orderflow-approvals/phase4'


@pytest.mark.parametrize('target',['root','record'])
def test_rev02_writable_approval_refused(monkeypatch,target):
    root,path,key,_=record(monkeypatch)
    (root if target=='root' else path).chmod(0o777)
    with pytest.raises(ValueError,match='writ|permission'):
        approvals.verify_approval('full_history_tfc',key,argv=['python','build'])


def test_rev02_wrong_owner_refused(monkeypatch):
    root,path,key,_=record(monkeypatch)
    monkeypatch.setattr(approvals.os,'getuid',lambda:os.stat(root).st_uid+1)
    with pytest.raises(ValueError,match='own|uid'):
        approvals.verify_approval('full_history_tfc',key,argv=['python','build'])


def test_rev02_consumed_full_build_record_refused(monkeypatch):
    root,path,key,_=record(monkeypatch)
    verified=approvals.verify_approval('full_history_tfc',key,argv=['python','build'])
    approvals.consume_approval(verified)
    assert (root/(key+'.consumed')).is_file()
    with pytest.raises(ValueError,match='consum|used'):
        tfc.validate_build_range(None,None,True,key,['python','build'])


def test_rev03_cache_open_timing_and_scaling(tmp_path,monkeypatch):
    ds=paired(tmp_path)
    monkeypatch.setattr(tfc,'pinned_root',lambda _:ds.root_hash)
    monkeypatch.setattr(tfc,'checked_write',lambda p:Path(p))
    paths,_=tfc.build(ds,tmp_path/'cache',start='2023-01-01',end='2023-01-04')
    cache=tfc.CachedTradeUpdates(paths[BT]);timing=cache.open_timings
    assert timing['total_seconds']>=timing['hash_seconds']+timing['digest_seconds']>=0
    assert timing['hash_bytes']==sum((cache.path/name).stat().st_size for name in cache.manifest['files'])
    rows=[]
    for i,trades in enumerate([100,200,400]):
        n=10+i*10
        rows.append({'day':str(i),'trade_count':trades,'verified_trade_count':trades,'build_seconds':2+.01*trades,
            'bar_counts':{'fixed_tick_imbalance':n},'cache_bytes':100+10*n+50,
            'disk_components':{'fixed_bytes':100,'bar_bytes':10*n,'day_bytes':50,'finalised_days':1},
            'peak_rss_bytes':100,'replay_seconds':{'fixed_tick_imbalance':{m:.1*n for m in bench.MODES}},
            'cache_open':{'fixed_tick_imbalance':{'hash_seconds':1.,'hash_bytes':100.,'digest_seconds':2.,'bars':10,'other_seconds':.5,'total_seconds':3.5}}})
    r=bench.extrapolate(rows,10000,{'fixed_tick_imbalance':1000},full_days=10)
    opening=r['cache_open_seconds']['fixed_tick_imbalance']
    assert opening==pytest.approx(r['disk_bytes']/100+200+.5)
    assert r['study_wall_seconds']['fixed_tick_imbalance']['absorption_levels']==pytest.approx(100*r['trial_seconds']['fixed_tick_imbalance']['absorption_levels']+opening)
    assert r['campaign_open_seconds']==pytest.approx(5*opening) # one A, three feature studies, one P2
    assert not r['resource_gates']['arms_under_3x']


def test_rev03_ablation_hash_checks_are_per_bar_type():
    import ast,inspect
    tree=ast.parse(inspect.getsource(campaign.ablation))
    modes=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='mode')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='protected_state' for n in ast.walk(modes))


def test_rev04_partial_production_publish_not_attachable(tmp_path,monkeypatch):
    from tests.paired_fixture import make_trades,write_paired
    from orderflow_system.data.paired_dataset import PairedDataset
    other='fixed_tick_imbalance'
    directory,root=write_paired(tmp_path,make_trades(days=3,per_day=80),{bt:[4,41,90,160,200] for bt in tfc.PRODUCTION_LANES})
    ds=PairedDataset.open(directory,root)
    monkeypatch.setattr(tfc,'pinned_root',lambda _:root)
    monkeypatch.setattr(tfc,'checked_write',lambda p:Path(p))
    monkeypatch.setattr(tfc,'validate_build_range',lambda *a:'production')
    monkeypatch.setattr(tfc,'require_source_mode',lambda *a:policy.source_state()|{'dirty':False})
    monkeypatch.setattr(tfc,'verify_source',lambda *a:None)
    monkeypatch.setattr(tfc,'production_preflight',lambda *a,**kw:{'protected_state':{'unit':'same'}})
    monkeypatch.setattr(tfc,'protected_state',lambda:{'unit':'same'})
    monkeypatch.setattr(tfc,'open_registered',lambda *a:ds)
    monkeypatch.setattr(approvals,'consume_approval',lambda *a:None,raising=False)
    monkeypatch.setattr(approvals,'verify_approval',lambda *a,**kw:{'sha256':'unit','record':{}})
    original=Path.rename;renames=[]
    def fail_second(self,target):
        renames.append(target)
        if len(renames)==2:raise OSError('injected second rename failure')
        return original(self,target)
    monkeypatch.setattr(Path,'rename',fail_second)
    dest=tmp_path/'production'
    with pytest.raises(OSError,match='injected'):tfc.build(ds,dest,full_history=True,owner_approved='unit')
    assert len(list(dest.glob('*/manifest.json')))==1
    args=SimpleNamespace(trade_cache=dest,bar_type=BT,dataset_id=ds.dataset_id)
    with pytest.raises(ValueError,match='complete|matching cache'):
        studies.attach_cache(args,None)
    partial=next(dest.glob('*/manifest.json')).parent
    with pytest.raises(ValueError,match='complete|COMPLETE'):
        tfc.CachedTradeUpdates(partial)


def test_rev05_campaign_seal_precedes_manifest_and_half_seal_errors(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'checked_write',lambda p:Path(p))
    snapshots=[]
    for i in range(3):
        p=tmp_path/f'legacy{i}';p.write_bytes(bytes([i]));snapshots.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    dest=tmp_path/'campaign.json';original=os.replace;calls=[]
    def replace(src,dst):
        calls.append(str(dst))
        if Path(dst)==dest:assert Path(str(dest)+'.sha256').is_file()
        return original(src,dst)
    monkeypatch.setattr(campaign.os,'replace',replace)
    campaign.register(dest,snapshots)
    assert calls==[str(dest)+'.sha256',str(dest)]
    dest.unlink()
    with pytest.raises(ValueError,match='half.sealed|incomplete'):
        campaign.register(dest,snapshots)


def test_rev05_empty_freeze_and_missing_source_identity(monkeypatch,tmp_path):
    monkeypatch.setattr(opt,'read_study_trials',lambda *a:[])
    with pytest.raises(ValueError,match='^no feasible trial$'):
        campaign.freeze_baseline(tmp_path/'db','B0',200,tmp_path/'out')
    monkeypatch.setattr(opt,'read_study_attrs',lambda *a:{})
    with pytest.raises(ValueError,match='^study has no stored identity$'):
        studies.source_b0(SimpleNamespace(),{'study_db':'db','study':'B0','budget':200})


def test_rev05_campaign_cli_parser_error(monkeypatch,capsys):
    monkeypatch.setattr(campaign,'freeze_baseline',lambda *a:(_ for _ in ()).throw(ValueError('no feasible trial')))
    with pytest.raises(SystemExit) as error:campaign.main(['freeze'])
    assert error.value.code==2 and 'error: no feasible trial' in capsys.readouterr().err


def test_rev04_complete_marker_required_by_reader_and_verify(tmp_path,monkeypatch,capsys):
    from tests.paired_fixture import make_trades,write_paired
    from orderflow_system.data.paired_dataset import PairedDataset
    directory,root=write_paired(tmp_path,make_trades(days=3,per_day=80),{bt:[4,41,90,160,200] for bt in tfc.PRODUCTION_LANES})
    ds=PairedDataset.open(directory,root)
    monkeypatch.setattr(tfc,'pinned_root',lambda _:ds.root_hash)
    monkeypatch.setattr(tfc,'checked_write',lambda p:Path(p))
    monkeypatch.setattr(tfc,'validate_build_range',lambda *a:'production')
    monkeypatch.setattr(tfc,'require_source_mode',lambda *a:policy.source_state()|{'dirty':False})
    monkeypatch.setattr(tfc,'verify_source',lambda *a:None)
    monkeypatch.setattr(tfc,'production_preflight',lambda *a,**kw:{'protected_state':{'unit':'same'}})
    monkeypatch.setattr(tfc,'protected_state',lambda:{'unit':'same'})
    monkeypatch.setattr(tfc,'open_registered',lambda *a:ds)
    monkeypatch.setattr(approvals,'verify_approval',lambda *a,**kw:{'sha256':'unit','record':{}})
    monkeypatch.setattr(approvals,'consume_approval',lambda *a:None)
    paths,_=tfc.build(ds,tmp_path/'production',full_history=True,owner_approved='unit')
    cache=tfc.CachedTradeUpdates(paths[BT],production=True)
    marker=paths[BT].parent/'COMPLETE.json'
    assert json.loads(marker.read_text())['manifests'][paths[BT].name+'/manifest.json']==cache.identity['manifest_sha256']
    marker.unlink()
    with pytest.raises(SystemExit) as exc:
        tfc.main(['verify','--full-history','--owner-approved','unit','--out',str(marker.parent),'--dataset-id',ds.dataset_id])
    assert exc.value.code==2 and 'COMPLETE' in capsys.readouterr().err


def test_rev05_interrupted_campaign_publish_is_clear(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'checked_write',lambda p:Path(p))
    snapshots=[]
    for i in range(3):
        p=tmp_path/f'legacy{i}';p.write_bytes(bytes([i]));snapshots.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    dest=tmp_path/'campaign.json';original=os.replace
    def fail_manifest(src,dst):
        if Path(dst)==dest:raise OSError('injected manifest publish failure')
        return original(src,dst)
    monkeypatch.setattr(campaign.os,'replace',fail_manifest)
    with pytest.raises(OSError,match='injected'):campaign.register(dest,snapshots)
    assert not dest.exists() and Path(str(dest)+'.sha256').exists()
    with pytest.raises(ValueError,match='half.sealed'):campaign.register(dest,snapshots)
    assert not list(tmp_path.glob('.campaign-*'))


def test_rev01_objective_stores_blown_payload(monkeypatch):
    from test_pipeline import synth_bars
    from test_optimize_orderflow import params
    from orderflow_system.config.settings import TradeSignalConfig
    bars=synth_bars(100,step_ms=86400000)
    args=SimpleNamespace(symbol='BTCUSDT',n_splits=5,embargo=.02,holdout_fraction=.2,
        min_total_trades=105,min_trades_per_fold=11,fee_bps=5.,slippage_bps=1.,phase4=True,
        trade_signal_config=TradeSignalConfig(),space={k:(v,v,opt.SEARCH_SPACE_BOUNDS[k][2]) for k,v in params().items()})
    n_search,folds=opt.split_timeline(bars,args)
    # Force a bankruptcy in the daily diagnostic source without changing fold scoring.
    original=report.search_returns
    def blown_returns(full,search,folds):
        days=pd.to_datetime([b.timestamp_ms for b in search],unit='ms',utc=True).floor('D').unique()
        bankrupt=pd.DataFrame({'exit_time':days,'net_pnl':[-2*opt.INITIAL_CAPITAL_USD]+[0.]*(len(days)-1)})
        with monkeypatch.context() as m:
            m.setattr(report.diagnose,'scored_only',lambda *a:bankrupt)
            return original(bankrupt,search,folds)
    monkeypatch.setattr(report,'search_returns',blown_returns)
    class Trial:
        attrs={}
        def suggest_int(self,name,lo,hi):return lo
        def suggest_float(self,name,lo,hi,**kw):return lo
        def set_user_attr(self,key,value):self.attrs[key]=value
    trial=Trial();opt.objective(trial,bars[:n_search],folds,args)
    stored=trial.attrs['daily_returns_search']
    assert stored['blown'] is True
    values=np.frombuffer(base64.b64decode(stored['base64']),dtype='<f4')
    assert values[0]==-1. and (values[1:]==0.).all()

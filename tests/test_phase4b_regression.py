"""Synthetic fixtures for the host-only digest/protection proof tool."""
import hashlib
import importlib
from pathlib import Path

import pytest

from tests.test_optimize_orderflow import params
from scripts import phase4_studies as p4
from orderflow_system.config.settings import TradeSignalConfig


def test_regression_local_paths_never_resolve_original_checkout(tmp_path):
    regression=importlib.import_module('scripts.phase4b_regression')
    path=regression.checkout_path('/another/checkout/output/phase4/frozen/tick.json',tmp_path)
    assert path==tmp_path/'output/phase4/frozen/tick.json'
    with pytest.raises(ValueError):regression.checkout_path('../../outside',tmp_path)


def test_regression_protection_paths_include_notebooks_and_worktree_root_files(tmp_path):
    regression=importlib.import_module('scripts.phase4b_regression')
    assert regression.checkout_path('/original/notebooks/research.ipynb',tmp_path)==tmp_path/'notebooks/research.ipynb'
    assert regression.checkout_path('/original/.claude/worktrees/prebuild/README.md',tmp_path)==tmp_path/'README.md'
    assert regression.checkout_path(tmp_path/'README.md',tmp_path)==tmp_path/'README.md'


def test_regression_protected_synthetic_fixture_and_tampering(tmp_path):
    regression=importlib.import_module('scripts.phase4b_regression')
    path=tmp_path/'scripts/frozen.py';path.parent.mkdir();path.write_bytes(b'frozen')
    baseline={'scripts/frozen.py':hashlib.sha256(b'frozen').hexdigest()}
    assert regression.verify_protected(tmp_path,baseline)=={'checked':1,'unchanged':True}
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='protected'):regression.verify_protected(tmp_path,baseline)


def test_regression_36_legacy_and_8_stageb_digests_exact_stored_flags():
    regression=importlib.import_module('scripts.phase4b_regression')
    calls=[]
    legacy=[p4.LegacyReplay(params(),TradeSignalConfig(absorption_levels=i%6==1,
        exhaustion_contrarian='bonus' if i%6==2 else 'required' if i%6==3 else 'off',
        framing_profile_source='trades' if i%6 in (4,5) else 'bar_vwap'),f'a{i}') for i in range(36)]
    frozen=[p4.FrozenComparator({**params(),'absorption_min_delta_pct':.2525 if i<4 else .5423},f'b{i}')
        for i in range(8)]
    def replay(value):
        calls.append(value)
        return {'replay_digest':value.replay_digest}
    result=regression.verify_replays(legacy,frozen,replay,replay)
    assert result=={'stageA':36,'stageB':8,'byte_identical':True}
    assert calls==legacy+frozen
    with pytest.raises(ValueError,match='digest'):
        regression.verify_replays(legacy,frozen,lambda v:{'replay_digest':'wrong'},replay)


def test_search_prefix_complete_synthetic_cache_open_path_and_tampered_holdout(tmp_path,monkeypatch):
    from scripts import trade_feature_cache as tfc, optimize_orderflow as opt
    from tests.test_phase4_cache import paired,BT
    from types import SimpleNamespace
    ds=paired(tmp_path)
    monkeypatch.setattr(tfc,'pinned_root',lambda _:ds.root_hash)
    monkeypatch.setattr(tfc,'checked_write',lambda p:Path(p))
    paths,_=tfc.build(ds,tmp_path/'cache',start='2023-01-01',end='2023-01-04')
    path=paths[BT]
    settings=SimpleNamespace(holdout_fraction=.2,n_splits=2,embargo=.02)
    prefix=p4.SearchPrefix.open(path/'bars.bin',BT,path,settings)
    assert len(prefix.bars)==4
    accessed=[];original=tfc.CachedTradeUpdates.__getitem__
    def spy(self,index):
        accessed.append(index);assert index<4
        return original(self,index)
    monkeypatch.setattr(tfc.CachedTradeUpdates,'__getitem__',spy)
    # All three execution consumers use the exact same bounded replay entry point.
    for label in ('objective','S1','report'):
        p4.replay_full_strategy(p4.FullStrategySample(p4.full_strategy_defaults()),prefix)
    assert max(accessed)==3
    with pytest.raises(p4.HoldoutAccessError):prefix.trade_updates[4]
    monkeypatch.setattr(tfc.CachedTradeUpdates,'__getitem__',original)
    # Alter the last (holdout-only) stats byte. The full integrity constructor still refuses it.
    with (path/'stats.npy').open('r+b') as handle:handle.seek(-1,2);handle.write(b'x')
    with pytest.raises(ValueError,match='hash'):
        p4.SearchPrefix.open(path/'bars.bin',BT,path,settings)


def test_search_prefix_actual_objective_s1_and_report_never_index_holdout(tmp_path,monkeypatch):
    monkeypatch.setenv('MPLCONFIGDIR',str(tmp_path/'matplotlib'))
    from types import SimpleNamespace
    from orderflow_system.data.paired_dataset import PairedDataset
    from scripts import trade_feature_cache as tfc, optimize_orderflow as opt, phase4_report as report
    from tests.paired_fixture import make_trades,write_paired
    from tests.test_phase4_cache import BT
    directory,root=write_paired(tmp_path,make_trades(days=3,per_day=80),{BT:list(range(4,201,4))})
    ds=PairedDataset.open(directory,root)
    monkeypatch.setattr(tfc,'pinned_root',lambda _:root)
    monkeypatch.setattr(tfc,'checked_write',lambda p:Path(p))
    paths,_=tfc.build(ds,tmp_path/'cache',start='2023-01-01',end='2023-01-04')
    args=SimpleNamespace(holdout_fraction=.2,n_splits=5,embargo=.02,phase='phase4b',phase4=True,
        symbol='BTCUSDT',space=p4.FULL_STRATEGY_SPACE,fee_bps=5.,slippage_bps=1.,min_trades_per_fold=11,min_total_trades=105)
    prefix=p4.SearchPrefix.open(paths[BT]/'bars.bin',BT,paths[BT],args)
    stop=len(prefix.bars);original=tfc.CachedTradeUpdates.__getitem__;accessed=[]
    def spy(self,index):
        assert index<stop;accessed.append(index);return original(self,index)
    monkeypatch.setattr(tfc.CachedTradeUpdates,'__getitem__',spy)
    values=p4.full_strategy_defaults();attrs={}
    trial=SimpleNamespace(suggest_int=lambda n,*a:values[n],suggest_float=lambda n,*a,**kw:values[n],
        suggest_categorical=lambda n,*a:values[n],set_user_attr=attrs.__setitem__)
    folds=opt.build_chronological_folds(prefix.bars,5,.02)
    value=opt.objective(trial,prefix,folds,args)
    comparator_params=params()
    baseline=opt.run_backtest(prefix.bars,opt.build_instrument_config(comparator_params),
        comparator_params['price_proximity_pct'],prefix.bars[0].bar_type)
    comparator=p4.FrozenComparator(comparator_params,baseline['replay_digest'])
    assert len(p4.run_s1(prefix,comparator)['replays'])==6
    trials=[{'state':'COMPLETE','feasible':True,'number':i,'value':value,'attrs':attrs} for i in range(3)]
    monkeypatch.setattr(report,'campaign_union',lambda *a:{'n_symbol_all_studies':3,'trial_sharpe_std':.1})
    result=report.build_full_strategy_report(prefix,trials,comparator,[],folds)
    assert result['G4b_N'] is False and len(result['parameter_stability'])==46
    assert accessed and max(accessed)==stop-1

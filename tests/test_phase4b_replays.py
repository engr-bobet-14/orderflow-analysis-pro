"""C6, C9 and §5.1: type-separated replay and search-prefix boundaries."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import phase4_studies as p4, phase4_report as report, optimize_orderflow as opt
from orderflow_system.config.settings import TradeSignalConfig
from tests.test_optimize_orderflow import params
from tests.test_pipeline import synth_bars


def sampled():
    return {key:(bounds[0][0] if bounds[2] == 'categorical' else bounds[0])
            for key,bounds in p4.FULL_STRATEGY_SPACE.items()}


def test_c6_full_space_categorical_and_fixed_structure():
    space = p4.FULL_STRATEGY_SPACE
    assert len(space) == 23 and 'absorption_min_delta_pct' not in space
    fake = SimpleNamespace(suggest_int=lambda n,lo,hi:lo, suggest_float=lambda n,lo,hi,**kw:lo,
                           suggest_categorical=lambda n,choices:choices[-1])
    values = opt.suggest_params(fake, space)
    assert values['ts_imbalance_bucket_mult'] == 10
    effective = p4.full_strategy_params(values)
    assert effective['absorption_min_delta_pct'] == opt.instrument_config('BTCUSDT').absorption.min_delta_pct
    cfg = p4.effective_trade_config(SimpleNamespace(phase='phase4b'), effective).signals
    assert cfg.absorption_source == 'trade' and cfg.framing_profile_source == 'trades'
    assert cfg.initiative_imbalance == cfg.exhaustion_contrarian == 'required'
    with pytest.raises(ValueError, match='bonus'):
        p4.effective_trade_config(SimpleNamespace(phase='phase4b', trade_signal_config=TradeSignalConfig(
            initiative_imbalance='bonus')), effective)


def test_c6_neighbours_relative_oracle():
    values = {'f':10., 'log':10., 'i':1, 'z':0., 'clip':2., 'cat':2}
    space = {'f':(0.,20.,'float'), 'log':(1.,100.,'float_log'), 'i':(0,5,'int'),
             'z':(-10.,10.,'float'), 'clip':(2.,3.,'float'), 'cat':((1,2,5,10),None,'categorical')}
    rows = report.neighbours_rel(values, space)
    assert [r['effective_value'] for r in rows] == pytest.approx([9,11,9,11,0,2,-2,2,2,2.2,1,5])
    assert sum(r['status']=='available' for r in rows) == 11


def test_search_prefix_rejects_holdout_and_bounds_iteration():
    bars = synth_bars(10)
    updates = Mock()
    updates.__len__ = Mock(return_value=10)
    updates.__getitem__ = Mock(side_effect=lambda i:i)
    prefix = p4.SearchPrefix(bars, updates, 8)
    assert len(prefix.bars) == len(prefix.trade_updates) == 8
    assert list(prefix.trade_updates) == list(range(8))
    with pytest.raises(p4.HoldoutAccessError):
        prefix.trade_updates[8]
    with pytest.raises(p4.HoldoutAccessError):
        prefix.trade_updates[-1]
    assert [c.args[0] for c in updates.__getitem__.call_args_list] == list(range(8))


@pytest.mark.parametrize('entry', ['objective', 's1', 'report', 'neighbour', 'holdout'])
def test_c6_every_full_builder_materializes_inert_value(entry, monkeypatch):
    bars = synth_bars(30, step_ms=86400000)
    prefix = p4.SearchPrefix(bars, [None]*len(bars), 24)
    seen = []
    build = opt.build_instrument_config
    monkeypatch.setattr(opt,'build_instrument_config',lambda p,s='BTCUSDT':(seen.append(dict(p)) or build(p,s)))
    monkeypatch.setattr(opt,'run_backtest',lambda b,c,p,bt,**kw:{'ledger':None,'replay_digest':'ok',
        'raw_signal_digest':'raw', 'signal_counts':{},'prefix_digests':{23:'ok'}})
    values = p4.FullStrategySample(sampled())
    if entry == 'objective':
        trial = SimpleNamespace(suggest_int=lambda n,lo,hi:lo, suggest_float=lambda n,lo,hi,**kw:lo,
            suggest_categorical=lambda n,choices:choices[0], set_user_attr=lambda *a:None)
        monkeypatch.setattr(opt,'simulate_trades_risk_based',lambda *a,**k:None)
        monkeypatch.setattr(opt,'evaluate_fold',lambda *a:dict(sharpe_ratio=1.,n_trades=30,n_days=1,
            penalized=False,account_blown=False,n_scored_bars=1))
        import pandas as pd
        monkeypatch.setattr(report,'search_returns',lambda *a:pd.Series([1.],index=pd.date_range('2023-01-01',periods=1,tz='UTC')))
        # The normal objective stores the same daily-return payload; its blown attr is supplied.
        def returns(*a):
            r = pd.Series([1.],index=pd.date_range('2023-01-01',periods=1,tz='UTC'));r.attrs['blown']=False;return r
        monkeypatch.setattr(report,'search_returns',returns)
        args = SimpleNamespace(phase='phase4b',phase4=True,symbol='BTCUSDT',space=p4.FULL_STRATEGY_SPACE,
            n_splits=5,min_trades_per_fold=11,min_total_trades=105,fee_bps=5.,slippage_bps=1.)
        opt.objective(trial,prefix,[{}]*5,args)
    elif entry=='holdout':
        import pandas as pd
        from scripts import phase4b_campaign as driver, trade_feature_cache as tfc
        class FullBars(list):
            bar_type=bars[0].bar_type
        full=FullBars(bars)
        prefix.input_paths={'bars':'synthetic','trade':'synthetic','dataset_id':'synthetic'}
        monkeypatch.setattr(opt,'CachedBars',lambda *a,**kw:full)
        monkeypatch.setattr(tfc,'CachedTradeUpdates',lambda *a,**kw:[None]*30)
        trades=pd.DataFrame({'dollar_risk':[1.],'net_pnl':[1.]})
        monkeypatch.setattr(opt,'simulate_trades_risk_based',lambda *a,**kw:trades)
        monkeypatch.setattr(opt,'score_window',lambda *a,**kw:({'account_blown':False},trades))
        result=driver.holdout_replay(prefix,values,'ok',{'embargo':.02},
            {'dataset_id':'synthetic','start_seq':bars[24].seq,'end_seq':bars[-1].seq})
        assert result['replay_identity']['inert_params']=={'absorption_min_delta_pct':seen[-1]['absorption_min_delta_pct']}
    else:
        result = p4.replay_full_strategy(values,prefix)
        assert result['replay_identity']['inert_params'] == {'absorption_min_delta_pct':seen[-1]['absorption_min_delta_pct']}
    assert seen and all('absorption_min_delta_pct' in p for p in seen)


def test_c6_comparator_and_legacy_types_preserve_exact_flags_and_params(monkeypatch):
    seen = []
    monkeypatch.setattr(opt,'run_backtest',lambda *a,**k:(seen.append((a,k)) or {'replay_digest':'digest'}))
    values = {**params(), 'absorption_min_delta_pct':.2525}
    prefix = p4.SearchPrefix(synth_bars(10),[None]*10,8)
    comparator = p4.FrozenComparator(values,'digest')
    assert p4.replay_comparator(comparator,prefix)['replay_digest'] == 'digest'
    assert not seen[-1][1] and seen[-1][0][1].absorption.min_delta_pct == .2525
    legacy = p4.LegacyReplay(values,TradeSignalConfig(absorption_levels=True),'digest')
    p4.replay_legacy(legacy,prefix)
    assert seen[-1][1]['trade_config'].signals == legacy.signals
    assert seen[-1][1]['trade_config'].signals.absorption_source == 'bar'
    for fn,wrong in ((p4.replay_comparator,legacy),(p4.replay_legacy,comparator),
                     (p4.replay_full_strategy,comparator)):
        with pytest.raises(TypeError):fn(wrong,prefix)


def test_c9_s1_six_replays_and_three_benchmarks_are_search_only(monkeypatch):
    prefix = p4.SearchPrefix(synth_bars(30),[None]*30,24)
    calls = []
    def replay(sample, prefix, **kw):
        assert len(prefix.bars) == 24
        calls.append(kw)
        return {'ledger':None,'signal_counts':{'absorption':1},'replay_digest':'digest'}
    monkeypatch.setattr(p4,'replay_full_strategy',replay)
    benchmarks=[]
    monkeypatch.setattr(p4,'benchmark_trial_equivalent',lambda prefix,values,**kw:benchmarks.append(values))
    monkeypatch.setattr(opt,'simulate_trades_risk_based',lambda *a,**k:None)
    from scripts import diagnose_trades as diagnose
    monkeypatch.setattr(diagnose,'add_gross',lambda x:x)
    monkeypatch.setattr(diagnose,'decompose',lambda x:{'gross_expectancy_r':1.,'net_expectancy_r':.5,'n_trades':1})
    result = p4.run_s1(prefix,p4.FrozenComparator(params(),'digest'))
    assert len(result['replays']) == 6 and len(result['trial_equivalent_seconds']) == 3
    assert len(calls) == 6 and len(benchmarks)==4
    assert benchmarks[-1]==p4.full_strategy_probe()
    assert all(r['peak_rss_bytes'] > 0 and r['runtime_seconds'] >= 0 for r in result['replays'].values())


def test_c9_benchmark_is_the_complete_registered_objective(monkeypatch):
    prefix=p4.SearchPrefix(synth_bars(30),[None]*30,24)
    calls=[]
    def objective(trial,prefix,folds,args):
        assert len(prefix.bars)==24 and len(folds)==5
        assert args.phase=='phase4b' and args.min_total_trades==105
        assert args.fee_bps==5. and args.slippage_bps==1.
        assert opt.suggest_params(trial,p4.FULL_STRATEGY_SPACE)==p4.full_strategy_defaults()
        calls.append(1)
    monkeypatch.setattr(opt,'objective',objective)
    p4.benchmark_trial_equivalent(prefix,p4.full_strategy_defaults())
    assert calls==[1]


def test_c9_replay_rss_measurement_resets_between_replays(monkeypatch):
    import psutil
    values=iter([100,200,50,60])
    monkeypatch.setattr(psutil,'Process',lambda:SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=next(values))))
    with p4.replay_measurement() as first:pass
    with p4.replay_measurement() as second:pass
    assert first['peak_rss_bytes']==200 and second['peak_rss_bytes']==60
    assert first['runtime_seconds']>=0 and second['runtime_seconds']>=0

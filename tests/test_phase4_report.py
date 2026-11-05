"""Pre-registered statistics: joint days, gross R and valid neighbours (T4 red first)."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scripts import phase4_report as report


def test_pbo_known_cases_and_single_column():
    noise = np.tile([-.01,.01], 8)
    stable = np.column_stack([noise+.03, noise-.03])
    assert report.cscv_pbo(stable, blocks=2)['pbo'] == 0
    reversing = np.column_stack([noise+np.r_[np.ones(8)*.1,-np.ones(8)*.1],
                                 noise+np.r_[-np.ones(8)*.1,np.ones(8)*.1]])
    assert report.cscv_pbo(reversing, blocks=2)['pbo'] == 1
    with pytest.raises(ValueError, match='two'):
        report.cscv_pbo(stable[:,:1])


def test_joint_bootstrap_reproducible_and_identical_difference_zero():
    day = pd.date_range('2023-01-01', periods=40, tz='UTC')
    trades = pd.DataFrame({'exit_time':day, 'gross_pnl':np.arange(40)-5., 'dollar_risk':np.ones(40)*10})
    a = report.gross_bootstrap(trades, trades, day[0], day[-1])
    b = report.gross_bootstrap(trades, trades, day[0], day[-1])
    assert a == b
    assert a['difference_ci'] == [0.,0.]
    assert a['n_resamples'] == 2000 and a['seed'] == 20261105
    empty = trades.iloc[:0]
    assert report.gross_bootstrap(trades, empty, day[0], day[-1])['status'] == 'indeterminate'


@pytest.mark.parametrize('name,base,expected', [
    ('ts_max_through_bps',0.,[0.,.4]), ('ts_min_side_ratio',.55,[.55,.585]),
    ('lookback_bars',5,[5,7]), ('lookback_bars',30,[28,30]),
    ('ts_level_zone_frac',.5,[.46,.5]),
])
def test_domain_aware_neighbours(name,base,expected):
    rows = report.neighbours({name:base})
    assert [r['effective_value'] for r in rows] == pytest.approx(expected)
    assert sum(r['status'] == 'unavailable_at_bound' for r in rows) == 1
    if name == 'lookback_bars':
        assert all(isinstance(r['effective_value'], int) for r in rows)


def test_cost_stress_hand_computation():
    t = pd.DataFrame({'entry_price':[100.], 'exit_price':[110.], 'quantity':[2.],
                      'side':['buy'], 'dollar_risk':[5.], 'gross_pnl':[20.]})
    grid = report.cost_stress(t)
    assert len(grid['grid']) == 9
    zero = next(r for r in grid['grid'] if r['fee_bps'] == r['slippage_bps'] == 0)
    assert zero['net_expectancy_r'] == 4
    row = next(r for r in grid['grid'] if r['fee_bps'] == 5 and r['slippage_bps'] == 1)
    assert row['net_expectancy_r'] == pytest.approx((20-.042-.0005*(100.01+109.989)*2)/5)


def test_cross_database_dedupe(tmp_path, monkeypatch):
    from scripts import optimize_orderflow as opt
    paths = [tmp_path/'old.db',tmp_path/'copy.db',tmp_path/'phase4.db']
    for p in paths:
        p.write_bytes(b'snapshot')
    monkeypatch.setattr(opt, 'list_study_names', lambda path:['old'] if Path(path).name != 'phase4.db' else ['phase4'])
    monkeypatch.setattr(opt, 'read_study_trials', lambda path,name:[
        {'state':'COMPLETE','number':0,'params':{'x':1},'attrs':{'replay_digest':'abc','raw_daily_sharpe':float(len(name))}}])
    snapshots = [{'path':str(p),'sha256':report.sha256_file(p)} for p in paths]
    result = report.campaign_union(snapshots)
    assert result['n_symbol_all_studies'] == 2
    assert result['per_db'][0]['complete'] == 1
    assert result['duplicate_trials'] == 1


def test_database_union_opens_all_sources_read_only(tmp_path,monkeypatch):
    import sqlite3
    from test_warm_start import make_source_study
    from scripts import optimize_orderflow as opt
    db = tmp_path/'unit.db'
    make_source_study(db,'unit',[{'value':1.,'attrs':{'replay_digest':'a','raw_daily_sharpe':.1}}])
    original = sqlite3.connect
    opens = []
    def connect(path,*a,**kw):
        opens.append(str(path))
        assert str(path).endswith('?mode=ro') and kw.get('uri') is True
        return original(path,*a,**kw)
    monkeypatch.setattr(sqlite3,'connect',connect)
    snapshot = {'path':str(db),'sha256':report.sha256_file(db)}
    assert report.campaign_union([snapshot])['n_symbol_all_studies']==1
    assert opens and all('mode=ro' in p for p in opens)


@pytest.mark.parametrize('failure',['arm_ci','difference_ci','pbo','frequency','top3','total','fold'])
def test_g2_requires_every_registered_condition(failure):
    bootstrap = {'status':'evaluated','arm_ci':[.1,.3],'difference_ci':[.1,.3]}
    trial = {'attrs':{'total_trades':200,'fold_results':[{'n_trades':40} for _ in range(5)]}}
    top,pbo,frequency = [.2,.3,.4],.2,2.
    assert report.gate(bootstrap,trial,top,pbo,frequency)
    if failure in ('arm_ci','difference_ci'):
        bootstrap[failure][0]=0.
    elif failure=='pbo':
        pbo=.5
    elif failure=='frequency':
        frequency=3.1
    elif failure=='top3':
        top[2]=0.
    elif failure=='total':
        trial['attrs']['total_trades']=104
    else:
        trial['attrs']['fold_results'][0]['n_trades']=10
    assert not report.gate(bootstrap,trial,top,pbo,frequency)


def test_p2_gate_does_not_use_pbo_or_top3():
    bootstrap = {'status':'evaluated','arm_ci':[.1,.3],'difference_ci':[.1,.3]}
    trial = {'attrs':{'total_trades':200,'fold_results':[{'n_trades':40}]*5}}
    assert report.gate(bootstrap,trial,[],None,2.,deterministic=True)


def test_objective_daily_storage_and_report_integration(monkeypatch):
    from types import SimpleNamespace
    from test_pipeline import synth_bars
    from test_optimize_orderflow import params
    from orderflow_system.config.settings import TradeSignalConfig
    from scripts import optimize_orderflow as opt
    bars = synth_bars(100,step_ms=86400000)
    args = SimpleNamespace(symbol='BTCUSDT',n_splits=5,embargo=.02,holdout_fraction=.2,
        min_total_trades=105,min_trades_per_fold=11,fee_bps=5.,slippage_bps=1.,phase4=True,
        trade_signal_config=TradeSignalConfig(),space={k:(v,v,opt.SEARCH_SPACE_BOUNDS[k][2]) for k,v in params().items()})
    class Trial:
        def __init__(self):
            self.attrs={}
        def suggest_int(self,name,lo,hi):
            return lo
        def suggest_float(self,name,lo,hi,**kw):
            return lo
        def set_user_attr(self,key,value):
            self.attrs[key]=value
    n_search,folds = opt.split_timeline(bars,args)
    fake = Trial()
    value = opt.objective(fake,bars[:n_search],folds,args)
    payload = fake.attrs['daily_returns_search']
    assert payload['dtype']=='<f4' and payload['n']>=16
    assert payload['blown'] is False
    trials = [{'state':'COMPLETE','number':i,'value':value,'feasible':True,'params':params(),'attrs':fake.attrs} for i in range(3)]
    monkeypatch.setattr(report,'campaign_union',lambda *a:{'n_symbol_all_studies':3,'trial_sharpe_std':.1})
    result = report.build_study_report(args,bars,trials,trials[0],args,[],{'lookback_bars':(5,30,'int')})
    assert result['replay_digest']==fake.attrs['replay_digest']
    assert result['G2'] is False
    assert len(result['top3'])==3
    assert len(result['parameter_stability'])==2
    assert result['bootstrap']['difference_ci']==[0.,0.]


def test_frequency_gate_cannot_dilute_absorption_in_other_counts():
    result = report.signal_multiples({'absorption':100,'initiative':1000},{'absorption':10,'initiative':1000})
    assert result['maximum']==10
    assert result['total']<2


def test_missing_snapshots_refused_before_study_reads(monkeypatch,capsys):
    from scripts import phase4_studies as p4,optimize_orderflow as opt
    monkeypatch.setattr(p4,'read_campaign',lambda *a:({'snapshots':[]},'hash'))
    monkeypatch.setattr(opt,'read_study_attrs',lambda *a:pytest.fail('read study with incomplete multiplicity registration'))
    with pytest.raises(SystemExit) as error:
        report.main(['--study-db','missing.db','--study-name','s','--baseline-study','b','--campaign-manifest','missing.json'])
    assert error.value.code==2 and 'snapshots' in capsys.readouterr().err

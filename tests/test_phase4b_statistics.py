"""C8 / §7: mechanically evaluated gates, bootstrap and strict empirical tails."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from scripts import phase4_report as report, optimize_orderflow as opt, diagnose_trades as diagnose


def trades(r=1., days=40):
    return pd.DataFrame({'exit_time':pd.date_range('2023-01-01',periods=days,tz='UTC'),
                         'dollar_risk':np.ones(days),'gross_pnl':np.ones(days)*(r+.1),'net_pnl':np.ones(days)*r})


def test_c8_net_bootstrap_joint_seed_invalid_and_mean_r_oracle():
    t=trades();negative=trades(-1.)
    a=report.net_bootstrap(t,negative,t.exit_time.iloc[0],t.exit_time.iloc[-1])
    assert a==report.net_bootstrap(t,negative,t.exit_time.iloc[0],t.exit_time.iloc[-1])
    assert a['arm_ci']==[1.,1.] and a['difference_ci']==[2.,2.]
    assert report.net_bootstrap(t,t,t.exit_time.iloc[0],t.exit_time.iloc[-1])['difference_ci']==[0.,0.]
    assert report.net_bootstrap(negative,t,t.exit_time.iloc[0],t.exit_time.iloc[-1])['arm_ci']==[-1.,-1.]
    assert report.net_bootstrap(t,t.iloc[:0],t.exit_time.iloc[0],t.exit_time.iloc[-1])['status']=='indeterminate'
    pair=pd.DataFrame({'exit_time':np.repeat(t.exit_time.to_numpy(),2),'dollar_risk':[1.,10.]*40,
                       'net_pnl':[2.,-10.]*40,'gross_pnl':[2.,-10.]*40})
    assert report.expectancy_r(pair)==.5
    assert report.net_bootstrap(pair,t.iloc[:0].reindex(columns=pair.columns).combine_first(pair),
        t.exit_time.iloc[0],t.exit_time.iloc[-1])['arm_ci']==[.5,.5]


def passing_gate_args():
    return dict(net={'status':'evaluated','arm_ci':[.1,.2],'difference_ci':[.1,.2]},
        gross={'status':'evaluated','arm_ci':[.1,.2]},top3=[.1,.2,.3],pbo=.1,
        dsr={'status':'evaluated','dsr_pass':True,'tail_pass':True},
        folds=[{'sharpe_ratio':1.,'n_trades':30,'penalized':False} for _ in range(5)],
        neighbours=[{'status':'available','net_expectancy_r':.1} for _ in range(46)],
        stress=.1,feasible=True)


@pytest.mark.parametrize('item',range(1,11))
def test_c8_each_of_ten_gates_is_required(item):
    args=passing_gate_args()
    assert report.net_profitability_gates(**args)['G4b_N'] is True
    if item==1:args['net']['arm_ci'][0]=0
    if item==2:args['net']['difference_ci'][0]=0
    if item==3:args['gross']['arm_ci'][0]=0
    if item==4:args['top3'][-1]=-.1
    if item==5:args['pbo']=.5
    if item==6:args['dsr']['tail_pass']=False
    if item==7:args['folds']=[{'sharpe_ratio':float('nan'),'penalized':False,'n_trades':30}]*3+args['folds'][:2]
    if item==8:args['neighbours']=[{'status':'unavailable_at_bound'}]*24+args['neighbours'][:22]
    if item==9:args['stress']=0.
    if item==10:args['feasible']=False
    result=report.net_profitability_gates(**args)
    assert result['gates'][str(item)] is False and result['G4b_N'] is False


@pytest.mark.parametrize('mutation', ['top_short','pbo_none','bootstrap_invalid','penalized_fold',
                                     'neighbour_zero','stress_nan','floor_total'])
def test_c8_indeterminate_and_boundary_gate_cases(mutation):
    args=passing_gate_args()
    if mutation=='top_short':args['top3']=[1.,1.]
    if mutation=='pbo_none':args['pbo']=None
    if mutation=='bootstrap_invalid':args['net']['status']='indeterminate'
    if mutation=='penalized_fold':
        for fold in args['folds'][:3]:fold['penalized']=True
    if mutation=='neighbour_zero':
        for row in args['neighbours']:row['net_expectancy_r']=0.
    if mutation=='stress_nan':args['stress']=float('nan')
    if mutation=='floor_total':
        for fold in args['folds']:fold['n_trades']=20
    assert report.net_profitability_gates(**args)['G4b_N'] is False


def test_c8_fold_three_of_five_and_half_neighbours():
    args=passing_gate_args()
    for fold in args['folds'][3:]:fold['sharpe_ratio']=0.
    args['neighbours']=args['neighbours'][:23]+[{'status':'unavailable_at_bound'}]*23
    assert report.net_profitability_gates(**args)['G4b_N'] is True


def test_s72_bootstrap_draws_match_literal_scheme_and_seed():
    values=np.random.default_rng(1).normal(size=151)
    rng=np.random.default_rng(20261107)
    draws=report.bootstrap_sharpes(values,7,100,rng)
    rng=np.random.default_rng(20261107)
    expected=[]
    for _ in range(100):
        starts=rng.integers(0,len(values)-7+1,size=math.ceil(len(values)/7))
        x=values[(starts[:,None]+np.arange(7)).ravel()[:len(values)]]
        expected.append(x.mean()/x.std(ddof=0) if x.std(ddof=0) else 0.)
    assert draws==pytest.approx(expected,abs=1e-12)
    assert np.array_equal(draws,report.bootstrap_sharpes(values,7,100,np.random.default_rng(20261107)))


def test_s72_null_sigma_iid_ar1_and_reproducible():
    rng=np.random.default_rng(4)
    iid=rng.normal(size=6000)
    a=report.null_sharpe_calibration(iid,1913,sigma_draws=1000,tail_draws=2000)
    assert a==report.null_sharpe_calibration(iid,1913,sigma_draws=1000,tail_draws=2000)
    floor=1/math.sqrt(len(iid)-1)
    assert a['sigma_null']/floor==pytest.approx(1.,abs=.14)
    ar=np.zeros(12000)
    for i in range(1,len(ar)):ar[i]=.3*ar[i-1]+rng.normal()
    b=report.null_sharpe_calibration(ar,1913,sigma_draws=1000,tail_draws=2000)
    assert b['sigma_null']*math.sqrt(len(ar)-1)==pytest.approx(math.sqrt(1.3/.7),rel=.15)
    assert b['sigma_null']==max(b['iid_floor'],*b['block_sds'].values())
    assert b['c_tail']==max(b['tail_quantiles'].values())


def test_s72_expected_max_and_exact_dsr_boundary():
    n=1913;g=.5772156649
    e=(1-g)*norm.ppf(1-1/n)+g*norm.ppf(1-1/(n*np.e))
    boundary=.03*(e+norm.ppf(.95))
    gate=report.dsr_conditions(boundary,.03,-1.,n)
    assert gate['expected_max_standard_normal']==pytest.approx(e)
    assert gate['dsr']==pytest.approx(.95,abs=1e-14) and gate['dsr_pass'] is True
    assert gate['tail_pass'] is True


def test_s72_strict_tail_atom_inverted_cdf_and_next_float():
    draws=np.array([0.]*17+[1.]*2+[3.])
    # N=2 gives q=.975: the whole upper atom is protected, unlike interpolation.
    tail=report.bootstrap_tail_quantile(draws,2)
    assert tail==3. and np.quantile(draws,.975,method='linear') < tail
    assert (draws>tail).mean() <= .05/2
    assert report.dsr_conditions(tail,.01,tail,2)['dsr_pass'] is True
    assert report.dsr_conditions(tail,.01,tail,2)['tail_pass'] is False
    assert report.dsr_conditions(np.nextafter(tail,np.inf),.01,tail,2)['tail_pass'] is True
    # Quantile inside a tied atom: both methods coincide; strict comparison still rejects it.
    tied=np.array([0.]*17+[3.]*3)
    assert report.bootstrap_tail_quantile(tied,2)==np.quantile(tied,.975,method='linear')==3.
    assert (tied>3).mean() <= .05/2 < (tied>=3).mean()


@pytest.mark.parametrize('values,n', [([1,2],2),([1]*90,2),([float('nan')]*90,2),
                                    (list(range(90)),1),(list(range(89)),2)])
def test_s72_indeterminate_fails(values,n):
    assert report.dependence_dsr_gate(values,n)['status']=='indeterminate'


@pytest.mark.parametrize('sigma,tail', [(float('nan'),1),(.01,float('nan'))])
def test_s72_nonfinite_calibration_fails(monkeypatch,sigma,tail):
    monkeypatch.setattr(report,'null_sharpe_calibration',lambda *a,**k:{'sigma_null':sigma,'c_tail':tail})
    assert report.dependence_dsr_gate(np.arange(90),2)['status']=='indeterminate'


def test_s72_requires_dsr_and_tail_and_never_library_passes():
    assert report.dsr_conditions(.1,1.,.01,1913)['dsr_pass'] is False
    assert report.dsr_conditions(.1,1.,.01,1913)['tail_pass'] is True
    assert report.dsr_conditions(1.,.01,2.,1913)['dsr_pass'] is True
    assert report.dsr_conditions(1.,.01,2.,1913)['tail_pass'] is False


def test_s71_slippage_stress_resimulates_compounding(monkeypatch):
    from tests.test_ledger import replay
    ledger,_=replay()
    baseline=opt.simulate_trades_risk_based(ledger,fee_bps=5,slippage_bps=1)
    stress=opt.simulate_trades_risk_based(ledger,fee_bps=5,slippage_bps=2)
    assert stress.quantity.iloc[1] != baseline.quantity.iloc[1]
    monkeypatch.setattr(diagnose,'scored_only',lambda t,*a:t)
    actual=report.slippage_resimulation(ledger,[],[])
    expected=float((stress.net_pnl/stress.dollar_risk).mean())
    assert actual['net_expectancy_r']==expected
    assert actual['sizing']=='risk-based re-simulation at 5/2 bps'
    import hashlib
    assert actual['quantities_sha256']==hashlib.sha256(stress.quantity.to_numpy(dtype='<f8').tobytes()).hexdigest()
    assert actual['quantities_length']==len(stress)


def test_c8_report_keeps_comparator_params_and_routes_candidate_neighbours(monkeypatch):
    from scripts import phase4_studies as p4
    from tests.test_pipeline import synth_bars
    from tests.test_optimize_orderflow import params
    from tests.test_phase4b_replays import sampled
    from tests.test_ledger import replay
    ledger,_=replay()
    prefix=p4.SearchPrefix(synth_bars(120,step_ms=86400000),[None]*120,96)
    values=sampled();values['ts_stacked_min_levels']=2
    seen=[]
    def replay_candidate(sample,prefix,**kw):
        seen.append(p4.full_strategy_params(sample.params))
        return {'ledger':ledger,'replay_digest':'candidate','signal_counts':{'initiative':3},
                'replay_identity':{'inert_params':{'absorption_min_delta_pct':.3}}}
    monkeypatch.setattr(p4,'replay_full_strategy',replay_candidate)
    comparator_calls=[]
    monkeypatch.setattr(p4,'replay_comparator',lambda c,prefix,**kw:(comparator_calls.append(c.params.copy()) or
        {'ledger':ledger,'replay_digest':'comparator','signal_counts':{'initiative':1}}))
    monkeypatch.setattr(diagnose,'scored_only',lambda t,*a:t)
    monkeypatch.setattr(report,'campaign_union',lambda *a:{'n_symbol_all_studies':1913,'trial_sharpe_std':.1})
    monkeypatch.setattr(report,'dependence_dsr_gate',lambda *a:{'status':'evaluated','dsr_pass':True,'tail_pass':True})
    monkeypatch.setattr(report,'cscv_pbo',lambda *a,**kw:{'pbo':.1})
    monkeypatch.setattr(report,'daily_matrix',lambda *a:np.ones((96,3)))
    monkeypatch.setattr(report,'search_returns',lambda *a:pd.Series(np.arange(96)))
    monkeypatch.setattr(report,'rev5_dsr_diagnostic',lambda *a:{'diagnostic':True})
    trials=[{'state':'COMPLETE','feasible':True,'value':3-i,'number':i,'attrs':{
        'params':values,'replay_digest':'candidate','fold_results':[{'sharpe_ratio':1.,'n_trades':30,'penalized':False}]*5}}
        for i in range(3)]
    frozen=p4.FrozenComparator({**params(),'absorption_min_delta_pct':.2525},'comparator')
    result=report.build_full_strategy_report(prefix,trials,frozen,[],[{}]*5)
    assert result['comparator']['replay_digest']=='comparator'
    assert comparator_calls==[frozen.params] and comparator_calls[0]['absorption_min_delta_pct']==.2525
    assert len(result['top3'])==3 and len(result['parameter_stability'])==46
    assert seen and all(p['absorption_min_delta_pct']==opt.instrument_config('BTCUSDT').absorption.min_delta_pct for p in seen)
    assert result['cost_grid']['label']=='diagnostic: fixed-quantity repricing'
    assert result['slippage_stress']['sizing']=='risk-based re-simulation at 5/2 bps'


def test_s72_union_three_synthetic_sqlite_dbs_and_tamper(tmp_path):
    import sqlite3
    from orderflow_system.data.paired_dataset import sha256_file
    snapshots=[]
    for i in range(3):
        path=tmp_path/f'{i}.db'
        with sqlite3.connect(path) as db:
            db.executescript('CREATE TABLE studies(study_id INTEGER, study_name TEXT);'
                'CREATE TABLE trials(trial_id INTEGER,study_id INTEGER,number INTEGER,state TEXT);'
                'CREATE TABLE trial_values(trial_id INTEGER,value REAL,value_type TEXT,objective INTEGER);'
                'CREATE TABLE trial_params(trial_id INTEGER,param_name TEXT,param_value REAL,distribution_json TEXT);'
                'CREATE TABLE trial_user_attributes(trial_id INTEGER,key TEXT,value_json TEXT);')
            db.execute('INSERT INTO studies VALUES(1,?)',('BTCUSDT__old' if i<2 else 'BTCUSDT__new',))
            db.execute("INSERT INTO trials VALUES(1,1,0,'COMPLETE')")
            db.execute("INSERT INTO trial_values VALUES(1,1,'FINITE',0)")
            db.execute("INSERT INTO trial_user_attributes VALUES(1,'replay_digest','\"same\"')")
        snapshots.append({'path':str(path),'sha256':sha256_file(path)})
    assert report.campaign_union(snapshots)['n_symbol_all_studies']==2
    assert report.campaign_union(snapshots)['duplicate_trials']==1
    (tmp_path/'0.db').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='snapshot changed'):report.campaign_union(snapshots)


def test_c8_no_feasible_candidate_is_a_reported_failure_before_replay(monkeypatch):
    from scripts import phase4_studies as p4
    from tests.test_pipeline import synth_bars
    from tests.test_optimize_orderflow import params
    prefix=p4.SearchPrefix(synth_bars(30),[None]*30,24)
    monkeypatch.setattr(p4,'replay_full_strategy',lambda *a,**kw:pytest.fail('no candidate to replay'))
    result=report.build_full_strategy_report(prefix,[{'state':'COMPLETE','feasible':False,'value':-99.,'number':0}],
        p4.FrozenComparator(params(),'unused'),[],[])
    assert result['G4b_N'] is False and result['gates']['10'] is False
    assert result['status']=='indeterminate' and result['reason']=='no feasible candidate'

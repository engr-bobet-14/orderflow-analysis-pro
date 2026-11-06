"""Read-only Phase 4 diagnostics and pre-registered gross-edge gates."""
from __future__ import annotations

import argparse
import base64
from datetime import date
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from orderflow_system.config.settings import TradeSignalConfig
from orderflow_system.data.paired_dataset import sha256_file
from scripts import optimize_orderflow as opt, diagnose_trades as diagnose
from scripts.phase4_policy import checked_write
from scripts.phase4_studies import TRADE_SEARCH_SPACE_BOUNDS


def cscv_pbo(matrix, blocks=16):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError('CSCV needs at least two candidate columns; single-candidate PBO is not applicable')
    if blocks < 2 or blocks % 2 or matrix.shape[0] < blocks or not np.isfinite(matrix).all():
        raise ValueError('CSCV requires finite aligned daily returns and an even number of nonempty blocks')
    from RiskLabAI.backtest.probability_of_backtest_overfitting import probability_of_backtest_overfitting
    pbo, logits = probability_of_backtest_overfitting(matrix, n_partitions=blocks, n_jobs=1)
    if not np.isfinite(logits).all():
        raise ValueError('CSCV produced non-finite ranks')
    return {'pbo':float(pbo), 'blocks':blocks, 'n_candidates':matrix.shape[1], 'n_splits':len(logits)}


def daily_matrix(trials):
    series = []
    for trial in trials:
        payload = trial['attrs']['daily_returns_search']
        values = np.frombuffer(base64.b64decode(payload['base64'], validate=True), dtype='<f4')
        if len(values) != payload['n']:
            raise ValueError('daily return length mismatch')
        series.append(pd.Series(values, index=pd.date_range(payload['start'], periods=len(values), tz='UTC')))
    if any(not s.index.equals(series[0].index) for s in series[1:]):
        raise ValueError('trial daily-return calendars differ')
    frame = pd.concat(series, axis=1)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('trial daily returns must be finite')
    return frame.to_numpy()


def search_returns(full, bars, folds):
    """Scored exit P&L divided by actual continuous equity at each UTC day's open."""
    days = pd.date_range(pd.Timestamp(bars[0].timestamp_ms, unit='ms', tz='UTC').floor('D'),
                         pd.Timestamp(bars[-1].timestamp_ms, unit='ms', tz='UTC').floor('D'))
    scored = diagnose.scored_only(full, bars, folds)
    def sums(trades):
        return trades.groupby(pd.to_datetime(trades.exit_time, utc=True).dt.floor('D')).net_pnl.sum().reindex(days, fill_value=0.)
    daily_pnl = sums(full)
    equity = opt.INITIAL_CAPITAL_USD + daily_pnl.cumsum()
    prior = equity.shift(1, fill_value=opt.INITIAL_CAPITAL_USD)
    blown = equity.le(0).cummax()
    already_blown = blown.shift(1, fill_value=False)
    returns = sums(scored)/prior.where(~already_blown, 1.)
    first_blow = blown & ~already_blown
    returns.loc[first_blow] = (daily_pnl/prior).loc[first_blow].clip(lower=-1.)
    returns.loc[already_blown] = 0.
    returns.attrs['blown'] = bool(blown.any())
    return returns


def campaign_union(snapshots, symbol='BTCUSDT'):
    seen, sharpes, per_db, duplicates = set(), [], [], 0
    for snapshot in snapshots:
        path = Path(snapshot['path'])
        if sha256_file(path) != snapshot['sha256']:
            raise ValueError(f'campaign database snapshot changed: {path}')
        count = added = 0
        for name in opt.list_study_names(path):
            # Study names encode symbol; synthetic test names need no symbol prefix.
            if '__' in name and not name.startswith(symbol+'__'):
                continue
            for trial in opt.read_study_trials(path, name):
                if trial['state'] != 'COMPLETE':
                    continue
                count += 1
                key = (name, trial['number'], opt.fingerprint(trial['params']), trial['attrs'].get('replay_digest'))
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                added += 1
                value = trial['attrs'].get('raw_daily_sharpe')
                if value is not None and math.isfinite(value):
                    sharpes.append(value)
        if sha256_file(path) != snapshot['sha256']:
            raise ValueError(f'campaign snapshot changed while reading: {path}')
        per_db.append({'path':str(path), 'sha256':snapshot['sha256'], 'complete':count, 'unique_added':added})
    return {'n_symbol_all_studies':len(seen), 'n_dispersion_trials':len(sharpes),
            'trial_sharpe_std':float(np.std(sharpes, ddof=1)) if len(sharpes)>1 else None,
            'per_db':per_db, 'duplicate_trials':duplicates}


def report_snapshots(campaign,campaign_hash,study_db,outdir=None):
    """One read snapshot checks campaign completion and feeds SQLite's backup API."""
    from scripts import phase4_policy as policy,phase4_studies as p4
    legacy=campaign.get('snapshots',[])
    if len(legacy)!=3 or 'phase4_db' not in campaign:
        raise ValueError('campaign snapshots require three legacy hashes and the Phase 4 DB path')
    source=policy.phase4_db_path(study_db)
    if source!=policy.phase4_db_path(campaign['phase4_db']):
        raise ValueError('snapshot source differs from the registered Phase 4 DB')
    if len({Path(item['path']).resolve() for item in legacy})!=3 or any(Path(item['path']).resolve()==source for item in legacy):
        raise ValueError('legacy snapshot paths must be three distinct sources')
    for item in legacy:
        if sha256_file(Path(item['path']))!=item['sha256']:
            raise ValueError('legacy snapshot hash changed')
    expected=policy.ROOT/'output/phase4/reports/snapshots'
    out=checked_write(outdir if outdir is not None else expected)
    if not out.is_relative_to(expected):
        raise ValueError('Phase 4 snapshots must be under output/phase4/reports/snapshots')
    with opt._readonly_db(source) as con:
        admitted={}
        recovery={};runtime={}
        for sid,name,raw in con.execute('SELECT s.study_id,s.study_name,a.value_json FROM studies s '
                "JOIN study_user_attributes a ON a.study_id=s.study_id AND a.key='identity'"):
            identity=json.loads(raw)
            if identity.get('campaign_sha256')!=campaign_hash:
                continue
            attrs={key:json.loads(value) for key,value in con.execute('SELECT key,value_json FROM study_user_attributes WHERE study_id=?',(sid,))}
            runtime[name]=runtime_diagnostics(attrs)
            stop=con.execute("SELECT value_json FROM study_user_attributes WHERE study_id=? AND key='phase4_runtime_stop'",(sid,)).fetchone()
            if stop and json.loads(stop[0]) is not None:
                raise ValueError('Phase 4 runtime stop requires owner decision before report admission')
            arm=p4.study_arm(identity,name)
            if arm in admitted or arm not in campaign['arms'] or campaign['arms'][arm].get('kind','study')!='study':
                raise ValueError('duplicate or unregistered campaign study arm')
            counts=dict(con.execute('SELECT state,count(*) FROM trials WHERE study_id=? GROUP BY state',(sid,)))
            if counts.get('RUNNING',0) or counts.get('WAITING',0) or counts.get('COMPLETE',0)!=campaign['arms'][arm]['budget']:
                raise ValueError('campaign study requires exact budget and no RUNNING/WAITING trials')
            admitted[arm]=name
            reconciled=con.execute("SELECT count(*) FROM trial_user_attributes a JOIN trials t ON a.trial_id=t.trial_id "
                "WHERE t.study_id=? AND t.state='FAIL' AND a.key='phase4_reconciled'",(sid,)).fetchone()[0]
            recovery[name]={'failed':counts.get('FAIL',0),'reconciled':reconciled}
        if set(admitted)!={key for key,arm in campaign['arms'].items() if arm.get('kind','study')=='study'}:
            raise ValueError('campaign study budget is incomplete: missing registered study')
        out.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out,prefix='.snapshot-') as staging:
            path=Path(staging)/'phase4.db'
            with sqlite3.connect(path) as target:
                con.backup(target)
            digest=sha256_file(path)
            final=out/(digest+'.db')
            if final.exists():
                if sha256_file(final)!=digest:
                    raise ValueError('existing Phase 4 snapshot hash mismatch')
            else:
                path.rename(final)
    return [*legacy,{'path':str(final),'sha256':digest,'source_path':str(source),'campaign_sha256':campaign_hash,
                     'trial_recovery':recovery,'runtime_diagnostics':runtime}]


def runtime_diagnostics(attrs):
    return {'policy':attrs.get('phase4_runtime_policy'),'runtime_seconds':attrs.get('phase4_runtime_seconds'),
            'memory_preflight':attrs.get('phase4_memory_preflight',[]),
            'reconciliation':attrs.get('phase4_reconciliation_audit',[])}


def _gross_days(trades, days):
    valid = trades[trades.dollar_risk > 0].copy()
    valid['day'] = pd.to_datetime(valid.exit_time, utc=True).dt.floor('D')
    valid['gross_r'] = valid.gross_pnl/valid.dollar_risk
    grouped = valid.groupby('day').gross_r.agg(['sum', 'count']).reindex(days, fill_value=0)
    return grouped['sum'].to_numpy(), grouped['count'].to_numpy(), int(len(trades)-len(valid))


def gross_bootstrap(arm, baseline, start, end, *, block_length=7, n_resamples=2000, seed=20261105):
    days = pd.date_range(pd.Timestamp(start).floor('D'), pd.Timestamp(end).floor('D'), freq='D')
    if days.tz is None:
        days = days.tz_localize('UTC')
    if len(days) < block_length:
        raise ValueError('gross bootstrap needs at least one full 7-day block')
    arm_sum, arm_n, excluded = _gross_days(arm, days)
    b0_sum, b0_n, excluded_b0 = _gross_days(baseline, days)
    rng = np.random.default_rng(seed)
    arm_est, differences, invalid = [], [], 0
    for _ in range(n_resamples):
        starts = rng.integers(0, len(days)-block_length+1, size=math.ceil(len(days)/block_length))
        indices = (starts[:,None]+np.arange(block_length)).ravel()[:len(days)]
        na, nb = arm_n[indices].sum(), b0_n[indices].sum()
        if na == 0 or nb == 0:
            invalid += 1
            continue
        a, b = arm_sum[indices].sum()/na, b0_sum[indices].sum()/nb
        arm_est.append(a)
        differences.append(a-b)
    determinate = invalid/n_resamples <= .01 and bool(arm_est)
    return {'status':'evaluated' if determinate else 'indeterminate', 'invalid_resamples':invalid,
            'n_resamples':n_resamples, 'seed':seed, 'block_length':block_length,
            'excluded_nonpositive_risk':excluded, 'baseline_excluded_nonpositive_risk':excluded_b0,
            'arm_expectancy_r':float(arm_sum.sum()/arm_n.sum()) if arm_n.sum() else None,
            'baseline_expectancy_r':float(b0_sum.sum()/b0_n.sum()) if b0_n.sum() else None,
            'arm_ci':np.percentile(arm_est, [2.5,97.5]).tolist() if arm_est else None,
            'difference_ci':np.percentile(differences, [2.5,97.5]).tolist() if differences else None}


def neighbours(params, space=None):
    space = space or {**opt.SEARCH_SPACE_BOUNDS, **TRADE_SEARCH_SPACE_BOUNDS}
    rows = []
    for name, base in params.items():
        if name not in space:
            continue
        low, high, kind = space[name]
        # Intersect the search space with the same config/load_space validity domain.
        if name.startswith('ts_'):
            TradeSignalConfig(**{name[3:]:low})
            TradeSignalConfig(**{name[3:]:high})
        if kind == 'int':
            step = max(1, round(.1*(high-low)))
        else:
            step = .1*(math.log(high)-math.log(low)) if kind == 'float_log' else .1*(high-low)
        for sign in (-1,1):
            proposed = math.exp(math.log(base)+sign*step) if kind == 'float_log' else base+sign*step
            effective = min(high, max(low, proposed))
            if kind == 'int':
                effective = int(effective)
            if name.startswith('ts_'):
                TradeSignalConfig(**{name[3:]:effective})
            rows.append({'dimension':name, 'direction':sign, 'base':base, 'requested_value':proposed,
                         'effective_value':effective, 'effective_change':effective-base,
                         'status':'unavailable_at_bound' if effective == base else 'available'})
    return rows


def cost_stress(trades):
    grid = diagnose.cost_grid(trades, fees=(4.,5.), slips=(0.,1.,2.,3.)).to_dict('records')
    grid += diagnose.cost_grid(trades, fees=(0.,), slips=(0.,)).to_dict('records')
    break_even = {}
    risk = trades.dollar_risk.where(trades.dollar_risk > 0)
    for fee in (4.,5.):
        zero = float((diagnose.reprice_costs(trades,fee,0).net_pnl/risk).mean()) if len(trades) else math.nan
        one = float((diagnose.reprice_costs(trades,fee,1).net_pnl/risk).mean()) if len(trades) else math.nan
        break_even[str(fee)] = zero/(zero-one) if math.isfinite(zero) and zero>one else None
    return {'grid':opt.json_safe(grid), 'break_even_slippage_bps':opt.json_safe(break_even),
            'sizing':'fixed entries, exits, quantities; compounding not recomputed'}


def funding_exposure(trades):
    flags = []
    for row in trades.itertuples():
        first = pd.Timestamp(row.entry_time).floor('8h')+pd.Timedelta(hours=8)
        flags.append(first <= row.exit_time)
    mask = np.asarray(flags, dtype=bool)
    return {'modelled':False, 'crossing_00_08_16_utc':int(mask.sum()),
            'share':float(mask.mean()) if len(mask) else None,
            'notional':float(trades.loc[mask, 'position_notional'].sum()) if len(mask) else 0.}


def signal_multiples(counts,baseline):
    per_type = {k:counts.get(k,0)/baseline[k] if baseline.get(k,0) else
                (1. if counts.get(k,0)==0 else None) for k in counts.keys()|baseline.keys()}
    total = sum(counts.values())/sum(baseline.values()) if sum(baseline.values()) else (1. if not sum(counts.values()) else None)
    return {'per_type':per_type,'total':total,
            'maximum':max(per_type.values(),default=1.) if all(v is not None for v in per_type.values()) else None,
            'gate_definition':'maximum per-signal-type multiple; new types without baseline counts fail closed'}


def gate(bootstrap, trial, top3, pbo, frequency_multiple, *, deterministic=False):
    attrs = trial['attrs']
    floors = attrs.get('total_trades',0) >= 105 and all(f['n_trades']>=11 for f in attrs['fold_results'])
    return bool(bootstrap['status'] == 'evaluated' and bootstrap['arm_ci'][0]>0
                and bootstrap['difference_ci'][0]>0 and floors
                and frequency_multiple is not None and frequency_multiple<=3
                and (deterministic or (len(top3)>=3 and all(v is not None and v>0 for v in top3)
                                       and pbo is not None and pbo<.5)))


def replay_trial(args, bars, trial, *, verify=True):
    from scripts.phase4_studies import trial_replay_kwargs
    params = trial['attrs']['params']
    result = opt.run_backtest(bars, opt.build_instrument_config(params,args.symbol), params['price_proximity_pct'],
                               bars[0].bar_type, **trial_replay_kwargs(args,params))
    if verify and result['replay_digest'] != trial['attrs']['replay_digest']:
        raise ValueError('report replay digest mismatch')
    if verify and trial['attrs'].get('raw_signal_digest') != result.get('raw_signal_digest'):
        raise ValueError('report raw signal digest mismatch')
    trades = diagnose.add_gross(opt.simulate_trades_risk_based(result['ledger'],fee_bps=args.fee_bps,
                                                               slippage_bps=args.slippage_bps))
    return result, trades


def build_study_report(args, bars, trials, baseline_trial, baseline_args, snapshots, space):
    complete = [t for t in trials if t['state']=='COMPLETE']
    feasible = sorted([t for t in complete if t['feasible'] and t['value'] is not None],
                      key=lambda t:(-t['value'],t['number']))
    if not feasible:
        raise ValueError('report needs a feasible candidate')
    best = feasible[0]
    n_search, folds = opt.split_timeline(bars,args)
    search = bars[:n_search]
    baseline_result, baseline_trades = replay_trial(baseline_args,search,baseline_trial)
    top = []
    for trial in feasible[:3]:
        result, trades = replay_trial(args,search,trial)
        decomposition = diagnose.decompose(trades)
        top.append({'trial':trial['number'], 'replay_digest':result['replay_digest'],
                    'decomposition':decomposition, 'cost_stress':cost_stress(trades),
                    'funding_exposure':funding_exposure(trades)})
        if trial is best:
            candidate_result, candidate_trades = result,trades
    bootstrap = gross_bootstrap(candidate_trades,baseline_trades,
        pd.Timestamp(search[0].timestamp_ms,unit='ms',tz='UTC'), pd.Timestamp(search[-1].timestamp_ms,unit='ms',tz='UTC'))
    multiples = signal_multiples(candidate_result['signal_counts'],baseline_result['signal_counts'])
    multiple = multiples['maximum']
    try:
        pbo = cscv_pbo(daily_matrix(complete))
    except ValueError as exc:
        pbo = {'pbo':None,'status':'indeterminate','reason':str(exc)}
    campaign = campaign_union(snapshots,args.symbol)
    returns = search_returns(candidate_trades,search,folds)
    # deflated_gate consumes a Study-like read-only view; no Optuna storage is opened.
    study_view = SimpleNamespace(get_trials=lambda **kw:[SimpleNamespace(user_attrs=t['attrs']) for t in complete])
    dsr = opt.deflated_gate(study_view,returns,campaign=campaign)
    stability = []
    for row in neighbours(best['attrs']['params'],space):
        if row['status']=='available':
            params = {**best['attrs']['params'],row['dimension']:row['effective_value']}
            result,trades = replay_trial(args,search,{'attrs':{'params':params}},verify=False)
            row.update(replay_digest=result['replay_digest'],decomposition=diagnose.decompose(trades))
        stability.append(row)
    from scripts.search_space_audit import boundary_hits
    top10 = feasible[:10]
    dispersion = {key:float(np.std([t['attrs']['params'][key] for t in top10])) for key in space}
    from scripts.phase4_studies import trial_recovery
    return opt.json_safe({'trial':best['number'],'trial_recovery':trial_recovery(trials),'replay_digest':candidate_result['replay_digest'],
        'raw_signal_digest':candidate_result.get('raw_signal_digest'),'top3':top,'bootstrap':bootstrap,
        'CSCV':pbo,'campaign':campaign,'deflated_sharpe':dsr,'signal_frequency_multiple':multiple,
        'signal_frequency_multiples':multiples,
        'signal_counts':candidate_result['signal_counts'],'baseline_signal_counts':baseline_result['signal_counts'],
        'parameter_stability':stability,'top10_dispersion':dispersion,'boundary_audit':boundary_hits(complete,space),
        'fold_results':best['attrs']['fold_results'],
        'fold_calendar_bounds':[{'start':str(pd.Timestamp(search[f['score_start_idx']].timestamp_ms,unit='ms',tz='UTC')),
                                'end':str(pd.Timestamp(search[int(f['test_idx'].max())].timestamp_ms,unit='ms',tz='UTC'))} for f in folds],
        'G2':gate(bootstrap,best,[t['decomposition']['gross_expectancy_r'] for t in top],pbo['pbo'],multiple),
        'limitations':['Funding not modelled','Stop-fill model is optimistic for fast moves',
                        'Pseudo-holdout conditional on full-history bar calibration','CSCV dependent blocks is indicative']})


def build_candidate_report(args,bars,artifact,baseline_trial,baseline_trials,baseline_args,snapshots):
    """P2 has no selection study: PBO and top-three are explicitly not applicable."""
    n_search,folds = opt.split_timeline(bars,args)
    search = bars[:n_search]
    result,trades = replay_trial(args,search,{'attrs':artifact})
    baseline_result,baseline_trades = replay_trial(baseline_args,search,baseline_trial)
    bootstrap = gross_bootstrap(trades,baseline_trades,pd.Timestamp(search[0].timestamp_ms,unit='ms',tz='UTC'),
                                 pd.Timestamp(search[-1].timestamp_ms,unit='ms',tz='UTC'))
    counts,base_counts = result['signal_counts'],baseline_result['signal_counts']
    multiples = signal_multiples(counts,base_counts)
    frequency = multiples['maximum']
    fold_results = [opt.evaluate_fold(f,search,artifact['params'],args,trades) for f in folds]
    trial = {'attrs':{'total_trades':sum(f['n_trades'] for f in fold_results),'fold_results':fold_results}}
    try:
        baseline_pbo = cscv_pbo(daily_matrix([t for t in baseline_trials if t['state']=='COMPLETE']))
    except ValueError as exc:
        baseline_pbo = {'status':'indeterminate','reason':str(exc)}
    campaign = campaign_union(snapshots,args.symbol)
    study_view = SimpleNamespace(get_trials=lambda **kw:[SimpleNamespace(user_attrs=t['attrs']) for t in baseline_trials if t['state']=='COMPLETE'])
    dsr = opt.deflated_gate(study_view,search_returns(trades,search,folds),campaign=campaign)
    return opt.json_safe({'replay_digest':result['replay_digest'],'raw_signal_digest':result['raw_signal_digest'],
        'CSCV':{'status':'not_applicable'},'top3':{'status':'not_applicable'},'inherited_B0_PBO':baseline_pbo,
        'campaign':campaign,'deflated_sharpe':dsr,'bootstrap':bootstrap,'decomposition':diagnose.decompose(trades),
        'cost_stress':cost_stress(trades),'funding_exposure':funding_exposure(trades),'signal_counts':counts,
        'baseline_signal_counts':base_counts,'signal_frequency_multiple':frequency,'signal_frequency_multiples':multiples,
        'fold_results':fold_results,
        'G2':gate(bootstrap,trial,[],None,frequency,deterministic=True),
        'parameter_stability':{'status':'not_applicable','reason':'P2 has no tuned dimensions'},
        'selection_risk':'inherited from frozen B0 selection; campaign multiplicity includes all studies'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-db',type=Path,required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--study-name')
    source.add_argument('--candidate')
    parser.add_argument('--baseline-study')
    parser.add_argument('--campaign-manifest',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,default=ROOT/'output/optimize/phase4/cache')
    parser.add_argument('--trade-cache',type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'output/phase4/reports')
    cli = parser.parse_args(argv)
    try:
        out = checked_write(cli.out)
        if not out.is_relative_to(ROOT/'output/phase4'):
            raise ValueError('reports must be under output/phase4')
        from scripts import phase4_studies as p4
        campaign, campaign_hash = p4.read_campaign(cli.campaign_manifest)
        source_db=cli.study_db
        snapshots=report_snapshots(campaign,campaign_hash,source_db)
        cli.study_db=Path(snapshots[-1]['path'])
        artifact = p4.read_candidate(cli.candidate) if cli.candidate else None
        if artifact:
            cli.baseline_study = artifact['source_b0']['source_study']
        if not cli.baseline_study:
            raise ValueError('--baseline-study is required for a study report')
        study_attrs=opt.read_study_attrs(cli.study_db,cli.study_name or cli.baseline_study)
        identity = study_attrs['identity']
        if identity.get('phase')!='phase4' or identity['strategy_fp']!=opt.STRATEGY_FP:
            raise ValueError('report requires current Phase 4 identity')
        if identity.get('campaign_sha256') != campaign_hash or (artifact and artifact['campaign_sha256'] != campaign_hash):
            raise ValueError('report campaign identity mismatch')
        path,meta = diagnose.find_cached_bars(cli.cache_dir,identity['data_fp'])
        from scripts.dataset_policy import ensure_dev_window
        ensure_dev_window(date.fromisoformat(meta['start']),date.fromisoformat(meta['end']))
        bars = opt.CachedBars(path,meta['bar_type'])
        settings = {k:identity[k] for k in ('fee_bps','slippage_bps','holdout_fraction','embargo','n_splits','min_trades_per_fold','min_total_trades')}
        sig = {'signals':artifact['trade_signals']} if artifact else identity['trade_signals']
        args = SimpleNamespace(**settings,symbol=meta['symbol'],bar_type=meta['bar_type'],trade_cache=cli.trade_cache,
            study_db=source_db,study_name=cli.study_name,
            trade_signal_config=TradeSignalConfig(**sig['signals']) if isinstance(sig,dict) else TradeSignalConfig())
        args.trade_signals = p4.mode_for(args.trade_signal_config)
        args.campaign_manifest,args.seed = cli.campaign_manifest,identity['seed']
        args.n_trials = campaign['arms'][p4.arm_key(args)]['budget']
        p4.campaign_arm(args)
        p4.attach_cache(args,bars)
        expected_cache = artifact['trade_cache'] if artifact else identity.get('trade_cache')
        if getattr(args,'trade_cache_identity',None)!=expected_cache:
            raise ValueError('report cache identity mismatch')
        baseline_identity = opt.read_study_attrs(cli.study_db,cli.baseline_study)['identity']
        if baseline_identity['trade_signals']!='disabled' or baseline_identity['data_fp']!=identity['data_fp']:
            raise ValueError('baseline must be B0 on identical bars')
        baseline_budget = p4.baseline_budget(args,baseline_identity)
        baseline_trials = opt.read_study_trials(cli.study_db,cli.baseline_study)
        baseline = max((t for t in baseline_trials if t['state']=='COMPLETE' and t['feasible']),key=lambda t:(t['value'],-t['number']))
        _,_,baseline_reservation = p4.completed_candidate(cli.study_db,cli.baseline_study,baseline['number'],baseline_budget)
        frozen = artifact['source_b0'] if artifact else identity.get('frozen_params')
        if frozen and any(frozen[k] != v for k,v in baseline_reservation.items()):
            raise ValueError('report B0 is not the frozen baseline')
        if any(baseline_identity[k] != identity[k] for k in settings):
            raise ValueError('baseline scoring/cost settings differ')
        baseline_args = SimpleNamespace(**settings,symbol=args.symbol)
        if artifact:
            p4.baseline_budget(args,baseline_identity,artifact['source_b0']['complete'])
            _,_,source = p4.completed_candidate(cli.study_db,cli.baseline_study,baseline['number'],baseline_budget)
            if any(artifact['source_b0'][k]!=v for k,v in source.items()):
                raise ValueError('candidate source B0 changed')
            result = build_candidate_report(args,bars,artifact,baseline,baseline_trials,baseline_args,snapshots)
            reservation = {'artifact_sha256':cli.candidate,**source}
        else:
            trials = opt.read_study_trials(cli.study_db,cli.study_name)
            best = max((t for t in trials if t['state']=='COMPLETE' and t['feasible']),key=lambda t:(t['value'],-t['number']))
            _,_,reservation = p4.completed_candidate(cli.study_db,cli.study_name,best['number'],args.n_trials)
            result = build_study_report(args,bars,trials,baseline,baseline_args,snapshots,identity['search_space'])
        out.mkdir(parents=True,exist_ok=True)
        result['database_snapshots']=snapshots
        result['runtime_diagnostics']=runtime_diagnostics(study_attrs)
        opt.atomic_json(out/(opt.fingerprint(reservation)+'.json'),result)
        return result
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__=='__main__':
    print(json.dumps(main(),indent=2))

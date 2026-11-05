"""Phase 4 admission, immutable candidates and single-look holdout reservations."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from orderflow_system.config.settings import TradeFlowConfig, TradeSignalConfig, parse_trade_signals
from scripts.phase4_policy import ROOT, checked_write, canonical_bytes

TRADE_SEARCH_SPACE_BOUNDS = {
    'ts_level_zone_frac': (.10, .50, 'float'),
    'ts_min_level_share': (.02, .20, 'float_log'),
    'ts_min_side_ratio': (.55, .90, 'float'),
    'ts_min_reject_bps': (.5, 8., 'float_log'),
    'ts_max_through_bps': (0., 4., 'float'),
    'ts_contrarian_ratio': (1.5, 5., 'float_log'),
    'ts_contrarian_zone_frac': (.10, .50, 'float'),
    'ts_contrarian_min_level_share': (.005, .10, 'float_log'),
}
A2_DIMS = tuple(TRADE_SEARCH_SPACE_BOUNDS)[:5]
E2_DIMS = tuple(TRADE_SEARCH_SPACE_BOUNDS)[5:]


def signals_for(mode):
    return TradeSignalConfig() if mode == 'off' else parse_trade_signals(mode)


def mode_for(signals):
    tokens = []
    if signals.absorption_levels:
        tokens.append('absorption_levels')
    if signals.exhaustion_contrarian != 'off':
        tokens.append('exhaustion_'+signals.exhaustion_contrarian)
    if signals.framing_profile_source == 'trades':
        tokens.append('framing_trades')
    return ','.join(tokens) or 'off'


def is_phase4(args):
    return bool(getattr(args, 'phase4', False) or getattr(args, 'trade_signals', 'off') != 'off'
        or any(getattr(args, k, None) for k in ('trade_cache', 'freeze_params', 'holdout_look', 'campaign_manifest'))
        or any(Path(getattr(args, k)).expanduser().resolve().is_relative_to(ROOT/'output/optimize/phase4')
               for k in ('study_db', 'output', 'cache_dir')))


def configure(args):
    args.phase4_implied = not getattr(args, 'phase4', False) and is_phase4(args)
    args.phase4 = is_phase4(args)
    args.trade_signal_config = signals_for(args.trade_signals)
    if not args.phase4:
        return
    if 'time_1m' in args.bar_type:
        raise ValueError('time_1m is refused in every Phase 4 run')
    if args.start_date < date(2019, 9, 10) or args.end_date > date(2023, 12, 31):
        raise ValueError('Phase 4 requires the development window (2019-09-10..2023-12-31)')
    if args.n_jobs != 1:
        raise ValueError('Phase 4 requires --n-jobs 1')
    if args.evaluate_holdout:
        raise ValueError('Phase 4 requires --holdout-look instead of --evaluate-holdout')
    if args.trade_signal_config.any_enabled and not args.trade_cache:
        raise ValueError('trade mode requires --trade-cache')
    if (args.trade_signals == 'framing_trades' or (args.holdout_look or '').startswith('candidate:')) and not args.freeze_params:
        raise ValueError('P2/candidate look requires verified --freeze-params from B0')
    if not args.no_warm_start:
        raise ValueError('Phase 4 requires --no-warm-start (cold equal budgets)')
    for name in ('study_db', 'output', 'cache_dir', 'trade_cache'):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, checked_write(value))
    checked_write(ROOT/'output/duckdb_tmp')
    if args.freeze_params:
        read_freeze_spec(args)
    if not args.dry_run and (not args.campaign_manifest or not args.owner_approved):
        raise ValueError('experiments and holdout looks require --campaign-manifest and --owner-approved token')
    if not args.dry_run:
        _,args.campaign_identity=read_campaign(args.campaign_manifest)
        verified=require_gates(args.owner_approved,args.campaign_identity)
        args.approval_sha256=verified['sha256']
        args.gates_sha256=verified['gates_sha256']
        from scripts.phase4_resources import require_memory_plan
        require_memory_plan([{'mapped_files':{}}])
    if args.holdout_look and (len(args.bar_type) != 1 or len(args.symbol) != 1):
        raise ValueError('holdout-look requires one symbol and bar type')


def require_gates(owner_approved=None,campaign_sha256=None,argv=None):
    from scripts.phase4_approvals import require_campaign_approval
    return require_campaign_approval(owner_approved,campaign_sha256,argv)


def active_space(space, signals, *, frozen):
    active = list(A2_DIMS if signals.absorption_levels else ())
    active += E2_DIMS if signals.exhaustion_contrarian != 'off' else ()
    result = {} if frozen else {k:v for k,v in space.items() if not k.startswith('ts_')}
    result.update({k:space.get(k, TRADE_SEARCH_SPACE_BOUNDS[k]) for k in active})
    return result


def effective_trade_config(args, params):
    base = getattr(args, 'trade_signal_config', TradeSignalConfig())
    fields = {k[3:]:v for k,v in params.items() if k.startswith('ts_')}
    return TradeFlowConfig(signals=replace(base, **fields))


def merged_params(args, params):
    return {**getattr(args, 'frozen_values', {}), **params}


def trial_replay_kwargs(args, params):
    updates = getattr(args, 'trade_updates', None)
    return ({'trade_updates': updates, 'trade_config': effective_trade_config(args, params)}
            if updates is not None else {})


def attach_cache(args, bars):
    if not args.trade_cache:
        return
    from scripts.trade_feature_cache import CachedTradeUpdates,production_generation,generation_directory
    from scripts.phase4_benchmark import DATASET_ID
    expected=getattr(args,'dataset_id',DATASET_ID)
    root=generation_directory(args.trade_cache,expected)
    generation=production_generation(root)
    if generation['dataset_id']!=expected:
        raise ValueError('production cache dataset mismatch')
    path=Path(generation['lanes'][args.bar_type])
    requested=Path(args.trade_cache).resolve()
    if requested.parent==root and requested!=path:
        raise ValueError('requested trade-cache is an unlisted orphan or wrong bar type')
    from scripts.phase4_resources import require_memory_plan,mapped_cache_files
    args.memory_admission=require_memory_plan([{'mapped_files':mapped_cache_files(path)}])
    args.trade_cache_orphans=generation['orphans']
    cache = CachedTradeUpdates(path, bars, production=True, bar_type=args.bar_type,dataset_id=expected)
    args.trade_updates = cache
    args.trade_cache_identity = cache.identity


def legacy_study_guard(args, name):
    if not getattr(args, 'phase4', False):
        from scripts import optimize_orderflow as opt
        stored = opt._stored_identity(args.study_db, name)
        if stored and stored.get('phase') == 'phase4':
            raise ValueError('legacy optimizer cannot open a Phase 4 study')


def read_campaign(path):
    try:
        raw = Path(path).read_bytes()
        expected = Path(str(path)+'.sha256').read_text().strip()
    except FileNotFoundError as exc:
        raise ValueError('half-sealed or missing campaign: manifest and seal must both exist') from exc
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('campaign manifest hash mismatch; edits require a new campaign')
    campaign = json.loads(raw)
    if campaign.get('phase') != 'phase4':
        raise ValueError('not a Phase 4 campaign')
    return campaign, expected


def arm_key(args):
    return args.symbol+':'+args.bar_type+':'+getattr(args,'trade_signals','off')


def campaign_arm(args,study_name=None,*,under_init_lock=False):
    campaign, digest = read_campaign(args.campaign_manifest)
    key = arm_key(args)
    if key not in campaign['arms']:
        raise ValueError('arm not pre-registered')
    if 'phase4_db' in campaign and Path(args.study_db).resolve()!=Path(campaign['phase4_db']).resolve():
        raise ValueError('study database differs from the registered campaign database')
    arm = campaign['arms'][key]
    for name in ('seed', 'fee_bps', 'slippage_bps', 'min_total_trades', 'min_trades_per_fold',
                 'n_splits', 'embargo', 'holdout_fraction'):
        if getattr(args, name) != campaign[name]:
            raise ValueError(f'campaign setting mismatch: {name}')
    if args.n_trials != arm['budget']:
        raise ValueError('campaign trial budget mismatch')
    args.campaign_identity = digest
    from scripts import optimize_orderflow as opt
    target=study_name or getattr(args,'study_name',None)
    if Path(args.study_db).is_file():
        existing=[]
        names=opt.list_study_names(args.study_db) if under_init_lock else opt._study_names(args.study_db)
        for name in names:
            identity=opt.read_study_attrs(args.study_db,name).get('identity',{})
            if identity.get('campaign_sha256')==digest and study_arm(identity,name)==key:
                existing.append(name)
        if len(existing)>1 or (existing and (arm.get('kind')=='candidate' or target is not None and existing!=[target])):
            raise ValueError('second study for the same arm and campaign is refused')
    return arm


def study_arm(identity,name):
    if identity.get('arm'):
        return identity['arm']
    parts=name.split('__')
    if len(parts)<2:
        raise ValueError('campaign study lacks an arm identity')
    signals=identity['trade_signals']
    mode='off' if signals=='disabled' else mode_for(TradeSignalConfig(**signals['signals']))
    return ':'.join((parts[0],parts[1],mode))


def read_freeze_spec(args):
    if not getattr(args,'freeze_params',None):
        raise ValueError('--freeze-params is required')
    try:
        spec=json.loads(Path(args.freeze_params).read_text())
        from scripts.phase4_policy import phase4_db_path
        db=phase4_db_path(spec['study_db'])
    except (KeyError,TypeError,OSError) as exc:
        raise ValueError('freeze study_db must be a Phase 4 database path') from exc
    if not {'study','trial','budget'} <= spec.keys():
        raise ValueError('freeze spec requires study, trial and budget')
    return {**spec,'study_db':str(db)}


def baseline_budget(args,identity,spec_budget=None):
    if identity.get('trade_signals')!='disabled':
        raise ValueError('source must be a B0 study')
    if not getattr(args,'campaign_manifest',None):
        return spec_budget
    campaign,digest=read_campaign(args.campaign_manifest)
    if digest!=args.campaign_identity or identity.get('campaign_sha256')!=digest:
        raise ValueError('source B0 campaign identity mismatch')
    budget=campaign['arms'][args.symbol+':'+args.bar_type+':off']['budget']
    if spec_budget is not None and spec_budget!=budget:
        raise ValueError('freeze/source B0 budget differs from the campaign budget')
    return budget


def source_b0(args,spec,number=None):
    from scripts import optimize_orderflow as opt
    identity=opt.read_study_attrs(spec['study_db'],spec['study']).get('identity')
    if not identity:
        raise ValueError('study has no stored identity')
    budget=baseline_budget(args,identity,spec['budget'])
    result=completed_candidate(spec['study_db'],spec['study'],spec['trial'] if number is None else number,budget)
    baseline_budget(args,result[1],budget)
    return result


def trial_recovery(trials):
    def state(t):return t['state'] if isinstance(t,dict) else t.state.name
    def attrs(t):return t['attrs'] if isinstance(t,dict) else t.user_attrs
    return {'failed':sum(state(t)=='FAIL' for t in trials),
            'reconciled':sum(state(t)=='FAIL' and bool(attrs(t).get('phase4_reconciled')) for t in trials)}


def completed_candidate(db, study_name, number, budget):
    from scripts import optimize_orderflow as opt
    attrs = opt.read_study_attrs(db, study_name)
    identity = attrs.get('identity')
    if not identity or identity.get('phase') != 'phase4':
        raise ValueError('candidate must come from a Phase 4 study')
    trials = opt.read_study_trials(db, study_name)
    complete = [t for t in trials if t['state'] == 'COMPLETE']
    if len(complete) != budget or any(t['state'] in ('RUNNING', 'WAITING') for t in trials):
        raise ValueError('candidate requires exact COMPLETE budget and no RUNNING/WAITING trials')
    feasible = [t for t in complete if t['feasible'] and t['value'] is not None and np.isfinite(t['value'])]
    if not feasible:
        raise ValueError('no feasible candidate')
    best = max(feasible, key=lambda t:(t['value'], -t['number']))
    if best['number'] != number:
        raise ValueError('candidate is not the pre-registered best feasible trial')
    params = best['attrs']['params']
    reservation = {'study_identity_hash': opt.fingerprint(identity), 'trial':number,
                   'params_sha256': opt.fingerprint(params), 'replay_digest': best['attrs']['replay_digest'],
                   'complete':len(complete),**trial_recovery(trials)}
    return best, identity, reservation


def load_frozen(args, bars):
    """Read-only B0 provenance, then confirm its stored search digest before accepting values."""
    if not args.freeze_params:
        return
    from scripts import optimize_orderflow as opt
    spec = read_freeze_spec(args)
    best, identity, reservation = source_b0(args,spec)
    if identity['trade_signals'] != 'disabled':
        raise ValueError('frozen bar parameters must come from B0')
    values = best['attrs']['params']
    if set(values) != set(opt.SEARCH_SPACE_BOUNDS) or values != spec.get('params') or reservation['replay_digest'] != spec.get('replay_digest'):
        raise ValueError('frozen values/source digest mismatch')
    n_search, _ = opt.split_timeline(bars, args)
    result = opt.run_backtest(bars[:n_search], opt.build_instrument_config(values, args.symbol),
                               values['price_proximity_pct'], bars[0].bar_type)
    if result['replay_digest'] != reservation['replay_digest']:
        raise ValueError('frozen B0 replay digest mismatch')
    args.frozen_values = values
    args.frozen_identity = {**reservation, 'values':values, 'source_study':spec['study']}


def reserved_look(path, arm, candidate, replay, revalidate=None):
    """Hold the exclusive lock across reserve, deterministic replay and result persistence."""
    from scripts import optimize_orderflow as opt
    path = checked_write(path)
    checked_write(str(path)+'.lock')
    with opt.file_lock(str(path)+'.lock'):
        if revalidate is not None and revalidate() != candidate:
            raise ValueError('candidate changed before reservation')
        data = json.loads(path.read_text()) if path.exists() else {}
        prior = data.get(arm)
        if prior and prior['candidate'] != candidate:
            raise ValueError('another candidate or changed trial count is already exposed for this arm')
        if prior and 'result' in prior:
            return prior['result']
        interrupted = prior is not None
        data[arm] = {'candidate':candidate, 'status':'reserved'}
        opt.atomic_json(path, data)
        result = opt.json_safe(replay())
        if interrupted:
            result['completed_after_interruption'] = True
        data[arm].update(status='complete', result=result)
        opt.atomic_json(path, data)
        return result


def write_candidate(args, bars):
    from scripts import optimize_orderflow as opt
    spec = read_freeze_spec(args)
    with opt._shared_lock_readonly(opt.study_lock_path(spec['study_db'],spec['study'])):
        _,_,current = source_b0(args,spec)
        if any(args.frozen_identity[k] != v for k,v in current.items()):
            raise ValueError('source B0 changed before candidate creation')
        return _write_candidate(args,bars)


def _write_candidate(args, bars):
    """P2 is one replay and an immutable artifact, never an Optuna study."""
    from scripts import optimize_orderflow as opt
    if not getattr(args, 'frozen_identity', None):
        raise ValueError('deterministic candidate requires verified --freeze-params')
    params = args.frozen_values
    n_search, _ = opt.split_timeline(bars, args)
    result = opt.run_backtest(bars[:n_search], opt.build_instrument_config(params, args.symbol),
                               params['price_proximity_pct'], bars[0].bar_type, **trial_replay_kwargs(args, params))
    artifact = {'arm':arm_key(args), 'mode':args.trade_signals, 'trade_signals':asdict(effective_trade_config(args, params).signals),
                'params':params, 'trade_cache':args.trade_cache_identity, 'source_b0':args.frozen_identity,
                'replay_digest':result['replay_digest'], 'raw_signal_digest':result['raw_signal_digest'],
                'campaign_sha256':args.campaign_identity}
    raw = canonical_bytes(artifact)
    digest = hashlib.sha256(raw).hexdigest()
    path = checked_write(ROOT/'output/phase4/candidates'/f'{digest}.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as handle:
            handle.write(raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError('content-addressed candidate has different bytes')
    return {'candidate':digest, 'path':str(path), **artifact}


def read_candidate(digest):
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('invalid candidate hash')
    path = ROOT/'output/phase4/candidates'/f'{digest}.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('tampered candidate artifact')
    return json.loads(raw)


def holdout_look(args, bars):
    from scripts import optimize_orderflow as opt
    arm = campaign_arm(args)
    candidate_ref = args.holdout_look
    if candidate_ref.startswith('candidate:'):
        artifact_hash = candidate_ref.split(':',1)[1]
        artifact = read_candidate(artifact_hash)
        if (artifact['arm'] != arm_key(args) or artifact['campaign_sha256'] != args.campaign_identity
                or artifact['trade_cache'] != args.trade_cache_identity):
            raise ValueError('candidate arm/campaign/cache mismatch')
        source = artifact['source_b0']
        spec = read_freeze_spec(args)
        _, _, current = source_b0(args,spec,source['trial'])
        if any(source[k] != current[k] for k in current):
            raise ValueError('source B0 candidate changed')
        params = artifact['params']
        expected = artifact['replay_digest']
        reservation = {'artifact_sha256':artifact_hash, **current}
    else:
        name, number = candidate_ref.rsplit(':', 1)
        trial, identity, reservation = completed_candidate(args.study_db, name, int(number), arm['budget'])
        if identity.get('campaign_sha256')!=args.campaign_identity:
            raise ValueError('holdout study campaign identity mismatch')
        if identity['trade_signals'] != opt.trade_signal_identity(effective_trade_config(args, {})):
            raise ValueError('holdout study arm mismatch')
        params, expected = trial['attrs']['params'], reservation['replay_digest']
        if identity.get('trade_cache') != getattr(args, 'trade_cache_identity', None):
            raise ValueError('holdout study cache mismatch')
    gate_path = ROOT/'output/phase4/reports'/f'{opt.fingerprint(reservation)}.json'
    gate = json.loads(gate_path.read_text())
    if gate.get('G2') is not True or gate.get('replay_digest') != expected:
        raise ValueError('candidate has not passed G2')
    n_search, _ = opt.split_timeline(bars, args)
    def replay():
        result = opt.run_backtest(bars, opt.build_instrument_config(params, args.symbol),
                                   params['price_proximity_pct'], bars[0].bar_type,
                                   prefix_ends=(n_search-1,), **trial_replay_kwargs(args, params))
        if result['prefix_digests'][n_search-1] != expected:
            raise ValueError('holdout search-prefix digest mismatch')
        full = opt.simulate_trades_risk_based(result['ledger'], fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
        start = n_search+int(len(bars)*args.embargo)
        metrics, trades = opt.score_window(full, bars, start, len(bars)-1, args.min_trades_per_fold)
        valid = trades[trades.dollar_risk > 0]
        net_r = float((valid.net_pnl/valid.dollar_risk).mean()) if len(valid) else None
        from types import SimpleNamespace
        source_trials = opt.read_study_trials(spec['study_db'] if candidate_ref.startswith('candidate:') else args.study_db,
                                              spec['study'] if candidate_ref.startswith('candidate:') else name)
        view = SimpleNamespace(get_trials=lambda **kw:[SimpleNamespace(user_attrs=t['attrs'])
            for t in source_trials if t['state']=='COMPLETE'])
        returns = opt.daily_returns(trades,pd.Timestamp(bars[start].timestamp_ms,unit='ms',tz='UTC'),
            pd.Timestamp(bars[-1].timestamp_ms,unit='ms',tz='UTC'),initial_capital=max(metrics['start_equity'],.01))
        return {**metrics, 'net_expectancy_r':net_r,
                'G3': bool(net_r is not None and net_r > 0 and len(trades) >= 11 and not metrics['account_blown']),
                'replay_digest':result['replay_digest'], 'conditional_on_full_history_bar_calibration':True,
                'deflated_sharpe':opt.deflated_gate(view,returns,campaign=gate['campaign'],
                    ineligible_reason='holdout account blown' if metrics['account_blown'] else None)}
    def revalidate():
        if candidate_ref.startswith('candidate:'):
            if read_candidate(artifact_hash) != artifact:
                raise ValueError('candidate artifact changed')
            _,_,current = source_b0(args,spec,source['trial'])
            return {'artifact_sha256':artifact_hash, **current}
        return completed_candidate(args.study_db,name,int(number),arm['budget'])[2]
    db = spec['study_db'] if candidate_ref.startswith('candidate:') else args.study_db
    study_name = spec['study'] if candidate_ref.startswith('candidate:') else name
    # The optimizer takes the corresponding exclusive lock for the entire study run.
    with opt._shared_lock_readonly(opt.study_lock_path(db,study_name)):
        return reserved_look(ROOT/'output/phase4/holdout/exposures.json',arm_key(args),reservation,replay,revalidate)

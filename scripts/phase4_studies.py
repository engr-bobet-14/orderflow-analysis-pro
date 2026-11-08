"""Phase 4 admission, immutable candidates and single-look holdout reservations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from contextlib import contextmanager
from datetime import date
import hashlib
import json
import math
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
        or any(getattr(args, k, None) for k in ('trade_cache', 'freeze_params', 'holdout_look', 'campaign_manifest',
                                              'max_trial_seconds','max_study_seconds'))
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
    if getattr(args, 'phase', None) == 'phase4b':
        if base.initiative_imbalance == 'bonus':
            raise ValueError('phase4b refuses diagnostic-only initiative bonus')
        base = TradeSignalConfig(absorption_levels=True, absorption_source='trade',
            framing_profile_source='trades', exhaustion_contrarian='required',
            initiative_imbalance='required', stacked_min_levels=3)
    config=TradeFlowConfig(signals=replace(base, **fields))
    if getattr(args,'phase',None)=='phase4b':
        validate_phase4b_signals(config.signals)
    return config


def validate_phase4b_signals(signals, *, s1=False):
    if signals.initiative_imbalance=='bonus':
        raise ValueError('phase4b refuses diagnostic-only initiative bonus')
    if (not signals.absorption_levels or signals.absorption_source!='trade'
            or signals.initiative_imbalance not in (('required','off') if s1 else ('required',))
            or signals.exhaustion_contrarian not in (('required','off') if s1 else ('required',))
            or signals.framing_profile_source not in (('trades','bar_vwap') if s1 else ('trades',))):
        raise ValueError('invalid final phase4b signals config')


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
    from scripts.phase4_resources import enforce_current
    enforce_current(args.memory_admission['unique_mapped_bytes'])
    args.trade_updates = cache
    args.trade_cache_identity = cache.identity


def legacy_study_guard(args, name):
    if not getattr(args, 'phase4', False):
        from scripts import optimize_orderflow as opt
        stored = opt._stored_identity(args.study_db, name)
        if stored and stored.get('phase') in ('phase4','phase4b'):
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
    sequential_campaign(campaign)
    return campaign, expected


def sequential_campaign(campaign):
    for settings in (campaign,*campaign.get('arms',{}).values()):
        for key in ('concurrency','n_jobs'):
            if key in settings and (type(settings[key]) is not int or settings[key]!=1):
                raise ValueError(f'Phase 4 sequential execution requires {key} == 1')


def arm_key(args):
    if getattr(args,'phase',None)=='phase4b':
        return args.symbol+':'+args.bar_type+':full'
    return args.symbol+':'+args.bar_type+':'+getattr(args,'trade_signals','off')


def campaign_arm(args,study_name=None,*,under_init_lock=False):
    if getattr(args,'phase',None)=='phase4b':
        from scripts.phase4b_campaign import campaign_arm as full_arm
        return full_arm(args,study_name)
    campaign, digest = read_campaign(args.campaign_manifest)
    key = arm_key(args)
    if key not in campaign['arms']:
        raise ValueError('arm not pre-registered')
    if 'phase4_db' in campaign and Path(args.study_db).resolve()!=Path(campaign['phase4_db']).resolve():
        raise ValueError('study database differs from the registered campaign database')
    arm = campaign['arms'][key]
    for name in ('max_trial_seconds','max_study_seconds'):
        value=arm.get(name)
        if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or value<=0):
            raise ValueError(f'campaign {name} must be finite and positive')
        if getattr(args,name,None) is not None and getattr(args,name)!=value:
            raise ValueError(f'campaign runtime limit mismatch: {name}')
        setattr(args,name,value)
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
    attrs = opt.phase4_db_admission(db,study_name)
    identity = attrs.get('identity')
    if not identity or identity.get('phase') != 'phase4':
        raise ValueError('candidate must come from a Phase 4 study')
    if attrs.get('phase4_runtime_stop') is not None:
        raise ValueError('Phase 4 runtime stop requires owner decision before candidate admission')
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
    if args.holdout_look.startswith('candidate:'):
        source=read_freeze_spec(args)
        opt.phase4_db_admission(source['study_db'],source['study'])
    else:opt.phase4_db_admission(args.study_db,args.holdout_look.rsplit(':',1)[0])
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


# The registered 23-dimensional joint space. The original Phase 4 space is untouched.
FULL_STRATEGY_SPACE = {
    'price_proximity_pct': (.0005, .005, 'float_log'),
    'min_composite_score': (10., 70., 'float'),
    'signal_cooldown_seconds': (5., 120., 'float'),
    'volume_decline_pct': (.05, .6, 'float'),
    'lookback_bars': (5, 30, 'int'),
    'delta_failure_pct': (.5, .95, 'float'),
    'initiative_min_delta_pct': (.2, .6, 'float'),
    'initiative_min_body_bps': (1., 8., 'float_log'),
    'initiative_volume_acceleration_min': (1., 3., 'float'),
    'default_stop_pct': (.001, .01, 'float_log'),
    'target_rr': (1., 4., 'float'),
    **TRADE_SEARCH_SPACE_BOUNDS,
    'ts_imbalance_ratio': (2., 6., 'float_log'),
    'ts_imbalance_min_level_share': (.002, .05, 'float_log'),
    'ts_imbalance_bucket_mult': ((1, 2, 5, 10), None, 'categorical'),
    'ts_stacked_min_levels': (1, 4, 'int'),
}


def full_strategy_params(sampled, symbol='BTCUSDT'):
    from scripts.optimize_orderflow import instrument_config
    if set(sampled) != set(FULL_STRATEGY_SPACE):
        raise ValueError('full strategy requires exactly the 23 sampled dimensions')
    return {**sampled, 'absorption_min_delta_pct': instrument_config(symbol).absorption.min_delta_pct}


class HoldoutAccessError(IndexError):
    """Strategy attempted to read beyond its registered search prefix."""


class TradeUpdatesPrefix:
    def __init__(self, updates, stop):
        self.__updates, self.__stop = updates, stop

    def __len__(self):
        return self.__stop

    @property
    def path(self):
        """Cache path for integrity/memory accounting; no unbounded updates are exposed."""
        return self.__updates.path

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < self.__stop:
            raise HoldoutAccessError('trade update outside search prefix')
        return self.__updates[index]

    def __iter__(self):
        for index in range(self.__stop):
            yield self[index]


class SearchPrefix:
    """Only bounded strategy views escape the full-cache integrity-opening path."""
    def __init__(self, bars, updates, n_search):
        if not 0 < n_search <= len(bars) or len(updates) != len(bars):
            raise ValueError('search prefix/cache length mismatch')
        self.bars = bars[:n_search]
        self.trade_updates = TradeUpdatesPrefix(updates, n_search)
        self.trade_cache_identity = getattr(updates, 'identity', None)

    @classmethod
    def open(cls, bars_path, bars_type, trade_path, args, **cache_options):
        # bars_type, not bar_type: production callers pass bar_type= as a cache identity option.
        from scripts.optimize_orderflow import CachedBars, split_timeline
        from scripts.trade_feature_cache import CachedTradeUpdates
        bars = CachedBars(bars_path, bars_type)
        # Frozen constructor hashes all bytes and validates the full-length bars.
        updates = CachedTradeUpdates(trade_path, bars, **cache_options)
        n_search, _ = split_timeline(bars, args)
        return cls(bars, updates, n_search)


@dataclass(frozen=True)
class FrozenComparator:
    params: dict
    replay_digest: str


@dataclass(frozen=True)
class LegacyReplay:
    params: dict
    signals: TradeSignalConfig
    replay_digest: str

    def __post_init__(self):
        defaults = TradeSignalConfig()
        for name in ('absorption_source', 'initiative_imbalance', 'imbalance_ratio',
                     'imbalance_min_level_share', 'imbalance_bucket_mult', 'stacked_min_levels'):
            if getattr(self.signals, name) != getattr(defaults, name):
                raise ValueError('legacy replay requires backward-compatible 4b defaults')


@dataclass(frozen=True)
class FullStrategySample:
    params: dict
    replay_digest: str | None = None


def _typed_replay(value, expected_type, prefix, params, config, symbol, verify):
    from scripts import optimize_orderflow as opt
    if not isinstance(value, expected_type) or not isinstance(prefix, SearchPrefix):
        raise TypeError('replay type/prefix mismatch')
    kwargs = {} if config is None else {'trade_config':config, 'trade_updates':prefix.trade_updates}
    result = opt.run_backtest(prefix.bars, opt.build_instrument_config(params, symbol),
        params['price_proximity_pct'], prefix.bars[0].bar_type, **kwargs)
    if verify and value.replay_digest is not None and result['replay_digest'] != value.replay_digest:
        raise ValueError('stored replay digest mismatch')
    return result


def replay_comparator(frozen_artifact, prefix, *, symbol='BTCUSDT'):
    if not isinstance(frozen_artifact, FrozenComparator):
        raise TypeError('comparator requires FrozenComparator')
    return _typed_replay(frozen_artifact, FrozenComparator, prefix, frozen_artifact.params, None, symbol, True)


def replay_legacy(stageA_config, prefix, *, symbol='BTCUSDT'):
    if not isinstance(stageA_config, LegacyReplay):
        raise TypeError('legacy replay requires LegacyReplay')
    config = TradeFlowConfig(signals=stageA_config.signals) if stageA_config.signals.any_enabled else None
    return _typed_replay(stageA_config, LegacyReplay, prefix, stageA_config.params, config, symbol, True)


def replay_full_strategy(sampled, prefix, *, symbol='BTCUSDT', signal_overrides=None):
    if not isinstance(sampled, FullStrategySample):
        raise TypeError('full strategy requires FullStrategySample')
    params = full_strategy_params(sampled.params, symbol)
    from types import SimpleNamespace
    config = effective_trade_config(SimpleNamespace(phase='phase4b'), params)
    if signal_overrides:
        allowed={'initiative_imbalance':'off','exhaustion_contrarian':'off','framing_profile_source':'bar_vwap'}
        if len(signal_overrides)!=1 or any(allowed.get(k)!=v for k,v in signal_overrides.items()):
            raise ValueError('unregistered phase4b signal override (initiative bonus is refused)')
        config = replace(config, signals=replace(config.signals, **signal_overrides))
    validate_phase4b_signals(config.signals,s1=True)
    result = _typed_replay(sampled, FullStrategySample, prefix, params, config, symbol, True)
    result['replay_identity'] = {'sampled_params':dict(sampled.params),
        'inert_params':{'absorption_min_delta_pct':params['absorption_min_delta_pct']},
        'trade_signals':asdict(config.signals)}
    return result


def full_strategy_defaults(symbol='BTCUSDT'):
    from scripts.optimize_orderflow import instrument_config
    base, signals = instrument_config(symbol), TradeSignalConfig()
    values = {
        'price_proximity_pct':.002, 'min_composite_score':base.risk.min_composite_score,
        'signal_cooldown_seconds':base.risk.signal_cooldown_seconds,
        'volume_decline_pct':base.exhaustion.volume_decline_pct,
        'lookback_bars':base.divergence.lookback_bars, 'delta_failure_pct':base.divergence.delta_failure_pct,
        'initiative_min_delta_pct':base.initiative.min_delta_pct, 'initiative_min_body_bps':base.initiative.min_body_bps,
        'initiative_volume_acceleration_min':base.initiative.volume_acceleration_min,
        'default_stop_pct':base.risk.default_stop_pct, 'target_rr':base.risk.default_target_pct/base.risk.default_stop_pct,
        **{name:getattr(signals,name[3:]) for name in TRADE_SEARCH_SPACE_BOUNDS},
        'ts_imbalance_ratio':3., 'ts_imbalance_min_level_share':.01,
        'ts_imbalance_bucket_mult':1, 'ts_stacked_min_levels':3,
    }
    return values


S1_REPLAY_NAMES=('defaults','initiative_off','k1','E2_off','P2_bar_vwap','stageB_bar_params')


def full_strategy_probe(symbol='BTCUSDT'):
    """Rev 11's fixed worst-case trial-equivalent, never an adaptive proposal."""
    return {**full_strategy_defaults(symbol),
        'ts_stacked_min_levels':1,'ts_imbalance_bucket_mult':1,'ts_imbalance_min_level_share':.002,
        'ts_imbalance_ratio':2.,'ts_min_level_share':.02,'ts_min_side_ratio':.55,
        'ts_min_reject_bps':.5,'ts_max_through_bps':4.,'ts_level_zone_frac':.50,
        'ts_contrarian_ratio':1.5,'ts_contrarian_zone_frac':.50,'ts_contrarian_min_level_share':.005,
        'signal_cooldown_seconds':5.,'min_composite_score':10.,'price_proximity_pct':.005}


def s1_timings(lane):
    """Require all ten named measurements, including the exact registered probe."""
    import math
    try:
        if set(lane['replays'])!=set(S1_REPLAY_NAMES) or len(lane['trial_equivalent_seconds'])!=3:
            raise ValueError('benchmark requires all ten S1 timings')
        if lane['worst_case_probe']['params']!=full_strategy_probe():
            raise ValueError('benchmark probe differs from registration')
        values={name:lane['replays'][name]['runtime_seconds'] for name in S1_REPLAY_NAMES}
        values.update({f'trial_equivalent_{i}':v for i,v in enumerate(lane['trial_equivalent_seconds'])})
        values['worst_case_probe']=lane['worst_case_probe']['runtime_seconds']
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values.values()):
            raise ValueError('benchmark timings must be finite and positive')
        return values
    except (KeyError,TypeError) as exc:
        raise ValueError('benchmark missing a timing or probe') from exc


def run_s1(prefix, comparator, *, symbol='BTCUSDT', guard=None):
    """Six replays, three default trial-equivalents and the rev 11 probe; no selection."""
    import time
    from contextlib import nullcontext
    from scripts import optimize_orderflow as opt, diagnose_trades as diagnose
    if not isinstance(prefix, SearchPrefix) or not isinstance(comparator, FrozenComparator):
        raise TypeError('S1 requires a SearchPrefix and FrozenComparator')
    defaults = full_strategy_defaults(symbol)
    cases = [('defaults', defaults, {}),
        ('initiative_off', defaults, {'initiative_imbalance':'off'}),
        ('k1', {**defaults,'ts_stacked_min_levels':1}, {}),
        ('E2_off', defaults, {'exhaustion_contrarian':'off'}),
        ('P2_bar_vwap', defaults, {'framing_profile_source':'bar_vwap'}),
        ('stageB_bar_params', {**defaults, **{k:v for k,v in comparator.params.items()
            if k in FULL_STRATEGY_SPACE and not k.startswith('ts_')}}, {})]
    rows, timings = {}, []
    for label, values, overrides in cases:
        with guard(label) if guard else nullcontext():
            with replay_measurement() as measurement:
                result = replay_full_strategy(FullStrategySample(values), prefix, symbol=symbol, signal_overrides=overrides)
                full = diagnose.add_gross(opt.simulate_trades_risk_based(result['ledger'],fee_bps=5.,slippage_bps=1.))
            rows[label] = {'decomposition':diagnose.decompose(full),'signal_counts':result['signal_counts'],
                'replay_digest':result['replay_digest'],**measurement}
    for index in range(3):
        with guard(f'benchmark-{index}') if guard else nullcontext():
            started=time.perf_counter()
            benchmark_trial_equivalent(prefix, defaults, symbol=symbol)
            timings.append(time.perf_counter()-started)
    probe=full_strategy_probe(symbol)
    with guard('worst_case_probe') if guard else nullcontext():
        started=time.perf_counter()
        benchmark_trial_equivalent(prefix,probe,symbol=symbol)
        probe_seconds=time.perf_counter()-started
    return {'replays':rows,'trial_equivalent_seconds':timings,'gate_input':False,
            'worst_case_probe':{'params':probe,'runtime_seconds':probe_seconds}}


@contextmanager
def replay_measurement():
    """Per-replay sampled RSS peak; strict lifetime RSS/private bounds remain in the watchdog."""
    import psutil
    import threading
    import time
    process=psutil.Process();done=threading.Event()
    measured={'peak_rss_bytes':process.memory_info().rss,'rss_metric':'per-replay RSS sampled every 20 ms'}
    def sample():
        measured['peak_rss_bytes']=max(measured['peak_rss_bytes'],process.memory_info().rss)
    def monitor():
        while not done.wait(.02):sample()
    started=time.perf_counter();thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:yield measured
    finally:
        done.set();thread.join();sample()
        measured['runtime_seconds']=time.perf_counter()-started


def benchmark_trial_equivalent(prefix, values, *, symbol='BTCUSDT'):
    """Measure the complete objective using fixed proposals, without an Optuna study."""
    from types import SimpleNamespace
    from scripts import optimize_orderflow as opt
    trial=SimpleNamespace(suggest_int=lambda name,*a:values[name],suggest_float=lambda name,*a,**kw:values[name],
        suggest_categorical=lambda name,*a:values[name],set_user_attr=lambda *a:None)
    args=SimpleNamespace(phase='phase4b',phase4=True,symbol=symbol,space=FULL_STRATEGY_SPACE,
        n_splits=5,embargo=.02,min_total_trades=105,min_trades_per_fold=11,fee_bps=5.,slippage_bps=1.)
    folds=opt.build_chronological_folds(prefix.bars,5,.02)
    return opt.objective(trial,prefix,folds,args)


def phase4b_reserved_look(layout, manifest, campaign_sha256, interval, candidate, report, replay):
    """Every entry and recovery checks gates and cross-campaign allocation under the look lock.

    The replay callback is the only place allowed to open the full cache/holdout.
    """
    from scripts.optimize_orderflow import file_lock, atomic_json, json_safe
    from scripts.phase4_approvals import require_holdout_allocation
    if layout.phase != 'phase4b' or manifest.get('phase') != 'phase4b':
        raise ValueError('holdout phase mismatch')
    path = layout.checked_write(layout.exposure)
    lock = layout.checked_write(str(path)+'.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(lock):
        from scripts.phase4b_campaign import s2_status
        if s2_status(layout,campaign_sha256,interval['bar_type'])!='complete':
            raise ValueError('look requires a completed S2 checkpoint for this arm')
        if (report.get('G4b_N') is not True or set(report.get('gates', {})) != {str(i) for i in range(1, 11)}
                or any(v is not True for v in report['gates'].values())
                or report.get('replay_digest')!=candidate.get('replay_digest')
                or report.get('campaign_sha256')!=campaign_sha256 or report.get('bar_type')!=interval['bar_type']):
            raise ValueError('candidate has not passed every G4b-N gate')
        require_holdout_allocation(layout, interval, campaign_sha256, manifest['phase4_origin'])
        arm = f"BTCUSDT:{interval['bar_type']}:full"
        data = json.loads(path.read_bytes()) if path.exists() else {}
        prior = data.get(arm)
        if prior and prior['candidate'] != candidate:
            raise ValueError('another candidate is already exposed')
        if prior and 'result' in prior:
            return prior['result']
        data[arm] = {'candidate':candidate, 'status':'reserved'}
        atomic_json(path, data)
        result = json_safe(replay())
        if prior:
            result['completed_after_interruption'] = True
        data[arm].update(status='complete', result=result)
        atomic_json(path, data)
        return result

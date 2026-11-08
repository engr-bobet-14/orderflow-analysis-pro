"""Sealed campaign registration and owner-gated host driver. Registration is not execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from orderflow_system.data.paired_dataset import sha256_file
from scripts.phase4_policy import checked_write, canonical_bytes, source_state, verify_source, protected_state, assert_no_conflicts
from scripts.phase4_studies import read_campaign, sequential_campaign


def runtime_limit(value):
    try:
        arm,seconds=value.rsplit('=',1);seconds=float(seconds)
        if not arm or not math.isfinite(seconds) or seconds<=0:raise ValueError('finite positive seconds required')
    except (ValueError,TypeError) as exc:
        raise argparse.ArgumentTypeError('runtime limit must be ARM=SECONDS with finite positive seconds') from exc
    return arm,seconds


def apply_runtime_limits(arms,trial_limits,study_limits):
    for name,limits in (('max_trial_seconds',trial_limits),('max_study_seconds',study_limits)):
        seen=set()
        for arm,seconds in limits:
            if arm not in arms or arms[arm]['kind']!='study' or arm in seen:
                raise ValueError('runtime limits require distinct registered study arms')
            if not math.isfinite(seconds) or seconds<=0:raise ValueError('runtime limits must be finite positive seconds')
            arms[arm][name]=seconds;seen.add(arm)


def register(path, snapshots=(),phase4_db=None,*,trial_limits=(),study_limits=()):
    path = checked_write(path)
    seal=Path(str(path)+'.sha256')
    if path.exists() != seal.exists():
        raise ValueError('half-sealed campaign: manifest and seal must both exist')
    if path.exists():
        stored=read_campaign(path)[0]
        arms={key:dict(arm) for key,arm in stored['arms'].items()}
        apply_runtime_limits(arms,trial_limits,study_limits)
        if arms!=stored['arms']:raise ValueError('registered runtime limits changed; register a new campaign')
        return stored
    from scripts.phase4_policy import phase4_db_path
    db=phase4_db_path(phase4_db if phase4_db is not None else ROOT/'output/optimize/phase4/studies.db')
    if not snapshots:
        snapshots=[{'path':str(ROOT/rel),'sha256':sha256_file(ROOT/rel)} for rel in
                   ('output/optimize/studies.db','output/optimize/main/studies.db','output/optimize/smoke/studies.db')]
    if len(snapshots)!=3 or len({Path(s['path']).resolve() for s in snapshots})!=3:
        raise ValueError('campaign requires three distinct legacy snapshot sources')
    for snapshot in snapshots:
        if sha256_file(Path(snapshot['path']))!=snapshot['sha256']:
            raise ValueError('legacy snapshot hash mismatch at registration')
    arms = {}
    for bt in ('fixed_tick_imbalance','fixed_volume_imbalance'):
        for mode in ('off','absorption_levels','exhaustion_bonus','exhaustion_required','framing_trades'):
            arms['BTCUSDT:'+bt+':'+mode] = {'kind':'candidate' if mode=='framing_trades' else 'study',
                'budget':200 if mode=='off' else 1 if mode=='framing_trades' else 100,
                'candidate_rule':'single deterministic frozen B0' if mode=='framing_trades' else 'best feasible trial after exact COMPLETE budget'}
    apply_runtime_limits(arms,trial_limits,study_limits)
    manifest = {'phase':'phase4','version':1,'arms':arms,'seed':42,'fee_bps':5.,'slippage_bps':1.,
                'min_total_trades':105,'min_trades_per_fold':11,'n_splits':5,'embargo':.02,'holdout_fraction':.2,
                'snapshots':list(snapshots),'phase4_db':str(db),'bootstrap':{'block_days':7,'resamples':2000,'seed':20261105},
                'CSCV_blocks':16,'stages':['A','B','C'],'owner_gated_optional':['C-prime'],
                'gates':{'G2':'gross lower CI > 0, joint improvement lower CI > 0, top3 > 0, PBO < .5, frequency <= 3, floors',
                         'G2-P2':'same gross, improvement, frequency and floors; PBO/top3 not applicable'},
                'source_state':source_state()}
    raw = canonical_bytes(manifest)
    path.parent.mkdir(parents=True,exist_ok=True)
    staged=[]
    try:
        for data in (raw,(hashlib.sha256(raw).hexdigest()+'\n').encode()):
            with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.campaign-',delete=False) as handle:
                staged.append(Path(handle.name))
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        # Publish the seal first: a crash can leave an orphan seal, never an unsealed manifest.
        os.replace(staged[1],seal)
        os.replace(staged[0],path)
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
    return manifest


def commands(path, manifest):
    sequential_campaign(manifest)
    commands = []
    for key,arm in manifest['arms'].items():
        symbol,bt,mode = key.split(':')
        command = [sys.executable,str(ROOT/'scripts/optimize_orderflow.py'),'--phase4','--symbol',symbol,'--bar-type',bt,
                   '--trade-signals',mode,'--campaign-manifest',str(path),'--no-warm-start','--n-jobs','1',
                   '--n-trials',str(arm['budget']),'--min-total-trades','105','--min-trades-per-fold','11',
                   '--study-db',manifest['phase4_db'],
                   '--output',str(ROOT/'output/optimize/phase4/results.json'),
                   '--cache-dir',str(ROOT/'output/optimize/phase4/cache')]
        if mode=='off':
            command += ['--search-space',str(ROOT/'scripts/search_spaces'/f'{bt}.json')]
        else:
            command += ['--trade-cache',str(ROOT/'output/optimize/trade_cache'),
                        '--freeze-params',str(ROOT/'output/phase4/frozen'/f'{bt}.json')]
        for name in ('max_trial_seconds','max_study_seconds'):
            if name in arm:command += ['--'+name.replace('_','-'),str(arm[name])]
        commands.append(command)
    # Baselines precede feature studies; feature arms cannot silently freeze a partial B0.
    return sorted(commands,key=lambda c:0 if c[c.index('--trade-signals')+1]=='off' else 1)


def protected_snapshot(paths):
    files = set()
    for path in paths:
        path = Path(path)
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(p for p in path.rglob('*') if p.is_file())
        else:
            raise ValueError(f'protected path missing: {path}')
    return {str(p.resolve()):sha256_file(p) for p in sorted(files)}


def preflight(manifest, protected_paths):
    if not Path('/Volumes/Market-Data').is_dir():
        raise ValueError('Market-Data mount missing')
    if shutil.disk_usage(ROOT).free < 100*1024**3:
        raise ValueError('preflight needs at least 100 GiB free disk')
    assert_no_conflicts()
    if source_state()!=manifest['source_state']:
        raise ValueError('campaign source_state changed; register a new campaign')
    return {**protected_state(),**protected_snapshot(protected_paths)}


def freeze_baseline(db, study_name, budget, output):
    from scripts import optimize_orderflow as opt
    from scripts.phase4_studies import completed_candidate
    opt.phase4_db_admission(db,study_name)
    trials = [t for t in opt.read_study_trials(db,study_name) if t['state']=='COMPLETE' and t['feasible']]
    if not trials:
        raise ValueError('no feasible trial')
    best = max(trials,key=lambda t:(t['value'],-t['number']))
    best,identity,reservation = completed_candidate(db,study_name,best['number'],budget)
    if identity['trade_signals']!='disabled':
        raise ValueError('freeze source must be B0')
    spec = {'study_db':str(db),'study':study_name,'trial':best['number'],'budget':budget,
            'params':best['attrs']['params'],'replay_digest':reservation['replay_digest']}
    output = checked_write(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() and json.loads(output.read_text())!=spec:
        raise ValueError('frozen B0 already exists with different values')
    opt.atomic_json(output,spec)


def clear_runtime_stop(db,study_name,selector):
    """Explicit owner recovery command, with an audit retained before clearing the stop."""
    import optuna
    from datetime import datetime,timezone
    from scripts import optimize_orderflow as opt
    from scripts.phase4_approvals import verify_runtime_decision
    from scripts.phase4_policy import phase4_db_path
    db=phase4_db_path(db);lock=checked_write(opt.study_lock_path(db,study_name))
    workers=checked_write(str(lock)+'.workers')
    with opt.phase4_execution(True),opt.file_lock(lock,nonblocking=True),opt.file_lock(workers,nonblocking=True):
        attrs=opt.read_study_attrs(db,study_name);identity=attrs.get('identity');stop=attrs.get('phase4_runtime_stop')
        if identity and identity.get('phase')=='phase4' and (attrs.get('phase4_accounting_pending') or attrs.get('phase4_memory_check_pending')):
            study=optuna.load_study(storage='sqlite:///'+str(db),study_name=study_name)
            opt.resolve_phase4_pending(study)
            attrs=study.user_attrs;stop=attrs.get('phase4_runtime_stop')
        if not identity or identity.get('phase')!='phase4' or not stop:
            raise ValueError('runtime decision requires a stopped Phase 4 study')
        verified=verify_runtime_decision(selector,identity,stop)
        study=optuna.load_study(storage='sqlite:///'+str(db),study_name=study_name)
        opt.reconcile_phase4_trials(study,SimpleNamespace(phase4=True))
        trials=opt.read_study_trials(db,study_name)
        audit=list(attrs.get('phase4_runtime_decisions',[]))
        if audit and audit[-1]['decision_sha256']==selector:
            entry=audit[-1]  # Repair interruption after audit persistence, before clearing.
        else:
            reset={'runtime_seconds':opt.phase4_runtime_baseline(study),
                   'last_trial_number':max((t['number'] for t in trials),default=-1)}
            entry={'decision_sha256':selector,'decision':verified['record'],'stop':stop,
                   'reset':reset,'utc':datetime.now(timezone.utc).isoformat()}
            study.set_user_attr('phase4_runtime_decisions',[*audit,entry])
        study.set_user_attr('phase4_runtime_reset',entry['reset'])
        study.set_user_attr('phase4_runtime_seconds',entry['reset']['runtime_seconds'])
        study.set_user_attr('phase4_runtime_stop',None)
        return entry


def ablation(owner_approved,campaign_manifest=None):
    from scripts import optimize_orderflow as opt
    with opt.phase4_execution(True):
        if not owner_approved:
            raise ValueError('Stage A requires later owner approval')
        from scripts import diagnose_trades as diagnose
        from scripts.phase4_studies import attach_cache, signals_for, require_gates
        if campaign_manifest is None:
            raise ValueError('Stage A requires a registered campaign')
        manifest,digest=read_campaign(campaign_manifest)
        verified=require_gates(owner_approved,digest)
        from scripts.phase4_resources import enforce_current
        enforce_current()
        preflight(manifest,())
        from scripts.phase4_benchmark import MODES
        from orderflow_system.config.settings import get_btcusd_config, TradeFlowConfig
        from orderflow_system.data.models import BarType
        base = get_btcusd_config()
        params = {**opt.baseline_extra_params(),'price_proximity_pct':.002,
                  'min_composite_score':base.risk.min_composite_score,
                  'signal_cooldown_seconds':base.risk.signal_cooldown_seconds,
                  'volume_decline_pct':base.exhaustion.volume_decline_pct,
                  'lookback_bars':base.divergence.lookback_bars,'delta_failure_pct':base.divergence.delta_failure_pct}
        reports = {}
        for bt in BarType:
            args = opt.build_parser().parse_args(['--phase4','--bar-type',bt.value,'--no-warm-start','--dry-run',
                '--min-total-trades','105','--min-trades-per-fold','11','--cache-dir',str(ROOT/'output/optimize/phase4/cache'),
                '--study-db',str(ROOT/'output/optimize/phase4/studies.db'),'--output',str(ROOT/'output/optimize/phase4/results.json')])
            args.symbol,args.bar_type = 'BTCUSDT',bt.value
            args.trade_cache = ROOT/'output/optimize/trade_cache'
            path,_ = opt.build_cache(args,args.symbol,args.bar_type)
            bars = opt.CachedBars(path,bt.value)
            attach_cache(args,bars)
            n_search,folds = opt.split_timeline(bars,args)
            search = bars[:n_search]
            reports[bt.value] = {}
            before = protected_state()
            for mode in MODES:
                result = opt.run_backtest(search,base,params['price_proximity_pct'],bt,
                    **({'trade_updates':args.trade_updates,'trade_config':TradeFlowConfig(signals=signals_for(mode))}
                       if mode!='off' else {}))
                trades = diagnose.add_gross(opt.simulate_trades_risk_based(result['ledger'],fee_bps=5.,slippage_bps=1.))
                reports[bt.value][mode] = {'signal_counts':result['signal_counts'],
                    'owner_approval_sha256':verified['sha256'],'gates_sha256':verified['gates_sha256'],
                    'decomposition':diagnose.decompose(trades),'replay_digest':result['replay_digest'],
                    'raw_signal_digest':result.get('raw_signal_digest'),
                    'folds':[opt.evaluate_fold(f,search,params,args,trades) for f in folds]}
            if protected_state()!=before:
                raise ValueError('protected set changed during Stage A')
            enforce_current()
            path = checked_write(ROOT/'output/phase4/ablation'/f'{bt.value}.json')
            path.parent.mkdir(parents=True,exist_ok=True)
            opt.atomic_json(path,reports[bt.value])
        return reports


def write_driver(path):
    path = checked_write(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    source = '''#!/usr/bin/env python3
"""Host entry point: review this driver and scripts/phase4_campaign.py before use."""
from pathlib import Path
import runpy
import sys
root = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root))
runpy.run_path(str(root/'scripts/phase4_campaign.py'),run_name='__main__')
'''
    path.write_text(source)
    return sha256_file(path)


def command_memory_plan(command,*,strict=False):
    """This driver launches one child at a time; Stage A holds one lane at a time."""
    from scripts.phase4_resources import memory_plan,require_memory_plan,mapped_cache_files
    from scripts.trade_feature_cache import production_generation,generation_directory
    from scripts.phase4_benchmark import DATASET_ID
    mode=command[command.index('--trade-signals')+1] if '--trade-signals' in command else 'off' if 'freeze' in command else 'ablation'
    files={};ready=True;source='bar-only: no TFC mapping'
    if mode!='off':
        path=Path(command[command.index('--trade-cache')+1]) if '--trade-cache' in command else ROOT/'output/optimize/trade_cache'
        try:
            generation=production_generation(generation_directory(path,DATASET_ID))
            bt=command[command.index('--bar-type')+1] if '--bar-type' in command else None
            lanes=[generation['lanes'][bt]] if bt else generation['lanes'].values()
            files=max((mapped_cache_files(p) for p in lanes),key=lambda f:sum(f.values()))
            source='COMPLETE generation, maximum one active lane'
        except ValueError:
            if strict:raise
            ready=False;source='production cache unavailable; diagnostic estimate only'
            benchmark=ROOT/'output/phase4/benchmark/benchmark.json'
            if benchmark.is_file():
                inputs=json.loads(benchmark.read_text())['extrapolation']['cache_open_inputs']
                files={'estimated-largest-lane':int(max(row['full_cache_bytes'] for row in inputs.values()))}
    plan=[{'mapped_files':files}]
    report=require_memory_plan(plan) if strict else memory_plan(plan)
    return {**report,'production_cache_ready':ready,'mapping_source':source,'planned_concurrency':1}


def stage_commands(path,manifest):
    plan=commands(path,manifest)
    baseline=[c for c in plan if c[c.index('--trade-signals')+1]=='off']
    features=[c for c in plan if c[c.index('--trade-signals')+1] not in ('off','framing_trades')]
    candidates=[c for c in plan if c[c.index('--trade-signals')+1]=='framing_trades']
    freezes=[]
    for command in baseline:
        bt=command[command.index('--bar-type')+1]
        freezes.append([sys.executable,str(ROOT/'scripts/phase4_campaign.py'),'freeze','--manifest',str(path),
                        '--bar-type',bt,'--out',str(ROOT/'output/phase4/frozen'/f'{bt}.json')])
    return {'A':[[sys.executable,str(ROOT/'scripts/phase4_campaign.py'),'ablation','--manifest',str(path)]],
            'B':baseline+freezes,'C':features+candidates}


def campaign_study(manifest,digest,arm):
    from scripts import optimize_orderflow as opt
    matches=[]
    db=Path(manifest['phase4_db'])
    if db.is_file():
        for name in opt.list_study_names(db):
            attrs=opt.read_study_attrs(db,name);identity=attrs.get('identity',{})
            if identity.get('phase')=='phase4' and identity.get('campaign_sha256')==digest and identity.get('arm')==arm:
                matches.append((name,attrs))
    if len(matches)!=1:raise ValueError('requires one completed campaign study for '+arm)
    return matches[0]


def frozen_source(manifest,path,digest,bt):
    from scripts import phase4_studies as p4
    args=SimpleNamespace(freeze_params=ROOT/'output/phase4/frozen'/f'{bt}.json',campaign_manifest=path,
        campaign_identity=digest,symbol='BTCUSDT',bar_type=bt,
        **{k:manifest[k] for k in ('holdout_fraction','n_splits','embargo')})
    spec=p4.read_freeze_spec(args)
    if Path(spec['study_db']).resolve()!=Path(manifest['phase4_db']).resolve():raise ValueError('frozen database mismatch')
    name,attrs=campaign_study(manifest,digest,'BTCUSDT:'+bt+':off')
    if name!=spec['study']:raise ValueError('frozen B0 study mismatch')
    best,_,reservation=p4.source_b0(args,spec)
    if best['attrs']['params']!=spec.get('params') or reservation['replay_digest']!=spec.get('replay_digest'):
        raise ValueError('frozen B0 parameters/replay digest mismatch')
    return args,attrs


def verify_frozen(manifest,path,digest,bt):
    from scripts import optimize_orderflow as opt,phase4_studies as p4,diagnose_trades as diagnose
    from scripts.dataset_policy import ensure_dev_window
    from datetime import date
    args,attrs=frozen_source(manifest,path,digest,bt)
    cache,meta=diagnose.find_cached_bars(ROOT/'output/optimize/phase4/cache',attrs['identity']['data_fp'])
    ensure_dev_window(date.fromisoformat(meta['start']),date.fromisoformat(meta['end']))
    if meta['bar_type']!=bt or meta['symbol']!='BTCUSDT':raise ValueError('frozen cache identity mismatch')
    p4.load_frozen(args,opt.CachedBars(cache,bt))  # Re-verify the stored search replay digest.
    from scripts.phase4_resources import enforce_current
    enforce_current()
    return args.frozen_identity


def freeze_registered(path,bt,output,owner_approved):
    from scripts import optimize_orderflow as opt
    from scripts.phase4_studies import require_gates
    manifest,digest=read_campaign(path)
    require_gates(owner_approved,digest)
    expected=checked_write(ROOT/'output/phase4/frozen'/f'{bt}.json')
    if checked_write(output)!=expected:raise ValueError('registered freeze output mismatch')
    with opt.phase4_execution(True):
        preflight(manifest,())
        command_memory_plan(['--trade-signals','off'],strict=True)
        name,_=campaign_study(manifest,digest,'BTCUSDT:'+bt+':off')
        opt.phase4_db_admission(manifest['phase4_db'],name)
        budget=manifest['arms']['BTCUSDT:'+bt+':off']['budget']
        with opt._shared_lock_readonly(opt.study_lock_path(manifest['phase4_db'],name)):
            freeze_baseline(manifest['phase4_db'],name,budget,expected)
            verify_frozen(manifest,path,digest,bt)
    return {'frozen':str(expected)}


def append_log(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('ab') as handle:
        handle.write(canonical_bytes(value));handle.flush();os.fsync(handle.fileno())


def atomic_bytes(path,raw):
    path=checked_write(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.stage-',delete=False) as handle:
        temporary=Path(handle.name);handle.write(raw);handle.flush();os.fsync(handle.fileno())
    try:os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def checkpoint_path(digest,stage,stopped=False):
    return checked_write(ROOT/'output/phase4/checkpoints'/digest/f'stage-{stage}{".stopped" if stopped else ""}.json')


def publish_checkpoint(record,log):
    path=checkpoint_path(record['campaign_sha256'],record['stage'],record['status']!='complete')
    raw=canonical_bytes(record);sha=hashlib.sha256(raw).hexdigest()
    # Preserve stopped attempts as well as successful checkpoints across retries.
    archive=path.parent/'records'/f'{sha}.json';atomic_bytes(archive,raw)
    append_log(log,{'status':'stage_complete' if record['status']=='complete' else 'stage_stopped',
        'stage':record['stage'],'campaign_sha256':record['campaign_sha256'],
        'owner_approval_sha256':record['owner_approval_sha256'],'checkpoint_sha256':sha,
        'checkpoint_path':str(path),'record_path':str(archive)})
    atomic_bytes(path,raw)  # Published last: an interrupted log/publish never exposes an unsealed checkpoint.
    return {'path':str(path),'sha256':sha}


def validate_checkpoint(manifest,digest,stage,protected,manifest_path):
    path=checkpoint_path(digest,stage)
    if not path.is_file():raise ValueError('required Stage '+stage+' checkpoint missing')
    raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest();record=json.loads(raw)
    log=ROOT/'output/phase4/campaign-log.jsonl'
    events=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    if not any(e.get('status')=='stage_complete' and e.get('stage')==stage and e.get('campaign_sha256')==digest
               and e.get('checkpoint_sha256')==sha for e in events):raise ValueError('checkpoint hash not sealed in campaign log')
    if (record.get('status')!='complete' or record.get('stage')!=stage or record.get('campaign_sha256')!=digest
            or record.get('source_state')!=source_state() or record.get('source_state')!=manifest['source_state']
            or record.get('source_state_after')!=record.get('source_state')
            or record.get('manifest_path')!=str(manifest_path)
            or record.get('protected_before')!=protected or record.get('protected_after')!=protected
            or record.get('next_stage_requires_owner_approval') is not True):raise ValueError('checkpoint source/protected state mismatch')
    for artifact in record['artifacts']:
        if sha256_file(Path(artifact['path']))!=artifact['sha256']:raise ValueError('checkpoint artifact hash mismatch')
    if any(c['status']!='complete' for c in record['commands']) or record['runtime_stops']:
        raise ValueError('checkpoint contains an incomplete command or runtime stop')
    if [r['command'] for r in record['commands']]!=stage_commands(manifest_path,manifest)[stage]:
        raise ValueError('checkpoint stage command plan mismatch')
    if record['child_approval_sha256s']!=[r['child_approval_sha256'] for r in record['commands']]:
        raise ValueError('checkpoint child approval mismatch')
    if stage in ('B','C'):
        previous=validate_checkpoint(manifest,digest,'A' if stage=='B' else 'B',protected,manifest_path)
        if record.get('prerequisite')!=previous:raise ValueError('checkpoint prerequisite hash mismatch')
    return {'path':str(path),'sha256':sha}


def command_outputs(command,manifest,digest,required):
    from scripts import optimize_orderflow as opt,phase4_studies as p4
    def field(name):return command[command.index(name)+1]
    result={'artifacts':[],'studies':[],'runtime_stops':[]}
    def artifact(path,**fields):
        if not path.is_file():
            if required:raise ValueError('command output artifact missing: '+str(path))
            return
        result['artifacts'].append({'path':str(path),'sha256':sha256_file(path),**fields})
    if 'ablation' in command:
        from orderflow_system.data.models import BarType
        from scripts.phase4_benchmark import MODES
        for bt in BarType:
            path=ROOT/'output/phase4/ablation'/f'{bt.value}.json'
            artifact(path,kind='ablation',bar_type=bt.value)
            if required and set(json.loads(path.read_bytes()))!=set(MODES):raise ValueError('ablation modes incomplete')
    elif 'freeze' in command:
        artifact(Path(field('--out')),kind='frozen',bar_type=field('--bar-type'))
        if required:frozen_source(manifest,field('--manifest'),digest,field('--bar-type'))
    else:
        arm=field('--symbol')+':'+field('--bar-type')+':'+field('--trade-signals')
        if manifest['arms'][arm]['kind']=='candidate':
            matches=[]
            for path in (ROOT/'output/phase4/candidates').glob('*.json'):
                data=p4.read_candidate(path.stem)
                if data['arm']==arm and data['campaign_sha256']==digest:matches.append(path)
            if required and len(matches)!=1:raise ValueError('requires one P2 candidate artifact, without a P2 study')
            for path in matches:artifact(path,kind='candidate',arm=arm)
        else:
            try:name,attrs=campaign_study(manifest,digest,arm)
            except ValueError:
                if required:raise
                return result
            trials=opt.read_study_trials(manifest['phase4_db'],name)
            counts={state:sum(t['state']==state for t in trials) for state in ('COMPLETE','FAIL','RUNNING','WAITING','PRUNED')}
            summary={'study_name':name,'arm':arm,'identity_sha256':opt.fingerprint(attrs['identity']),
                'counts':counts,**p4.trial_recovery(trials),'runtime_stop':attrs.get('phase4_runtime_stop'),
                'trials':[{'number':t['number'],'state':t['state'],'value':t['value'],
                           'params_sha256':opt.fingerprint(t['attrs'].get('params',t['params'])),
                           'replay_digest':t['attrs'].get('replay_digest')} for t in trials]}
            result['studies'].append(summary)
            if summary['runtime_stop'] is not None:result['runtime_stops'].append({'study_name':name,'stop':summary['runtime_stop']})
            raw=canonical_bytes(summary);sha=hashlib.sha256(raw).hexdigest()
            snapshot=ROOT/'output/phase4/checkpoints'/digest/'artifacts'/f'{sha}.json'
            atomic_bytes(snapshot,raw);artifact(snapshot,kind='study',study_name=name)
    return result


def run_driver(path, *, owner_approved=None, inspected_driver=None, driver=None, protected_paths=(), dry_run=True,stage=None):
    path=Path(path).expanduser().resolve()
    manifest,digest=read_campaign(path);stages=stage_commands(path,manifest)
    if stage is not None and stage not in stages:raise ValueError('unknown campaign stage')
    selected=stages if stage is None else {stage:stages[stage]}
    plan=[command for commands_ in selected.values() for command in commands_]
    if dry_run:
        return {'campaign_sha256':digest,'stage':stage,'stages':selected,'commands':plan,'concurrency':1,
                'next_stage_requires_owner_approval':True,'executed':False,
                'memory_plans':[command_memory_plan(c) for c in plan]}
    if stage is None:raise ValueError('--execute requires --stage A, B or C')
    if not owner_approved or not driver or sha256_file(driver)!=inspected_driver:
        raise ValueError('execution requires owner approval and independently inspected driver hash')
    from scripts import optimize_orderflow as opt
    from scripts.phase4_studies import require_gates
    from scripts.phase4_approvals import approval_for_command
    from scripts.phase4_resources import enforce_current,enforce_memory
    log=checked_write(ROOT/'output/phase4/campaign-log.jsonl')
    lock=checkpoint_path(digest,stage).parent/'driver.lock'
    with opt.file_lock(checked_write(lock),nonblocking=True,message='campaign stage already running'):
        verified=require_gates(owner_approved,digest)  # Driver argv includes the selected stage.
        before=preflight(manifest,protected_paths);begun=source_state();started=time.perf_counter()
        if checkpoint_path(digest,stage).exists():
            validate_checkpoint(manifest,digest,stage,before,path)
            raise ValueError('stage already has a valid checkpoint')
        events=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        checkpoints=[json.loads(p.read_bytes()) for p in
                     [*lock.parent.glob('stage-*.json'),*(lock.parent/'records').glob('*.json')]]
        if any(e.get('campaign_sha256')==digest and e.get('owner_approval_sha256')==owner_approved
               and e.get('status') in ('stage_started','stage_stopped','stage_complete','stopped','complete')
               for e in [*events,*checkpoints]):
            raise ValueError('stage approval already used; a new owner approval is required')
        append_log(log,{'status':'stage_started','stage':stage,'campaign_sha256':digest,
                        'owner_approval_sha256':verified['sha256'],'gates_sha256':verified['gates_sha256']})
        rows=[];artifacts=[];runtime_stops=[];prerequisite=None
        def record(status,error=None):
            try:after={**protected_state(),**protected_snapshot(protected_paths)};protection_error=None
            except Exception as exc:
                after={p:sha256_file(Path(p)) if Path(p).is_file() else None for p in before};protection_error=str(exc)
            return {'stage':stage,'status':status,'campaign_sha256':digest,'manifest_path':str(path),'owner_approval_sha256':verified['sha256'],
                'child_approval_sha256s':[r['child_approval_sha256'] for r in rows if r.get('child_approval_sha256')],
                'gates_sha256':verified['gates_sha256'],'source_state':begun,'source_state_after':source_state(),
                'protected_before':before,'protected_after':after,'protection_error':protection_error,
                'commands':rows,'artifacts':artifacts,'runtime_stops':runtime_stops,'prerequisite':prerequisite,
                'peak_rss_bytes':max((r['measurement']['peak_rss_bytes'] for r in rows if r.get('measurement')),default=None),
                'peak_private_bytes':max((r['measurement']['peak_private_bytes'] for r in rows if r.get('measurement') and r['measurement'].get('peak_private_bytes') is not None),default=None),
                'wall_seconds':time.perf_counter()-started,'next_stage_requires_owner_approval':True,'error':error}
        def gates():
            current=require_gates(owner_approved,digest)
            if current['gates_sha256']!=verified['gates_sha256']:raise ValueError('campaign gates changed during stage')
        try:
            enforce_current()
            if stage in ('B','C'):prerequisite=validate_checkpoint(manifest,digest,'A' if stage=='B' else 'B',before,path)
            if stage=='C':
                with opt.phase4_execution(True):
                    for command in stages['B']:
                        if 'freeze' in command:verify_frozen(manifest,path,digest,command[command.index('--bar-type')+1])
            for command in plan:
                row={'command':command,'status':'refused','child_approval_sha256':None,
                     'memory_preflight':{'pass':False,'status':'not_run'},'measurement':None}
                rows.append(row)
                try:
                    gates()
                    if source_state()!=begun:raise ValueError('source changed before command')
                    if {**protected_state(),**protected_snapshot(protected_paths)}!=before:raise ValueError('protected set changed before command')
                    child=approval_for_command(digest,command);row['child_approval_sha256']=child
                    require_gates(child,digest,command)
                    enforce_current()
                    try:row['memory_preflight']=command_memory_plan(command,strict=True)
                    except ValueError as exc:
                        row['memory_preflight']=getattr(exc,'report',{'pass':False,'error':str(exc)});raise
                    from scripts.measure_rss import run_measured
                    row['status']='stopped'
                    measurement=run_measured(command+['--owner-approved',child],memory=True);row['measurement']=measurement
                    outputs=command_outputs(command,manifest,digest,measurement['returncode']==0)
                    row.update(outputs);artifacts.extend(outputs['artifacts']);runtime_stops.extend(outputs['runtime_stops'])
                    if 'ablation' in command and measurement['returncode']==0:
                        for artifact in outputs['artifacts']:
                            modes=json.loads(Path(artifact['path']).read_bytes()).values()
                            if any(m.get('owner_approval_sha256')!=child or m.get('gates_sha256')!=verified['gates_sha256'] for m in modes):
                                raise ValueError('ablation output approval/gates mismatch')
                    measurement['memory_bounds']=enforce_memory(measurement,row['memory_preflight']['unique_mapped_bytes'])
                    if measurement['returncode']:raise ValueError('campaign child failed')
                    if runtime_stops:raise ValueError('campaign child has a runtime stop')
                    for study in outputs['studies']:
                        expected=manifest['arms'][study['arm']]['budget']
                        if study['counts']['COMPLETE']!=expected or study['counts']['RUNNING'] or study['counts']['WAITING']:
                            raise ValueError('campaign child requires exact COMPLETE budget and no pending trials')
                    gates()
                    if source_state()!=begun:raise ValueError('source changed during command')
                    if {**protected_state(),**protected_snapshot(protected_paths)}!=before:raise ValueError('protected set changed during command')
                    row['status']='complete'
                except BaseException as exc:row['error']=str(exc);raise
                finally:append_log(log,{'stage':stage,'campaign_sha256':digest,**row})
            gates();final=record('complete')
            if final['protected_before']!=final['protected_after'] or final['protection_error']:raise ValueError('protected set changed during stage')
            if final['source_state_after']!=begun:raise ValueError('source changed during stage')
            published=publish_checkpoint(final,log)
            return {'executed':True,'stage':stage,'checkpoint':published,'next_stage_requires_owner_approval':True}
        except BaseException as exc:
            publish_checkpoint(record('stopped',str(exc)),log)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('register','driver','freeze','ablation','clear-runtime-stop'))
    parser.add_argument('--manifest',type=Path,default=ROOT/'output/phase4/campaign.json')
    parser.add_argument('--owner-approved')
    parser.add_argument('--inspected-driver')
    parser.add_argument('--driver',type=Path,default=ROOT/'output/phase4/run_scripts/driver.py')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--stage',choices=('A','B','C'))
    parser.add_argument('--bar-type')
    parser.add_argument('--protected',type=Path,action='append',default=[])
    parser.add_argument('--study-db',type=Path)
    parser.add_argument('--study-name')
    parser.add_argument('--budget',type=int,default=200)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--snapshots',type=Path,help='JSON list of three legacy database paths and SHA256 values')
    parser.add_argument('--max-trial-seconds',type=runtime_limit,action='append',default=[],metavar='ARM=SECONDS')
    parser.add_argument('--max-study-seconds',type=runtime_limit,action='append',default=[],metavar='ARM=SECONDS')
    parser.add_argument('--runtime-decision',help='host runtime-decision record SHA256')
    cli = parser.parse_args(argv)
    try:
        if cli.command!='register' and (cli.max_trial_seconds or cli.max_study_seconds):
            raise ValueError('ARM=SECONDS runtime limits are registration options only')
        if cli.command=='driver' and cli.execute and not cli.stage:raise ValueError('--execute requires --stage A, B or C')
        if cli.command!='driver' and cli.stage:raise ValueError('--stage is a driver option')
        if cli.command=='clear-runtime-stop':
            if not cli.study_db or not cli.study_name:raise ValueError('runtime clear requires --study-db and --study-name')
            return clear_runtime_stop(cli.study_db,cli.study_name,cli.runtime_decision)
        if cli.command=='ablation':
            return ablation(cli.owner_approved,cli.manifest)
        if cli.command=='register':
            register(cli.manifest,json.loads(cli.snapshots.read_text()) if cli.snapshots else (),
                     trial_limits=cli.max_trial_seconds,study_limits=cli.max_study_seconds)
            return {'driver_sha256':write_driver(cli.driver),'manifest':str(cli.manifest)}
        if cli.command=='freeze' and cli.bar_type:
            return freeze_registered(cli.manifest,cli.bar_type,cli.out,cli.owner_approved)
        if cli.command=='freeze':
            freeze_baseline(cli.study_db,cli.study_name,cli.budget,cli.out)
            return {'frozen':str(cli.out)}
        return run_driver(cli.manifest,owner_approved=cli.owner_approved,inspected_driver=cli.inspected_driver,
                          driver=cli.driver,protected_paths=cli.protected,dry_run=not cli.execute,stage=cli.stage)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__=='__main__':
    print(json.dumps(main(),indent=2))

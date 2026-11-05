"""Sealed campaign registration and owner-gated host driver. Registration is not execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from orderflow_system.data.paired_dataset import sha256_file
from scripts.phase4_policy import checked_write, canonical_bytes, source_state, verify_source, protected_state, assert_no_conflicts
from scripts.phase4_studies import read_campaign


def register(path, snapshots=(),phase4_db=None):
    path = checked_write(path)
    seal=Path(str(path)+'.sha256')
    if path.exists() != seal.exists():
        raise ValueError('half-sealed campaign: manifest and seal must both exist')
    if path.exists():
        return read_campaign(path)[0]
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


def ablation(owner_approved,campaign_manifest=None):
    if not owner_approved:
        raise ValueError('Stage A requires later owner approval')
    from scripts import optimize_orderflow as opt, diagnose_trades as diagnose
    from scripts.phase4_studies import attach_cache, signals_for, require_gates
    if campaign_manifest is None:
        raise ValueError('Stage A requires a registered campaign')
    manifest,digest=read_campaign(campaign_manifest)
    verified=require_gates(owner_approved,digest)
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
    mode=command[command.index('--trade-signals')+1] if '--trade-signals' in command else 'ablation'
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


def run_driver(path, *, owner_approved=None, inspected_driver=None, driver=None, protected_paths=(), dry_run=True):
    manifest,digest = read_campaign(path)
    plan = commands(path,manifest)
    if dry_run:
        return {'campaign_sha256':digest,'commands':plan,'stage_a':'36 fixed-default replays before Stage B',
                'freeze':'freeze each completed B0 before feature studies', 'executed':False,
                'memory_plans':[command_memory_plan(command) for command in plan]}
    if not owner_approved or not driver or sha256_file(driver)!=inspected_driver:
        raise ValueError('execution requires owner approval and independently inspected driver hash')
    from scripts.phase4_studies import require_gates
    from scripts.phase4_approvals import approval_for_command
    verified=require_gates(owner_approved,digest)
    before = preflight(manifest,protected_paths)
    log = checked_write(ROOT/'output/phase4/campaign-log.jsonl')
    log.parent.mkdir(parents=True,exist_ok=True)
    def execute(command):
        before_run = {**protected_state(),**protected_snapshot(protected_paths)}
        if before_run!=before:
            raise ValueError('protected set changed before run')
        begun_state = source_state()
        result = None
        child_approval=None
        try:
            from scripts.measure_rss import run_measured
            from scripts.phase4_resources import enforce_memory
            memory=command_memory_plan(command,strict=True)
            child_approval=approval_for_command(digest,command)
            result = run_measured(command+['--owner-approved',child_approval],memory=True)
            result['memory_admission']=memory
            result['memory_bounds']=enforce_memory(result,memory['unique_mapped_bytes'])
            if result['returncode']:
                raise ValueError('campaign child failed')
            status = 'complete'
        except BaseException:
            status = 'stopped'
            raise
        finally:
            protection_error = None
            try:
                after = {**protected_state(),**protected_snapshot(protected_paths)}
                unchanged = after == before
            except Exception as exc:
                unchanged,protection_error = False,str(exc)
            source_unchanged = source_state() == begun_state
            with log.open('a') as handle:
                handle.write(json.dumps({'command':command,'status':status,'campaign_sha256':digest,
                    'owner_approval_sha256':verified['sha256'],'child_approval_sha256':child_approval,
                    'gates_sha256':verified['gates_sha256'],
                    'protected_unchanged':unchanged,'protection_error':protection_error,
                    'source_unchanged':source_unchanged,'measurement':result})+'\n')
            if not unchanged:
                raise ValueError('protected set changed during campaign')
            verify_source(begun_state)
    execute([sys.executable,str(Path(__file__)),'ablation','--manifest',str(path)])
    for command in plan:
        execute(command)
        if command[command.index('--trade-signals')+1]=='off':
            bt = command[command.index('--bar-type')+1]
            output = json.loads((ROOT/'output/optimize/phase4/results.json').read_text())
            studies = output['symbols']['BTCUSDT'][bt]
            matching = [name for name,entry in studies.items() if entry['identity'].get('campaign_sha256')==digest
                        and entry['identity']['trade_signals']=='disabled']
            if len(matching)!=1:
                raise ValueError('cannot identify the completed campaign B0')
            freeze_baseline(ROOT/'output/optimize/phase4/studies.db',matching[0],200,
                             ROOT/'output/phase4/frozen'/f'{bt}.json')
    with log.open('a') as handle:
        handle.write(json.dumps({'status':'studies_complete_reports_and_G2_required','campaign_sha256':digest})+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('register','driver','freeze','ablation'))
    parser.add_argument('--manifest',type=Path,default=ROOT/'output/phase4/campaign.json')
    parser.add_argument('--owner-approved')
    parser.add_argument('--inspected-driver')
    parser.add_argument('--driver',type=Path,default=ROOT/'output/phase4/run_scripts/driver.py')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--protected',type=Path,action='append',default=[])
    parser.add_argument('--study-db',type=Path)
    parser.add_argument('--study-name')
    parser.add_argument('--budget',type=int,default=200)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--snapshots',type=Path,help='JSON list of three legacy database paths and SHA256 values')
    cli = parser.parse_args(argv)
    try:
        if cli.command=='ablation':
            return ablation(cli.owner_approved,cli.manifest)
        if cli.command=='register':
            register(cli.manifest,json.loads(cli.snapshots.read_text()) if cli.snapshots else ())
            return {'driver_sha256':write_driver(cli.driver),'manifest':str(cli.manifest)}
        if cli.command=='freeze':
            freeze_baseline(cli.study_db,cli.study_name,cli.budget,cli.out)
            return {'frozen':str(cli.out)}
        return run_driver(cli.manifest,owner_approved=cli.owner_approved,inspected_driver=cli.inspected_driver,
                          driver=cli.driver,protected_paths=cli.protected,dry_run=not cli.execute)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__=='__main__':
    print(json.dumps(main(),indent=2))

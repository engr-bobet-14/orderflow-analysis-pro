"""Bounded Phase 4 timings in measured children. Never opens Optuna storage."""
from __future__ import annotations

import argparse
import ctypes
from datetime import date, timedelta
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orderflow_system.config.settings import TradeFlowConfig, get_btcusd_config, parse_trade_signals
from orderflow_system.data.paired_dataset import open_registered
from scripts.phase4_policy import checked_write, canonical_bytes, source_state, verify_source, bounded_preflight, protected_state
from scripts.measure_rss import run_measured
from scripts.phase4_resources import (current_memory,enforce_memory,mapped_cache_files,
    memory_plan,require_memory_plan)

DATASET_ID = '20261103T011635Z-d3a7b9603b37'
MODES = ('off', 'absorption_levels', 'exhaustion_bonus', 'exhaustion_required', 'framing_trades',
         'absorption_levels,exhaustion_bonus,framing_trades')


def validate_days(value):
    days = [date.fromisoformat(d) for d in value.split(',')]
    if not days or any(d<date(2019,9,10) or d>date(2023,12,31) for d in days):
        raise ValueError('benchmark days must be inside the development window')
    return days


def resident_cache_bytes(cache):
    """mincore measures currently resident mapped array pages, after authenticated loading."""
    import os
    page = os.sysconf('SC_PAGE_SIZE')
    libc = ctypes.CDLL(None, use_errno=True)
    resident, mapped = 0, 0
    for array in [*cache.arrays.values(),cache.bars().array]:
        if not array.nbytes:
            continue
        address = array.ctypes.data
        aligned = address//page*page
        length = address-aligned+array.nbytes
        count = math.ceil(length/page)
        vector = (ctypes.c_ubyte*count)()
        if libc.mincore(ctypes.c_void_p(aligned), ctypes.c_size_t(length), vector) != 0:
            return {'mapped_bytes':sum(a.nbytes for a in cache.arrays.values()),
                    'resident_bytes':None, 'unavailable':'mincore failed', 'errno':ctypes.get_errno()}
        resident += sum(v&1 for v in vector)*page
        mapped += count*page
    return {'mapped_bytes':mapped, 'resident_bytes':resident, 'measurement':'mincore mapped array pages'}


def extrapolate(rows, full_trades, full_bars, *, full_days):
    import numpy as np
    processed=np.array([r['verified_trade_count'] for r in rows],dtype=float)
    wall=np.array([r['build_seconds'] for r in rows],dtype=float)
    # Scale the predictor for numerical conditioning; retain ordinary least squares coefficients.
    design=np.column_stack((np.ones(len(rows)),processed/1e6))
    coefficients,_,rank,_=np.linalg.lstsq(design,wall,rcond=None)
    if rank<2:
        raise ValueError('build fit needs distinct authenticated processed trade counts')
    fixed,per_trade=float(coefficients[0]),float(coefficients[1]/1e6)
    full_build=fixed+per_trade*full_trades
    if full_build<0 or per_trade<0:
        raise ValueError('build fit cannot extrapolate a negative cost')
    components=[r['disk_components'] for r in rows]
    bar_n=np.array([sum(r['bar_counts'].values()) for r in rows],dtype=float)
    day_n=np.array([c['finalised_days'] for c in components],dtype=float)
    fixed_bytes=float(np.mean([c['fixed_bytes'] for c in components]))
    per_bar=float(np.dot(bar_n,[c['bar_bytes'] for c in components])/np.dot(bar_n,bar_n))
    if not np.dot(day_n,day_n):
        raise ValueError('disk fit needs observed finalised-day components')
    per_day=float(np.dot(day_n,[c['day_bytes'] for c in components])/np.dot(day_n,day_n))
    disk=fixed_bytes+per_bar*sum(full_bars.values())+per_day*full_days
    build_upper=full_trades*max(r['build_seconds']/r['trade_count'] for r in rows)
    disk_upper=max(sum(full_bars.values())*max(r['cache_bytes']/sum(r['bar_counts'].values()) for r in rows),
                   max(c['fixed_bytes'] for c in components)+sum(full_bars.values())*max(c['bar_bytes']/n for c,n in zip(components,bar_n))
                   +full_days*max(c['day_bytes']/n for c,n in zip(components,day_n) if n))
    trial_seconds={}
    for bt,n in full_bars.items():
        modes=set.intersection(*(set(r['replay_seconds'].get(bt,{})) for r in rows))
        trial_seconds[bt]={mode:.8*n*max(r['replay_seconds'][bt][mode]/r['bar_counts'][bt] for r in rows) for mode in modes}
    peak=max(r['peak_rss_bytes'] for r in rows)
    open_seconds,open_inputs={},{}
    for bt,n in full_bars.items():
        measured=[r['cache_open'][bt] for r in rows]
        hash_rate=max(t['hash_seconds']/t['hash_bytes'] for t in measured)
        digest_rate=max(t['digest_seconds']/t['bars'] for t in measured)
        other=max(t['other_seconds'] for t in measured)
        lane_bytes=fixed_bytes/len(full_bars)+per_bar*n+per_day*full_days/len(full_bars)
        open_seconds[bt]=hash_rate*lane_bytes+digest_rate*n+other
        open_inputs[bt]={'full_cache_bytes':lane_bytes,'hash_seconds_per_byte':hash_rate,
                        'digest_seconds_per_bar':digest_rate,'other_seconds':other}
    study_wall={bt:{m:((200 if m=='off' else 1 if m=='framing_trades' else 100)*seconds
                       +(open_seconds[bt] if m!='off' else 0.)) for m,seconds in modes.items() if m in MODES[:5]}
                for bt,modes in trial_seconds.items()}
    stage_a=sum(sum(modes.get(m,0.) for m in MODES) for modes in trial_seconds.values())+sum(open_seconds.values())
    selected=[b for b in full_bars if b in ('fixed_tick_imbalance','fixed_volume_imbalance')]
    stage_b=sum(study_wall[b]['off'] for b in selected)
    stage_c=sum(sum(study_wall[b][m] for m in MODES[1:5]) for b in selected)
    campaign_open=sum(open_seconds.values())+4*sum(open_seconds[b] for b in selected)
    ratios={bt:{m:v/modes['off'] if modes['off'] else None for m,v in modes.items()} for bt,modes in trial_seconds.items()}
    observed_ratios={r['day'] if 'day' in r else str(i):{bt:{m:t/modes['off'] for m,t in modes.items()}
                      for bt,modes in r['replay_seconds'].items()} for i,r in enumerate(rows)}
    amortized_ratios={bt:{m:(study_wall[bt][m]/(200 if m=='off' else 1 if m=='framing_trades' else 100))/modes['off']
                         for m in study_wall[bt]} for bt,modes in trial_seconds.items()}
    speed_ok=(all(v<=3 for day in observed_ratios.values() for modes in day.values() for v in modes.values())
              and all(v<=3 for modes in amortized_ratios.values() for v in modes.values()))
    planned_maps={bt:{'estimated-full-cache/'+bt:int(math.ceil(inputs['full_cache_bytes']))} for bt,inputs in open_inputs.items()}
    serial_plan=memory_plan([{'mapped_files':max(planned_maps.values(),key=lambda f:sum(f.values()))}])
    six_plan=memory_plan([{'mapped_files':planned_maps[bt]} for _ in range(3) for bt in selected])
    checks=[r['build_memory_bounds'] for r in rows if 'build_memory_bounds' in r]
    checks += [m['memory_bounds'] for r in rows for m in r.get('measurements',[]) if 'memory_bounds' in m]
    memory_ok=bool(checks) and all(c['pass'] for c in checks)
    private=[r['peak_private_bytes'] for r in rows if r.get('peak_private_bytes') is not None]
    gates={'build_under_4h':full_build<=4*3600,'disk_under_40GiB':disk<=40*1024**3,
           'worker_under_4GiB':memory_ok,'sum_under_24GiB':six_plan['pass'],
           'campaign_total_memory':serial_plan['pass'],'arms_under_3x':speed_ok}
    return {'build_wall_seconds':full_build,'disk_bytes':disk,'peak_rss_bytes_observed':peak,
        'build_fit':{'method':'ordinary least squares: fixed + seconds_per_trade * processed_trades','rank':int(rank),
                     'fixed_seconds':fixed,'seconds_per_trade':per_trade,'residuals_seconds':(wall-design@coefficients).tolist(),
                     'negative_intercept':fixed<0},
        'disk_fit':{'method':'separate component least squares: fixed + bytes_per_bar * bars + bytes_per_finalised_day * lane_day_events',
                    'fixed_bytes':fixed_bytes,'bytes_per_bar':per_bar,'bytes_per_finalised_day':per_day,
                    'joint_design_rank':int(np.linalg.matrix_rank(np.column_stack((np.ones(len(rows)),bar_n,day_n)))),
                    'reason_for_component_fit':'equal-duration windows cannot identify a joint fixed/day fit; physical components are measured separately',
                    'residuals_bytes':[r['cache_bytes']-(fixed_bytes+per_bar*b+per_day*d) for r,b,d in zip(rows,bar_n,day_n)]},
        'conservative_upper_bound':{'build_wall_seconds':build_upper,'disk_bytes':disk_upper,
            'resource_gates':{**gates,'build_under_4h':build_upper<=4*3600,'disk_under_40GiB':disk_upper<=40*1024**3}},
        'phase3_empirical_anchor':{'wall_seconds':[8063,8804],'source':'approved plan A4 and host fix item 9; paired replay without cache writing'},
        'rss_extrapolation':'max measured child peak only; no linear full-history RSS prediction is justified',
        'peak_private_bytes_observed':max(private) if private else None,
        'memory_accounting':{'observed_bounds':checks,'serial_campaign_plan':serial_plan,'six_study_plan':six_plan,
            'six_worker_lower_bound_admissible':six_plan['pass'],
            'private_extrapolation':'4 GiB planned private budget per process; bounded peaks are not full-history measurements',
            'mapped_extrapolation':'component-fit lane bytes, shared files deduplicated across planned processes'},
        'trial_seconds':trial_seconds,'arm_vs_bar_only_ratios':ratios,'observed_arm_vs_bar_only_ratios':observed_ratios,
        'cache_open_seconds':open_seconds,'cache_open_inputs':open_inputs,'study_wall_seconds':study_wall,
        'campaign_open_seconds':campaign_open,'cache_open_processes':{'stage_a':len(full_bars),'stage_b':0,'stage_c':4*len(selected)},
        'amortized_arm_vs_bar_only_ratios':amortized_ratios,
        'observed_speed_gate':{day:{bt:{mode:ratio<=3 for mode,ratio in modes.items()} for bt,modes in types.items()}
                               for day,types in observed_ratios.items()},
        'campaign_cpu_wall_seconds':{'stage_a':stage_a,'stage_b':stage_b,'stage_c':stage_c},
        'campaign_wall_seconds_6_workers_lower_bound':stage_a+stage_b/2+stage_c/6,'sum_worker_peaks_6_bytes':6*peak,
        'resource_gates':gates,'gate_model':'fitted estimates plus observed replay and amortized study ratios including cache open; conservative-bound gates also reported',
        'inputs':{'full_trades':full_trades,'full_bar_counts':full_bars,'full_finalised_lane_day_events':full_days,
                  'samples':[{'processed_trades':r['verified_trade_count'],'requested_day_trades':r['trade_count'],
                              'wall_seconds':r['build_seconds'],'bars':int(b),'lane_days':int(d),'disk_components':c}
                             for r,b,d,c in zip(rows,bar_n,day_n,components)]},
        'formulas':['wall = OLS fixed + OLS seconds_per_processed_trade * full_authenticated_trades',
                    'disk = mean(metadata_bytes) + LS(bar_component_bytes / bar_count) * full_bars + LS(day_component_bytes / lane_day_events) * full_lane_day_events',
                    'conservative wall = full_trades * max(whole_child_wall / requested_day_trades)',
                    'conservative disk = max(whole_artifact_per_bar_bound, componentwise_max_rate_bound)',
                    'trial wall = 0.8 * full_bar_count * max(sample_replay_seconds / sample_bar_count)',
                    'cache open per process = max(hash_seconds / hashed_bytes) * estimated lane bytes + max(digest_seconds / bars) * full lane bars + max(other_seconds)',
                    'study wall = budget * trial wall + one cache open (trade arms only; B0 has no TFC)',
                    'Stage A = sum(six arms, six bar types) + one open per type; B = sum(two B0 study walls); C = sum(six feature study walls + two P2 single-evaluation walls)',
                    'campaign open overhead = sum(all Stage A lane opens) + 4 * sum(two Stage C lane opens)',
                    'six-worker campaign lower bound = Stage A sequential + Stage B/2 + Stage C/6'],
        'limitations':['three fixed-default bounded days, no studies/trials; fits are advisory',
                       'negative OLS intercept, if present, is reported without clamping or hiding residuals',
                       'processed trade counts include authenticated causal warm-up and hour overscan',
                       'disk day coefficients use separately measured finalised-profile components, not an unidentifiable joint fit',
                       'objective/SQLite/reporting overhead excluded; page-cache and RSS growth require full-history gates']}


def disk_components(paths,report):
    import numpy as np
    from scripts.trade_feature_cache import ARRAYS
    bar=day=total=events=0
    for bt,path in paths.items():
        directory=Path(path)
        total+=sum(p.stat().st_size for p in directory.iterdir())
        arrays={name:np.load(directory/(name+'.npy'),mmap_mode='r') for name in ARRAYS}
        bar+=sum(a.nbytes for name,a in arrays.items() if not name.startswith('day_'))-8+(directory/'bars.bin').stat().st_size
        day+=sum(a.nbytes for name,a in arrays.items() if name.startswith('day_'))-8
        events+=report['lanes'][bt]['finalised_days']
    return {'fixed_bytes':total-bar-day,'bar_bytes':bar,'day_bytes':day,'finalised_days':events}


def _child(cli):
    from scripts import trade_feature_cache as tfc, optimize_orderflow as opt
    target = checked_write(cli.child_out)
    if cli.child == 'build':
        day = validate_days(cli.day)[0]
        ds = open_registered(Path(TradeFlowConfig().paired_root),DATASET_ID)
        paths,report = tfc.build(ds,cli.out/'caches'/cli.day,start=str(day),end=str(day+timedelta(days=1)))
        result = {'paths':{k:str(v) for k,v in paths.items()},'report':report}
    else:
        cache = tfc.CachedTradeUpdates(cli.cache) if cli.mode!='off' else None
        metadata=json.loads((cli.cache/'manifest.json').read_bytes())
        bars = cache.bars() if cache is not None else opt.CachedBars(cli.cache/'bars.bin',metadata['bar_type'])
        cfg = TradeFlowConfig(signals=parse_trade_signals(cli.mode) if cli.mode!='off' else TradeFlowConfig().signals)
        begun = time.perf_counter()
        result = opt.run_backtest(bars,get_btcusd_config(),.002,bars[0].bar_type,
                                   **({'trade_updates':cache,'trade_config':cfg} if cli.mode!='off' else {}))
        result = {k:v for k,v in result.items() if k!='ledger'} | {'replay_seconds':time.perf_counter()-begun,
                  'cache_open':cache.open_timings if cache else None,
                  'page_cache':resident_cache_bytes(cache) if cache else {'mapped_bytes':0,'resident_bytes':0},
                  'memory':current_memory(),
                  'mapped_cache_bytes':sum(mapped_cache_files(cli.cache).values()) if cache else 0}
        result['memory_bounds']=enforce_memory(result['memory'],result['mapped_cache_bytes'])
        if cache:
            import struct
            maximum=int(cache.arrays['fp_0'].max()) if len(cache.arrays['fp_0']) else 0
            equal=struct.pack('d',round(maximum/100,10))==struct.pack('d',maximum/100)
            if not equal:raise ValueError('bounded-cache maximum tick rounding equality failed')
            result['max_tick_round_equality']={'max_tick':maximum,'equal_float_bits':equal}
    target.write_bytes(canonical_bytes(result))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days',default='2019-09-11,2021-05-19,2023-10-16')
    parser.add_argument('--out',type=Path,default=ROOT/'output/phase4/benchmark')
    parser.add_argument('--bar-cache-manifests',type=Path,default=ROOT/'output/phase4/inputs/bar-cache-manifests')
    parser.add_argument('--child',choices=('build','replay'),help=argparse.SUPPRESS)
    parser.add_argument('--child-out',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--cache',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--day',help=argparse.SUPPRESS)
    parser.add_argument('--mode',choices=MODES,help=argparse.SUPPRESS)
    cli = parser.parse_args(argv)
    days = validate_days(cli.days)
    cli.out = checked_write(cli.out)
    if cli.child:
        return _child(cli)
    state = source_state()
    protected_before = bounded_preflight()
    from scripts.phase4_policy import assert_no_conflicts
    process_check=assert_no_conflicts(strict=False)
    # Authenticate counts before creating outputs or children.
    ds = open_registered(Path(TradeFlowConfig().paired_root),DATASET_ID)
    full_counts, count_sources = {}, []
    for path in sorted(cli.bar_cache_manifests.glob('*.json')):
        meta = json.loads(path.read_text())
        if (meta.get('bar_type') in ds.bar_types and meta.get('start')=='2019-09-10'
                and meta.get('end')=='2023-12-31'):
            bt = meta['bar_type']
            import pyarrow.parquet as pq
            authenticated_count = pq.read_metadata(ds.bar_path(bt)).num_rows
            if meta['n_bars']!=authenticated_count:
                raise ValueError('bar-cache count differs from authenticated paired bar count')
            full_counts[bt] = meta['n_bars']
            count_sources.append({'path':str(path),'sha256':tfc_sha(path),'n_bars':meta['n_bars']})
    if set(full_counts)!=set(ds.bar_types):
        raise ValueError('full development bar-cache manifests required for all six types; use --bar-cache-manifests')
    cli.out.mkdir(parents=True,exist_ok=True)
    samples = []
    for day in days:
        child_out = checked_write(cli.out/(str(day)+'-build.json'))
        build_admission=require_memory_plan([{'mapped_files':{}}])
        measured = run_measured([sys.executable,str(Path(__file__)), '--child','build','--day',str(day),
                                 '--out',str(cli.out),'--child-out',str(child_out)],memory=True)
        if measured['returncode']:
            raise RuntimeError('bounded cache child failed')
        build = json.loads(child_out.read_text())
        measured['peak_private_bytes']=max(measured.get('peak_private_bytes') or 0,build['report']['memory']['peak_private_bytes'] or 0) or None
        if build['report']['memory']['peak_private_bytes'] is not None:
            measured['private_is_fallback']=False;measured['private_metric']=build['report']['memory']['private_metric']
        build_bounds=enforce_memory(measured)
        row = {'day':str(day),'build_seconds':measured['wall_seconds'],'cache_bytes':build['report']['bytes'],
               'peak_rss_bytes':measured['peak_rss_bytes'],'bar_counts':{},'replay_seconds':{},'measurements':[],'cache_open':{},
               'peak_private_bytes':measured['peak_private_bytes'],'build_memory_bounds':build_bounds,'build_memory_admission':build_admission,
               'verified_trade_count':build['report']['stream']['n_verified'],
               'disk_components':disk_components(build['paths'],build['report']),
               'trade_count':int(ds.hours.loc[ds.hours.hour_ms//86400000==int(pd_day(day)), 'n_trades'].sum())}
        for bt,path in build['paths'].items():
            row['bar_counts'][bt] = build['report']['lanes'][bt]['bars']
            row['replay_seconds'][bt] = {}
            for i,mode in enumerate(MODES):
                result_path = checked_write(cli.out/(str(day)+'-'+bt+'-'+str(i)+'.json'))
                files=mapped_cache_files(path) if mode!='off' else {}
                admission=require_memory_plan([{'mapped_files':files}])
                measured = run_measured([sys.executable,str(Path(__file__)), '--child','replay','--cache',path,
                    '--mode',mode,'--out',str(cli.out),'--child-out',str(result_path)],memory=True)
                if measured['returncode']:
                    raise RuntimeError('bounded replay child failed')
                result = json.loads(result_path.read_text())
                measured['peak_private_bytes']=max(measured.get('peak_private_bytes') or 0,result['memory']['peak_private_bytes'] or 0) or None
                if result['memory']['peak_private_bytes'] is not None:
                    measured['private_is_fallback']=False;measured['private_metric']=result['memory']['private_metric']
                bounds=enforce_memory(measured,sum(files.values()))
                if result['cache_open']:
                    current=row['cache_open'].setdefault(bt,result['cache_open'].copy())
                    for component in ('hash_seconds','digest_seconds','other_seconds','total_seconds'):
                        current[component]=max(current[component],result['cache_open'][component])
                row['replay_seconds'][bt][mode] = result['replay_seconds']
                row['peak_rss_bytes'] = max(row['peak_rss_bytes'],measured['peak_rss_bytes'])
                row['peak_private_bytes']=max(row['peak_private_bytes'] or 0,measured['peak_private_bytes'] or 0) or None
                row['measurements'].append({'bar_type':bt,'mode':mode,**measured,**result,'memory_bounds':bounds,'memory_admission':admission})
        row['bytes_per_trade'] = row['cache_bytes']/row['trade_count']
        row['bytes_per_bar'] = row['cache_bytes']/sum(row['bar_counts'].values())
        samples.append(row)
        (cli.out/'progress.json').write_bytes(canonical_bytes({'completed_days':[r['day'] for r in samples]}))
    verify_source(state)
    if protected_state() != protected_before:
        raise ValueError('protected set changed during benchmark')
    spill_files = [str(p) for p in (ROOT/'output/duckdb_tmp').rglob('*') if p.is_file()]
    if spill_files:
        raise ValueError('DuckDB spill files remain after benchmark')
    result = {'source_state':state,'samples':samples,'count_sources':count_sources,'dataset_root_hash':ds.root_hash,
              'protected_unchanged':True, 'spill_files_at_end':spill_files,'process_preflight':process_check,
              'release_readiness_anchors':release_anchors(ROOT/'output/phase4/inputs/release-readiness'),
              'extrapolation':extrapolate(samples,int(ds.hours.n_trades.sum()),full_counts,
                  full_days=max(0,int(ds.hours.groupby(ds.hours.hour_ms//86400000).n_trades.sum().gt(0).sum())-1)*len(full_counts))}
    (cli.out/'benchmark.json').write_bytes(canonical_bytes(result))
    lines = ['# Phase 4 bounded benchmark','', 'No studies or trials were created. Full-history approval remains required.','',
             'Inputs and formulas are recorded in benchmark.json.','',
             '```json',json.dumps(result['extrapolation'],indent=2),'```']
    (cli.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def tfc_sha(path):
    from orderflow_system.data.paired_dataset import sha256_file
    return sha256_file(path)


def release_anchors(directory):
    """Read supplied evidence; never execute its experimental detector variants."""
    import numpy as np
    directory=Path(directory)
    required=[directory/'p4_daydist.json',directory/'stored-bar-only-durations.json']
    experiments=sorted(directory.glob('p4exp-*.json'))
    if not all(p.is_file() for p in required) or not experiments:
        return {'status':'unavailable','directory':str(directory)}
    distribution=json.loads(required[0].read_text())
    samples=[json.loads(p.read_text()) for p in experiments]
    design=np.array([[s['bars'],s['bars']*s['levels_per_bar']['mean']] for s in samples])
    fits={}
    for label,key in [('stock','absorption_levels'),('approved_reader_no_round','variant_noround')]:
        cost=np.array([(s['stock'][key] if label=='stock' else s[key])['seconds_min']-s['stock']['off']['seconds_min'] for s in samples])
        coeff,_,rank,_=np.linalg.lstsq(design,cost,rcond=None)
        fits[label]={'incremental_seconds_per_bar':float(coeff[0]),'incremental_seconds_per_level':float(coeff[1]),
                     'rank':int(rank),'residuals_seconds':(cost-design@coeff).tolist(),
                     'search80_increment_seconds':{bt:float(coeff[0]*.8*row['bars']+coeff[1]*row['levels_proxy_search80'])
                                                   for bt,row in distribution['per_bar_type'].items()}}
    return {'status':'read_only_evidence','phase3_full_replay_seconds':8063,
        'stored_bar_only_trial_durations':json.loads(required[1].read_text()),
        'day_distribution':distribution,'per_level_a2_model':fits,
        'formula':'A2 increment over bar-only = fitted seconds_per_bar * bars + fitted seconds_per_level * levels',
        'inputs':[{'bars':s['bars'],'levels':float(row[1]),'stock_off_seconds':s['stock']['off']['seconds_min'],
                   'stock_a2_seconds':s['stock']['absorption_levels']['seconds_min'],'reader_no_round_seconds':s['variant_noround']['seconds_min']}
                  for s,row in zip(samples,design)],
        'source_files':[{'path':str(p),'sha256':tfc_sha(p)} for p in [*required,*experiments]],
        'limitation':'fixed defaults on four bounded lanes; day-distribution levels are proxies; historical durations include other overhead. Diagnostic anchors only; speed gates unchanged. Detector experiment not used.'}


def pd_day(day):
    return (day-date(1970,1,1)).days


if __name__=='__main__':
    main()

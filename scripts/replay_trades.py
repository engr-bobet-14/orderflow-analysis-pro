"""Authenticated shared-pass replay and pre-drawn window equivalence workload.

Run under scripts/measure_rss.py for per-process peaks. Reports and DuckDB spill
stay local; datasets, certificate and source files are opened read-only.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from datetime import date, datetime, timedelta, timezone
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

# Support python -I scripts/replay_trades.py from this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
try:
    import psutil
except ImportError:
    psutil = None

from orderflow_system.config.settings import (TradeFlowConfig, get_btcusd_config, parse_trade_signals,
                                             validate_trade_signal_range)
from orderflow_system.data.infobar_feed import DataIntegrityError, open_bars
from orderflow_system.data.paired_dataset import open_registered, sha256_file
from orderflow_system.data.trade_stream import TradeReplay
from orderflow_system.data.trade_alignment import peak_rss_bytes
from orderflow_system.pipeline import InfoBarPipeline

MIB = 1 << 20
PEAK_RSS_LIMIT_BYTES = 4 * 1024**3


def _rss_slope(samples):
    """OLS bytes per elapsed UTC day; integer sums avoid cancellation of large RSS values."""
    x = [s['day']-samples[0]['day'] for s in samples]
    y = [s['bytes'] for s in samples]
    n, sx = len(x), sum(x)
    denominator = n*sum(v*v for v in x)-sx*sx
    return (n*sum(a*b for a, b in zip(x, y))-sx*sum(y))/denominator if denominator else None


def _stability_after_day(samples, rss_kind, anchor_day):
    stability = {'status': 'not_evaluable', 'n_post_anchor': 0, 'slope_bytes_per_day': None,
                 'limit_bytes_per_day': MIB, 'rolling_30_day_slopes': [], 'sample_scope': 'main_replay'}
    if not samples or any(s.get('bytes') is None or s.get('day') is None for s in samples):
        stability['reason'] = 'missing RSS day samples'
    elif anchor_day is None:
        stability['reason'] = 'anchor day unavailable'
    else:
        post = [s for s in samples if s['day'] > anchor_day]
        stability['n_post_anchor'] = len(post)
        if rss_kind != 'current' or any(s.get('rss_kind', rss_kind) != 'current' for s in samples):
            stability['reason'] = 'current RSS samples required; peak samples cannot measure stability'
        elif len(post) < 30:
            stability['reason'] = 'at least 30 samples after the anchor day required'
        else:
            slope = _rss_slope(post)
            stability['slope_bytes_per_day'] = slope
            if slope is None:
                stability['reason'] = 'post-anchor samples have no elapsed-day span'
            else:
                stability['status'] = 'pass' if slope <= MIB else 'fail'
                if stability['status'] == 'fail':
                    stability['reason'] = 'post-anchor OLS slope exceeds 1 MiB/day'
                stability['rolling_30_day_slopes'] = [
                    {'start_day': post[i]['day'], 'end_day': post[i+29]['day'],
                     'slope_bytes_per_day': _rss_slope(post[i:i+30])}
                    for i in range(len(post)-29)
                ]
    return stability


def evaluate_memory(samples, peak_bytes, rss_kind, *, day_loads=None, extra_samples=()):
    """§31: main replay trend; whole-process peak and quiescent memory including window checks."""
    extra_samples = list(extra_samples)
    all_samples = list(samples)+extra_samples
    peak = {'status': 'not_evaluable' if peak_bytes is None else ('pass' if peak_bytes <= PEAK_RSS_LIMIT_BYTES else 'fail'),
            'peak_rss_bytes': peak_bytes, 'limit_bytes': PEAK_RSS_LIMIT_BYTES,
            'source': 'resource.RUSAGE_SELF; host wait4 remains authoritative'}
    if peak_bytes is None:
        peak['reason'] = 'process peak RSS unavailable'
    elif peak['status'] == 'fail':
        peak['reason'] = f'process peak RSS exceeds {PEAK_RSS_LIMIT_BYTES // 1024**3} GiB'

    values = [s.get('duckdb_memory_bytes') for s in all_samples]
    maximum = max(values) if values and all(v is not None for v in values) else None
    duckdb = {'status': 'not_evaluable' if maximum is None else ('pass' if maximum <= 64*MIB else 'fail'),
              'max_bytes': maximum, 'limit_bytes': 64*MIB, 'n_samples': len(values),
              'sample_scope': 'main_replay_and_windows'}
    if maximum is None:
        duckdb['reason'] = 'missing quiescent DuckDB day samples'
    elif duckdb['status'] == 'fail':
        duckdb['reason'] = 'quiescent DuckDB memory exceeds 64 MiB'

    valid_samples = bool(samples) and all(s.get('bytes') is not None and s.get('day') is not None for s in samples)
    anchor = None
    if valid_samples and day_loads is not None and all(day_loads.get(s['day']) is not None for s in samples):
        anchor = min((s['day'] for s in samples), key=lambda day: (-day_loads[day], day))
    stability = _stability_after_day(samples, rss_kind, anchor)
    stability.update({'anchor_day': datetime.fromtimestamp(int(anchor)*86400, timezone.utc).date().isoformat()
                      if anchor is not None else None,
                      'anchor_n_trades': int(day_loads[anchor]) if anchor is not None else None,
                      'anchor_source': 'hours.parquet n_trades'})
    if valid_samples and anchor is None:
        stability['reason'] = 'missing authenticated daily trade counts'

    # Preserve the former RSS-maximum anchor for comparison only; it cannot affect the verdict.
    peak_day = max(samples, key=lambda s: s['bytes'])['day'] if valid_samples else None
    rss_max_anchor = _stability_after_day(samples, rss_kind, peak_day)
    rss_max_anchor['peak_day'] = peak_day
    rss_max_anchor['n_post_peak'] = rss_max_anchor.pop('n_post_anchor')

    first = samples[0].get('malloc_in_use_bytes') if samples else None
    last = samples[-1].get('malloc_in_use_bytes') if samples else None
    window_malloc = [s['malloc_in_use_bytes'] for s in extra_samples if s.get('malloc_in_use_bytes') is not None]
    heap = {'first_bytes': first, 'last_bytes': last, 'flagged': None,
            'windows_max_malloc_in_use_bytes': max(window_malloc, default=None)}
    if first is not None and last is not None:
        heap['flagged'] = last*10 > first*11+50*MIB*10
    else:
        heap['reason'] = 'malloc in-use endpoints unavailable'
    return {'gate_peak_rss': peak, 'gate_duckdb': duckdb, 'gate_stability': stability,
            'diagnostic_live_heap': heap, 'rss_max_anchor': rss_max_anchor,
            'verdict': 'pass' if all(g['status'] == 'pass' for g in (peak, duckdb, stability)) else 'fail'}


def _certificate(path, ds, bar_types):
    if path is None:
        return None
    cert = json.loads(Path(path).read_text())
    if cert.get('v1') != {'dataset_id': ds.dataset_id, 'root_hash': ds.root_hash}:
        raise DataIntegrityError('certificate dataset id/root differs from authenticated dataset')
    if cert.get('certification') != 'dataset-pipeline' or not cert.get('pipeline_equivalent_default'):
        raise DataIntegrityError('certificate does not certify default pipeline equivalence')
    expected = {}
    for bt in bar_types:
        l4 = cert.get('types', {}).get(bt, {}).get('L4', {})
        rows = [r for r in l4.get('configs', []) if r.get('config') == 'default' and r.get('params') is None
                and r.get('window') == [None, None] and r.get('equal')]
        if not l4.get('passed') or len(rows) != 1:
            raise DataIntegrityError(f'certificate missing default full-range digest for {bt}')
        expected[bt] = rows[0]
    return expected


def _draw_windows(ds, count, seed):
    """Draw before the full run: >= five midnight starts and the heaviest day if in range."""
    rng = np.random.default_rng(seed)
    first, last = map(date.fromisoformat, ds.manifest['requested_range'])
    days = (last-first).days+1
    result = []
    for i in range(count):
        if i == 0 and first <= date(2021, 5, 19) <= last:
            start = datetime(2021, 5, 19, tzinfo=timezone.utc)
        else:
            start = datetime.combine(first+timedelta(days=int(rng.integers(days))), datetime.min.time(), timezone.utc)
            if i >= 5:
                start += timedelta(hours=int(rng.integers(12)))
        end = min(start+timedelta(days=1), datetime.combine(last+timedelta(days=1), datetime.min.time(), timezone.utc))
        result.append((start.isoformat(), end.isoformat()))
    return result


def _snapshot(accumulator):
    return [{'day': day, 'ticks': ticks.tolist(), 'lots': lots.tolist()}
            for day,ticks,lots in accumulator.developing_profiles()]


def run(dataset_id, *, config=None, bar_types=None, start=None, end=None, windows=0, seed=20261103,
        certificate=None, batch_rows=1_000_000):
    if certificate is not None and (start is not None or end is not None):
        raise ValueError('--certificate requires a full-range replay without --start/--end')
    begun = time.perf_counter()
    config = config or TradeFlowConfig()
    if certificate is not None and config.signals.any_enabled:
        raise ValueError('--certificate cannot be combined with enabled --trade-signals')
    validate_trade_signal_range(config.signals, start, end)
    # Authentication and certificate identity precede all reader iteration.
    ds = open_registered(Path(config.paired_root), dataset_id)
    validate_trade_signal_range(config.signals, start, end, ds.manifest['requested_range'])
    types = list(bar_types or ds.bar_types)
    if not types or len(set(types)) != len(types) or any(bt not in ds.bar_types for bt in types):
        raise ValueError('bar types must be unique members of the paired dataset')
    if ds.manifest['symbol'] != 'BTCUSDT':
        raise ValueError('only BTCUSDT is configured')
    if windows < 0 or (windows and (start is not None or end is not None)):
        raise ValueError('--windows requires a full-range replay and a nonnegative count')
    certificate_hash = sha256_file(Path(certificate)) if certificate else None
    expected = _certificate(certificate, ds, types)

    def make_replay(lo=None, hi=None, ranges=None, raw_ranges=None, window_signals=False):
        lanes = [(open_bars(ds.bars_root, ds.manifest['symbol'], bt, lo, hi, version=ds.manifest['stamp']),
                  InfoBarPipeline(get_btcusd_config(), trade_config=config)) for bt in types]
        return TradeReplay(ds, lanes, temp_dir=Path(config.duckdb_temp_dir), threads=config.duckdb_threads,
                           memory_limit=config.duckdb_memory_limit, batch_rows=batch_rows, digest_ranges=ranges,
                           raw_signal_ranges=raw_ranges, window_signals=window_signals)

    drawn = _draw_windows(ds, windows, seed)
    bounds, ranges = [], {bt: [] for bt in types}
    raw_ranges = {bt: {'absorption': [], 'exhaustion': []} for bt in types}
    signal_bounds = []
    # Retain only bounds, not window readers/pipelines and their Arrow buffers.
    for lo, hi in drawn:
        probe = make_replay(lo, hi)
        bound = {bt: (lane.first_seq, lane.last_tn) for bt, lane in probe.lanes.items()}
        bounds.append(bound)
        e_bounds = {}
        for bt, lane in probe.lanes.items():
            # The probe already holds the first bar; exclude n+1 bars by count,
            # never by adding to a closing-trade ordinal.
            first_after_prefix = None
            for _ in range(lane.pipeline.config.exhaustion.min_bars_declining + 1):
                first_after_prefix = next(lane.iterator, None)
                if first_after_prefix is None:
                    break
            first = first_after_prefix.seq if first_after_prefix is not None else lane.last_tn + 1
            e_bounds[bt] = (first, lane.last_tn)
            raw_ranges[bt]['absorption'].append(bound[bt])
            raw_ranges[bt]['exhaustion'].append(e_bounds[bt])
        signal_bounds.append(e_bounds)
        for bt, pair in bound.items():
            ranges[bt].append(pair)
        del probe
    replay = make_replay(start, end, ranges, raw_ranges)
    snapshots = {}
    ends = {bt: {pair[1] for pair in pairs} for bt,pairs in ranges.items()}
    for bt,result in replay:
        if result.bar.seq in ends[bt]:
            snapshots[(bt, result.bar.seq)] = _snapshot(replay.lanes[bt].accumulator)
    report = replay.report
    report['replay_wall_seconds'] = time.perf_counter()-begun
    if expected and replay.full:
        for bt, row in expected.items():
            lane = report['lanes'][bt]
            if lane['replay_digest'] != row['v1'] or lane['bars'] != row['n_bars']:
                raise DataIntegrityError(f'{bt}: legacy ReplayDigest differs from certificate')
        report['certificate_match'] = True
    else:
        report['certificate_match'] = ('not_applicable: trade signals enabled' if config.signals.any_enabled else None)
    report['windows'] = []
    for (lo, hi), bound, e_bound in zip(drawn, bounds, signal_bounds):
        window_begun = time.perf_counter()
        window = make_replay(lo, hi, window_signals=True)
        for _ in window:
            pass
        checks = {}
        for bt, (first, last) in bound.items():
            lane = window.report['lanes'][bt]
            histogram = [{k:p[k] for k in ('day','ticks','lots')} for p in lane['developing_profiles']]
            checks[bt] = {'digest_equal': lane['trade_digest'] == replay.lanes[bt].trade_digest.range_hexdigest(first,last),
                          'developing_equal': histogram == snapshots[(bt,last)], 'identities': lane['identities'],
                          'trade_digest': lane['trade_digest'], 'first_seq':first, 'last_seq':last}
            signal_checks = {}
            for name, enabled, pair in (
                ('absorption', config.signals.absorption_levels, (first, last)),
                ('exhaustion', config.signals.exhaustion_contrarian != 'off', e_bound[bt]),
            ):
                if enabled:
                    signal_checks[name] = (lane['raw_signal_digests'][name] ==
                                           replay.lanes[bt].raw_digests[name].range_hexdigest(*pair))
            checks[bt]['raw_signal_equal'] = signal_checks
            checks[bt]['exhaustion_excluded_prefix_bars'] = get_btcusd_config().exhaustion.min_bars_declining + 1
        equal = all(all(c['raw_signal_equal'].values()) and c['digest_equal'] and c['developing_equal'] and all(c['identities'].values()) for c in checks.values())
        if not equal:
            raise DataIntegrityError(f'window {lo}..{hi}: trade range/state differs from full replay')
        report['windows'].append({'start':lo,'end':hi,'equal':equal,'lanes':checks,
                                  'wall_seconds':time.perf_counter()-window_begun, 'stream':window.report['stream']})
        del window
        gc.collect()
    ds.recheck()
    if certificate and sha256_file(Path(certificate)) != certificate_hash:
        raise DataIntegrityError('certificate changed during replay')
    report.update({'trade_signals': asdict(config.signals), 'status':'passed', 'wall_seconds':time.perf_counter()-begun,
                   'requested_window':[start,end], 'seed':seed, 'predrawn_windows':drawn,
                   'authentication_limit':'authenticated against our frozen archive, not Binance originals',
                   'certificate_sha256':certificate_hash,
                   'rss_kind':report['stream']['rss_kind'],
                   'versions':{**{name:version(name) for name in ('numpy','pyarrow','duckdb')},
                               'psutil':version('psutil') if psutil else None},
                   'python':sys.version, 'hardware':{'machine':platform.machine(), 'platform':platform.platform(),
                       'cores':psutil.cpu_count() if psutil else None,
                       'ram_bytes':psutil.virtual_memory().total if psutil else None},
                   'unavailable':[] if psutil else ['versions.psutil', 'hardware.cores', 'hardware.ram_bytes'],
                   'spill_directory':str(Path(config.duckdb_temp_dir).resolve()), 'spill_cleaned':True})
    report['memory_criteria'] = evaluate_memory(
        report['stream']['per_day_rss'], peak_rss_bytes(), report['rss_kind'],
        day_loads=ds.hours.groupby(ds.hours.hour_ms//86_400_000).n_trades.sum().to_dict(),
        extra_samples=[sample for window in report['windows'] for sample in window['stream']['per_day_rss']],
    )
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paired-dataset', required=True)
    p.add_argument('--paired-root', default=TradeFlowConfig().paired_root)
    p.add_argument('--bar-types', nargs='+')
    p.add_argument('--start')
    p.add_argument('--end')
    p.add_argument('--windows', type=int, default=0)
    p.add_argument('--seed', type=int, default=20261103)
    p.add_argument('--certificate', type=Path)
    p.add_argument('--trade-signals', default='', help='Comma-separated absorption_levels, exhaustion_bonus|exhaustion_required, framing_trades')
    p.add_argument('--enforce-memory-gates', action='store_true',
                   help='Exit nonzero after writing the report unless every memory gate passes')
    args = p.parse_args(argv)
    try:
        signals = parse_trade_signals(args.trade_signals)
        validate_trade_signal_range(signals, args.start, args.end)
        if args.certificate and signals.any_enabled:
            p.error('--certificate cannot be combined with enabled --trade-signals')
    except ValueError as exc:
        p.error(str(exc))
    report = run(args.paired_dataset, config=TradeFlowConfig(paired_root=args.paired_root, signals=signals),
                 bar_types=args.bar_types, start=args.start, end=args.end, windows=args.windows,
                 seed=args.seed, certificate=args.certificate)
    output = Path(__file__).resolve().parents[1]/'output'/'trade_replay'
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    label = f'{args.start or "begin"}-{args.end or "end"}'.replace(':','')
    path = output/f'{args.paired_dataset}-{label}-{stamp}.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status':report['status'], 'report':str(path), 'wall_seconds':report['wall_seconds']}))
    if args.enforce_memory_gates and report['memory_criteria']['verdict'] != 'pass':
        raise SystemExit(1)
    return report


if __name__ == '__main__':
    main()

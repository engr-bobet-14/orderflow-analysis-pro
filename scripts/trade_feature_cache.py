"""Authenticated, memory-mapped TradeBarUpdate cache. Build ranges use exclusive ends."""
from __future__ import annotations

import argparse
from dataclasses import astuple
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from orderflow_system.analytics.trade_accumulator import TradeBarStats, TradeBarUpdate
from orderflow_system.config.settings import TradeFlowConfig, get_btcusd_config, parse_trade_signals
from orderflow_system.data.infobar_feed import open_bars
from orderflow_system.data.models import BarType,FootprintLevel
from orderflow_system.analytics.footprint import FootprintBar
from orderflow_system.data.paired_dataset import open_registered, pinned_root, sha256_file
from orderflow_system.data.trade_stream import TradeReplay
from orderflow_system.data.trade_alignment import peak_rss_bytes
from orderflow_system.pipeline import InfoBarPipeline, TradeDigest, TradeView
from scripts.phase4_policy import checked_write, source_state, verify_source, canonical_bytes, production_preflight, protected_state

BUILDER_FILES = ('scripts/trade_feature_cache.py', 'orderflow_system/analytics/trade_accumulator.py',
                 'orderflow_system/data/trade_stream.py', 'orderflow_system/data/trade_alignment.py')
ARRAYS = ('seq', 'timestamp_ms', 'stats', 'fp_offsets', 'fp_0', 'fp_1', 'fp_2',
          'day_bar_index', 'day_id', 'day_offsets', 'day_0', 'day_1')
MAX_BYTES = 40 * 1024**3
PRODUCTION_LANES = frozenset(b.value for b in BarType)


class _CachedFootprint(FootprintBar):
    """Materialise Python level objects only if a detector actually reads them."""
    def __init__(self,bar,source):
        super().__init__(timestamp_ms=bar.timestamp_ms,open=bar.open,high=bar.high,low=bar.low,close=bar.close,levels=None)
        self._source=source

    @property
    def levels(self):
        if self._levels is None:
            levels={}
            # Conversion to Python integers in C avoids per-level NumPy scalar boxing.
            for p,b,s in zip(*(array.tolist() for array in self._source)):
                price=int(p)/100
                levels[price]=FootprintLevel(price,int(s)/1000,int(b)/1000)
            self._levels=levels
        return self._levels

    @levels.setter
    def levels(self,value):
        self._levels=value


class CachedFootprintLevels(tuple):
    """The original three read-only array views, with a deferred footprint factory."""
    def to_footprint(self,bar,engine):
        result=_CachedFootprint(bar,self)
        engine._remember(result)
        return result


def builder_fingerprint():
    h = hashlib.sha256()
    for name in BUILDER_FILES:
        h.update(name.encode())
        h.update((ROOT/name).read_bytes())
    return h.hexdigest()


def validate_build_range(start, end, full_history=False, owner_approved=None, command_argv=None):
    if full_history:
        if not owner_approved or start is not None or end is not None:
            raise ValueError('full-history requires --owner-approved host approval and no explicit range')
        from scripts.phase4_approvals import verify_approval
        verify_approval('full_history_tfc',owner_approved,argv=command_argv)
        return 'production'
    if start is None or end is None:
        raise ValueError('bounded build requires start and exclusive end (at most 4 days)')
    lo, hi = date.fromisoformat(start), date.fromisoformat(end)
    if lo < date(2019, 9, 10) or hi > date(2024, 1, 1) or hi <= lo:
        raise ValueError('cache range must be within the development window')
    if (hi-lo).days > 4:
        raise ValueError('build exceeds 4 days; full-history requires owner approval')
    return 'bounded_verification'


def cache_destination(path,mode):
    bounded=ROOT/'output/phase4/bounded_cache'
    production=ROOT/'output/optimize/trade_cache'
    target=checked_write(path if path is not None else production if mode=='production' else bounded)
    if mode=='bounded_verification' and target.is_relative_to(production):
        raise ValueError('bounded verification builds must use the separate bounded cache root')
    if mode=='production' and target.is_relative_to(bounded):
        raise ValueError('production builds cannot use the bounded cache root')
    return target


def require_source_mode(mode):
    state = source_state()
    if mode == 'production' and state['dirty']:
        raise ValueError('production build requires a clean tree')
    return state


def production_generation(destination):
    """Open only marker-listed manifests. Orphans are reported by name, never read."""
    destination=Path(destination).resolve()
    try:
        marker=json.loads((destination/'COMPLETE.json').read_bytes())
        entries=marker['manifests']
        lanes={};identities=set()
        if not isinstance(entries,dict):raise ValueError('invalid COMPLETE.json manifests')
        for name,digest in entries.items():
            parts=Path(name).parts
            path=destination/name
            if (len(parts)!=2 or parts[1]!='manifest.json' or '..' in parts or path.parent.is_symlink()
                    or path.is_symlink() or not path.resolve().is_relative_to(destination)):
                raise ValueError('invalid COMPLETE.json manifest path')
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=digest:
                raise ValueError('production COMPLETE.json manifest mismatch')
            meta=json.loads(raw);bt=meta['bar_type']
            if meta['mode']!='production' or bt in lanes or path.parent.name!=bt+'-'+digest:
                raise ValueError('invalid production COMPLETE.json lane')
            lanes[bt]=str(path.parent)
            identities.add((meta['dataset_id'],meta['root_hash']))
        if set(lanes)!=PRODUCTION_LANES or len(identities)!=1:
            raise ValueError('COMPLETE.json requires the complete expected six-lane set for one dataset')
    except (OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
        raise ValueError('production cache requires a valid COMPLETE.json marker') from exc
    listed=set(lanes.values())
    orphans=sorted(str(p) for p in destination.iterdir() if p.is_dir() and str(p) not in listed
                   and any(p.name.startswith(bt+'-') for bt in PRODUCTION_LANES))
    dataset,root=next(iter(identities))
    return {'lanes':lanes,'orphans':orphans,'dataset_id':dataset,'root_hash':root}


def generation_directory(path,dataset_id):
    path=Path(path).resolve()
    if (path/'COMPLETE.json').exists():return path
    if (path/dataset_id/'COMPLETE.json').exists():return path/dataset_id
    if (path.parent/'COMPLETE.json').exists():return path.parent
    matches=[p.parent for p in path.rglob('COMPLETE.json')
             if production_generation(p.parent)['dataset_id']==dataset_id]
    if len(matches)==1:return matches[0]
    raise ValueError('production cache requires a complete generation with COMPLETE.json')


def require_complete(directory):
    directory=Path(directory).resolve()
    generation=production_generation(directory.parent)
    if str(directory) not in generation['lanes'].values():
        raise ValueError('production cache lane is an unlisted orphan in COMPLETE.json')
    return generation


class _Writer:
    """Append raw arrays on disk, then convert in bounded chunks to .npy files."""
    def __init__(self, directory):
        self.path = directory
        directory.mkdir()
        self.handles, self.dtypes, self.counts = {}, {}, {}
        self.n = self.n_fp = self.n_days = self.n_day_levels = self.bytes = 0
        self.bar_handle = (directory/'bars.bin').open('wb')
        self.append('fp_offsets', [0])
        self.append('day_offsets', [0])

    def append(self, name, values):
        a = np.asarray(values)
        if name not in self.handles:
            self.dtypes[name] = a.dtype
            self.handles[name] = (self.path/(name+'.raw')).open('wb')
            self.counts[name] = 0
        if a.dtype != self.dtypes[name]:
            raise ValueError(f'{name}: dtype changed during build')
        self.handles[name].write(a.tobytes())
        self.counts[name] += a.size
        self.bytes += a.nbytes

    def capture(self, bar, update):
        from scripts.optimize_orderflow import BAR_DTYPE, bar_tuple
        self.bar_handle.write(np.array([bar_tuple(bar)], dtype=BAR_DTYPE).tobytes())
        self.bytes += BAR_DTYPE.itemsize
        self.append('seq', np.array([bar.seq], dtype='<i8'))
        self.append('timestamp_ms', np.array([bar.timestamp_ms], dtype='<i8'))
        self.append('stats', np.asarray(astuple(update.stats), dtype='<i8'))
        for i, a in enumerate(update.footprint_levels):
            self.append(f'fp_{i}', a)
        self.n_fp += len(update.footprint_levels[0])
        self.append('fp_offsets', np.array([self.n_fp], dtype='<i8'))
        for day, ticks, lots in update.finalised_days:
            self.append('day_bar_index', np.array([self.n], dtype='<i8'))
            self.append('day_id', np.array([day], dtype='<i8'))
            self.append('day_0', ticks)
            self.append('day_1', lots)
            self.n_day_levels += len(ticks)
            self.append('day_offsets', np.array([self.n_day_levels], dtype='<i8'))
            self.n_days += 1
        self.n += 1

    def close(self):
        self.bar_handle.close()
        for handle in self.handles.values():
            handle.close()

    def finish(self):
        for name in ARRAYS:
            if name not in self.handles:
                self.append(name, np.empty(0, dtype='<i8'))
        self.close()
        for name in ARRAYS:
            shape = (self.n, 13) if name == 'stats' else (self.counts[name],)
            target = np.lib.format.open_memmap(self.path/(name+'.npy'), mode='w+', dtype=self.dtypes[name], shape=shape)
            if self.counts[name]:
                source = np.memmap(self.path/(name+'.raw'), mode='r', dtype=self.dtypes[name])
                flat = target.reshape(-1)
                for start in range(0, source.size, 1_000_000):
                    flat[start:start+1_000_000] = source[start:start+1_000_000]
                del source, flat
            target.flush()
            del target
            (self.path/(name+'.raw')).unlink()


def build(ds, destination, *, start=None, end=None, full_history=False, owner_approved=None,
          query_observer=None, trade_config=None, command_argv=None):
    mode = validate_build_range(start, end, full_history, owner_approved,command_argv)
    dest = cache_destination(destination,mode)
    state = require_source_mode(mode)
    if mode=='production' and (dest/'COMPLETE.json').exists():
        raise ValueError('destination already holds a COMPLETE generation; owner must move/delete it explicitly')
    if mode=='production' and set(ds.bar_types)!=PRODUCTION_LANES:
        raise ValueError('production build requires the complete expected six-lane set')
    if date.fromisoformat(ds.manifest['requested_range'][1]) > date(2023, 12, 31):
        raise ValueError('development-only paired dataset required')
    cfg = trade_config or TradeFlowConfig()
    checked_write(cfg.duckdb_temp_dir)
    if cfg.signals.any_enabled:
        raise ValueError('cache builder requires flags off')
    from scripts.phase4_resources import current_memory,enforce_current
    if mode=='production':
        from scripts.phase4_approvals import verify_approval,consume_approval
        approval=verify_approval('full_history_tfc',owner_approved,argv=command_argv)
        preflight=production_preflight(ds,dest,state,MAX_BYTES)
        consume_approval(approval)
    else:
        preflight={'mode':'bounded_verification'}
    begun = time.perf_counter()
    dest.mkdir(parents=True, exist_ok=True)
    paths = {}
    writers = {}
    with tempfile.TemporaryDirectory(dir=dest, prefix='.staging-') as tmp:
        staging = Path(tmp)
        try:
            for bt in ds.bar_types:
                writers[bt] = _Writer(staging/bt)
            def capture(bt, bar, update, result):
                writers[bt].capture(bar, update)
                if sum(w.bytes for w in writers.values()) > MAX_BYTES:
                    raise ValueError('TFC exceeds 40 GiB budget')
                if time.perf_counter()-begun > 4*3600:
                    raise ValueError('TFC build exceeds 4 hour budget')
                if sum(w.n for w in writers.values()) % 16384 == 0 and peak_rss_bytes() > 4*1024**3:
                    raise ValueError('TFC build exceeds 4 GiB RSS budget')
            lanes = [(open_bars(ds.bars_root, ds.manifest['symbol'], bt, start, end, version=ds.manifest['stamp']),
                      InfoBarPipeline(get_btcusd_config(), BarType(bt), trade_config=cfg)) for bt in ds.bar_types]
            replay = TradeReplay(ds, lanes, temp_dir=Path(cfg.duckdb_temp_dir), development_only=True,
                                 query_observer=query_observer, update_sink=capture,
                                 threads=cfg.duckdb_threads, memory_limit=cfg.duckdb_memory_limit)
            for _ in replay:
                pass
            for bt, writer in writers.items():
                writer.finish()
                lane = replay.report['lanes'][bt]
                files = {p.name: sha256_file(p) for p in sorted(writer.path.iterdir())}
                manifest = {'format': 'trade-feature-cache/1', 'dataset_id': ds.dataset_id, 'root_hash': ds.root_hash,
                    'bar_type': bt, 'n_bars': writer.n, 'first_seq': lane['first_seq'], 'last_seq': lane['last_seq'],
                    'fp_bucket_ticks': round(cfg.footprint_bucket*100), 'pf_bucket_ticks': round(cfg.profile_bucket*100),
                    'trade_digest': lane['trade_digest'], 'files': files, 'builder_fp': builder_fingerprint(),
                    'git_sha': state['base'], 'dirty': state['dirty'], 'source_state': state, 'mode': mode,
                    'owner_approval_sha256':owner_approved if mode=='production' else None,
                    'start': start, 'end': end, 'created_utc': datetime.now(timezone.utc).isoformat()}
                (writer.path/'manifest.json').write_bytes(canonical_bytes(manifest))
                # Read back all bytes and reconstruct the trade digest before publishing.
                CachedTradeUpdates(writer.path, _staged=True)
                key = hashlib.sha256((writer.path/'manifest.json').read_bytes()).hexdigest()
                paths[bt] = dest/(bt+'-'+key)
            if sum(p.stat().st_size for p in staging.rglob('*') if p.is_file()) > MAX_BYTES:
                raise ValueError('TFC exceeds 40 GiB budget')
            if peak_rss_bytes() > 4*1024**3:
                raise ValueError('TFC build exceeds 4 GiB RSS budget')
            enforce_current()
            verify_source(state)
            if mode=='production' and protected_state()!=preflight['protected_state']:
                raise ValueError('protected set changed during production build')
            for bt, writer in writers.items():
                writer.path.rename(paths[bt])
            if mode=='production':
                verify_source(state)
                if protected_state()!=preflight['protected_state']:
                    raise ValueError('protected set changed during production publish')
                enforce_current()
                if time.perf_counter()-begun>4*3600:raise ValueError('TFC build exceeds 4 hour budget')
                from scripts.optimize_orderflow import atomic_json
                atomic_json(dest/'COMPLETE.json',{'manifests':{
                    path.name+'/manifest.json':sha256_file(path/'manifest.json') for path in paths.values()}})
        finally:
            for writer in writers.values():
                writer.close()
    report = {**replay.report, 'mode': mode, 'source_state': state, 'preflight':preflight,
              'orphans':production_generation(dest)['orphans'] if mode=='production' else [],'wall_seconds': time.perf_counter()-begun,
              'bytes': sum(sum(p.stat().st_size for p in directory.iterdir()) for directory in paths.values()),
              'cache_paths': {bt: str(p) for bt, p in paths.items()},'memory':current_memory()}
    return paths, report


class CachedTradeUpdates:
    def __init__(self, directory, bars=None, *, production=False, dataset_id=None, bar_type=None,
                 fp_bucket_ticks=100, pf_bucket_ticks=1000, _staged=False):
        opened=time.perf_counter()
        self.path = Path(directory)
        if production and not _staged:
            require_complete(self.path)
        raw = (self.path/'manifest.json').read_bytes()
        self.manifest = m = json.loads(raw)
        manifest_hash = hashlib.sha256(raw).hexdigest()
        if not _staged and self.path.name != m['bar_type']+'-'+manifest_hash:
            raise ValueError('cache manifest hash differs from content-addressed directory')
        self.identity = {'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'builder_fp': m['builder_fp'],
                         'dataset_id': m['dataset_id'], 'root_hash': m['root_hash']}
        if m.get('format') != 'trade-feature-cache/1' or m['builder_fp'] != builder_fingerprint():
            raise ValueError('cache builder fingerprint/format mismatch')
        if m['root_hash'] != pinned_root(m['dataset_id']):
            raise ValueError('cache dataset root mismatch')
        if production and m['mode'] != 'production':
            raise ValueError('optimizer requires a production cache')
        if m['mode']=='production' and not _staged:
            require_complete(self.path)
        if production and (m['dirty'] or m['source_state']['dirty'] or m['start'] is not None or m['end'] is not None):
            raise ValueError('production cache must bind clean full-history source')
        if ((dataset_id is not None and dataset_id != m['dataset_id']) or
                (bar_type is not None and bar_type != m['bar_type']) or
                fp_bucket_ticks != m['fp_bucket_ticks'] or pf_bucket_ticks != m['pf_bucket_ticks']):
            raise ValueError('cache dataset/bar type/bucket mismatch')
        if set(m['files']) != {n+'.npy' for n in ARRAYS} | {'bars.bin'}:
            raise ValueError('cache array manifest mismatch')
        hash_started=time.perf_counter()
        for name, digest in m['files'].items():
            if sha256_file(self.path/name) != digest:
                raise ValueError(f'cache file hash mismatch: {name}')
        hash_seconds=time.perf_counter()-hash_started
        # ndarray views retain the mmap as their base, without per-scalar memmap dispatch.
        self.arrays = {name: np.asarray(np.load(self.path/(name+'.npy'), mmap_mode='r', allow_pickle=False)) for name in ARRAYS}
        self._validate_layout()
        self._days={}
        a=self.arrays
        for j,index in enumerate(a['day_bar_index']):
            lo,hi=map(int,a['day_offsets'][j:j+2])
            self._days.setdefault(int(index),[]).append((int(a['day_id'][j]),a['day_0'][lo:hi],a['day_1'][lo:hi]))
        if bars is not None:
            self.validate_bars(bars)
        digest_started=time.perf_counter()
        if self.trade_digest() != m['trade_digest']:
            raise ValueError('cache trade digest mismatch')
        digest_seconds=time.perf_counter()-digest_started
        total=time.perf_counter()-opened
        self.open_timings={'hash_seconds':hash_seconds,'digest_seconds':digest_seconds,'total_seconds':total,
            'other_seconds':max(0.,total-hash_seconds-digest_seconds),'bars':m['n_bars'],
            'hash_bytes':sum((self.path/name).stat().st_size for name in m['files'])}

    def _validate_layout(self):
        a, n = self.arrays, self.manifest['n_bars']
        if any(array.ndim!=1 for key,array in a.items() if key!='stats'):
            raise ValueError('cache arrays must be one-dimensional')
        if a['stats'].shape != (n, 13) or not a['stats'].flags.c_contiguous or any(len(a[k]) != n for k in ('seq', 'timestamp_ms')):
            raise ValueError('cache stats/seq shape mismatch')
        if n < 1 or a['seq'][0] != self.manifest['first_seq'] or a['seq'][-1] != self.manifest['last_seq']:
            raise ValueError('cache first/last sequence mismatch')
        if (np.diff(a['seq']) <= 0).any() or (np.diff(a['timestamp_ms']) < 0).any():
            raise ValueError('cache sequence/time ordering mismatch')
        for name in ('seq', 'timestamp_ms', 'stats', 'fp_offsets', 'day_bar_index', 'day_id', 'day_offsets'):
            if a[name].dtype != np.dtype('<i8'):
                raise ValueError('cache integer dtype mismatch')
        for prefix, count, data in [('fp', n, ('fp_0', 'fp_1', 'fp_2')),
                                     ('day', len(a['day_id']), ('day_0', 'day_1'))]:
            offsets = a[prefix+'_offsets']
            if (len(offsets) != count+1 or offsets[0] != 0 or (np.diff(offsets) < 0).any()
                    or any(len(a[k]) != offsets[-1] for k in data)):
                raise ValueError('cache CSR offsets mismatch')
        indices = a['day_bar_index']
        if len(indices) != len(a['day_id']) or (np.diff(indices) < 0).any() or (indices < 0).any() or (indices >= n).any():
            raise ValueError('cache finalised-day index mismatch')
        self.validate_bars(self.bars())

    def bars(self):
        from scripts.optimize_orderflow import CachedBars
        return CachedBars(self.path/'bars.bin', self.manifest['bar_type'])

    def validate_bars(self, bars):
        if (len(bars) != len(self) or bars.bar_type.value != self.manifest['bar_type'] or
                any(not np.array_equal(bars.array[k], self.arrays[k]) for k in ('seq', 'timestamp_ms')) or
                bars.array.dtype != self.bars().array.dtype or not np.array_equal(bars.array,self.bars().array)):
            raise ValueError('cache bar alignment mismatch')

    def __len__(self):
        return self.manifest['n_bars']

    def __getitem__(self, i):
        if not 0 <= i < len(self):
            raise IndexError(i)
        a = self.arrays
        lo, hi = int(a['fp_offsets'][i]),int(a['fp_offsets'][i+1])
        return TradeBarUpdate(TradeBarStats(*struct.unpack_from('<13q',a['stats'],i*104)),
                              CachedFootprintLevels(a[f'fp_{j}'][lo:hi] for j in range(3)), list(self._days.get(i,())),
                              self.manifest['fp_bucket_ticks'])

    def trade_digest(self):
        digest = TradeDigest()
        cfg = TradeFlowConfig(footprint_bucket=self.manifest['fp_bucket_ticks']/100,
                              profile_bucket=self.manifest['pf_bucket_ticks']/100)
        view = TradeView(cfg, get_btcusd_config().volume_profile)
        bars = self.bars()
        for i in range(len(self)):
            bar, update = bars[i], self[i]
            _, stats, profiles = view.update(bar, update)
            digest.update(SimpleNamespace(bar=bar, trade_stats=stats, trade_profiles=profiles), update)
        return digest.hexdigest()


def reference_for(manifest,refs,mode):
    matches=[ref for key,ref in refs.items() if key.endswith('_'+mode)
             and ref.get('start')==manifest['start'] and ref.get('end')==manifest['end']]
    if len(matches)!=1:
        raise ValueError('cache window has no unique matching certified reference')
    return matches[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'verify'))
    parser.add_argument('--dataset-id', default='20261103T011635Z-d3a7b9603b37')
    parser.add_argument('--paired-root', type=Path, default=TradeFlowConfig().paired_root)
    parser.add_argument('--bar-type', default='all', choices=['all', *[b.value for b in BarType]])
    parser.add_argument('--start')
    parser.add_argument('--end', help='exclusive UTC day')
    parser.add_argument('--full-history', action='store_true')
    parser.add_argument('--owner-approved')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--against', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'build':
            validate_build_range(args.start, args.end, args.full_history, args.owner_approved)
            args.out=cache_destination(args.out,'production' if args.full_history else 'bounded_verification')
            ds = open_registered(args.paired_root, args.dataset_id)
            _, report = build(ds, args.out/args.dataset_id, start=args.start, end=args.end,
                              full_history=args.full_history, owner_approved=args.owner_approved)
            target = checked_write(ROOT/'output/phase4/cache-build.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(canonical_bytes(report))
        else:
            generation=None
            if args.full_history:
                validate_build_range(None,None,True,args.owner_approved)
                from scripts.phase4_approvals import verify_approval,consume_approval
            args.out=cache_destination(args.out,'production' if args.full_history else 'bounded_verification')
            if args.full_history:
                approval=verify_approval('full_history_tfc',args.owner_approved)
                ds=open_registered(args.paired_root,args.dataset_id)
                root=generation_directory(args.out,args.dataset_id)
                state=source_state()
                preflight=production_preflight(ds,root,state,MAX_BYTES)
                generation=production_generation(root)
                if generation['dataset_id']!=ds.dataset_id or generation['root_hash']!=ds.root_hash:
                    raise ValueError('production cache differs from the registered paired dataset')
                from scripts.phase4_resources import require_memory_plan,mapped_cache_files,enforce_current,current_memory
                mappings=[mapped_cache_files(p) for p in generation['lanes'].values()]
                preflight['memory']=require_memory_plan([{'mapped_files':max(mappings,key=lambda f:sum(f.values()))}])
                consume_approval(approval)
            from scripts.optimize_orderflow import run_backtest
            refs = json.loads(args.against.read_text()) if args.against else None
            report = {}
            verification_memory={}
            manifests=([Path(p)/'manifest.json' for p in generation['lanes'].values()]
                       if generation else args.out.rglob('manifest.json'))
            mapped_peak=0
            for manifest in manifests:
                metadata=json.loads(manifest.read_bytes())
                if (metadata['mode']!=('production' if args.full_history else 'bounded_verification')
                        or metadata['dataset_id']!=args.dataset_id or args.bar_type not in ('all',metadata['bar_type'])
                        or args.start is not None and args.start!=metadata['start']
                        or args.end is not None and args.end!=metadata['end']):
                    continue
                if generation:
                    files=mapped_cache_files(manifest.parent)
                    require_memory_plan([{'mapped_files':files}])
                    mapped_peak=max(mapped_peak,sum(files.values()))
                cache = CachedTradeUpdates(manifest.parent,production=bool(generation))
                if generation:enforce_current(mapped_peak)
                bt = cache.manifest['bar_type']
                if args.bar_type not in ('all', bt):
                    continue
                if refs:
                    for mode in ('off','all'):
                        reference_for(cache.manifest,refs,mode)
                outputs = {}
                for mode in ('off', 'all'):
                    cfg = TradeFlowConfig(signals=parse_trade_signals('absorption_levels,exhaustion_bonus,framing_trades')
                                          if mode == 'all' else TradeFlowConfig().signals)
                    result = run_backtest(cache.bars(), get_btcusd_config(), .002, BarType(bt),
                                          trade_updates=cache, trade_config=cfg)
                    outputs[mode] = {k: result[k] for k in ('replay_digest', 'raw_signal_digest')}
                    if refs:
                        expected = reference_for(cache.manifest,refs,mode)['lanes'][bt]
                        for key in ('replay_digest', 'raw_signal_digest'):
                            if key in expected and outputs[mode][key] != expected[key]:
                                raise ValueError(f'{bt}/{mode}: reference {key} mismatch')
                        if cache.manifest['trade_digest'] != expected['trade_digest']:
                            raise ValueError(f'{bt}: reference trade digest mismatch')
                report[str(manifest.parent)] = outputs
                if generation:
                    verification_memory[str(manifest.parent)]={'process':current_memory(),
                        'bounds':enforce_current(mapped_peak),'mapped_files':files}
                del cache,result
            if not report:
                raise ValueError('no cache manifests found')
            if generation:
                verify_source(state)
                if protected_state()!=preflight['protected_state']:
                    raise ValueError('protected set changed during verification')
                report={'lanes':report,'orphans':generation['orphans'],'preflight':preflight,'memory':verification_memory}
        print(json.dumps(report, indent=2))
        return report
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()

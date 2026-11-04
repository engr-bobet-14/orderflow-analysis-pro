"""Read-only authenticated day queries, released only as complete verified UTC hours."""
from __future__ import annotations

import gc
import hashlib
import ctypes
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from orderflow_system.analytics.trade_accumulator import sum_lots
from orderflow_system.data.infobar_feed import DataIntegrityError
from orderflow_system.data.paired_dataset import PRICE_SCALE, QTY_SCALE, ROW_DTYPE, PairedDataset
from orderflow_system.data.trade_alignment import _fetched, _on_grid, _check_file, HOUR_MS, DAY_MS, peak_rss_bytes


class _MallocStatistics(ctypes.Structure):
    _fields_ = [('blocks_in_use', ctypes.c_uint), ('size_in_use', ctypes.c_size_t),
                ('max_size_in_use', ctypes.c_size_t), ('size_allocated', ctypes.c_size_t)]


def malloc_in_use_bytes() -> int | None:
    """macOS live allocator bytes, or unavailable; NULL selects all malloc zones."""
    if sys.platform != 'darwin':
        return None
    try:
        statistics = ctypes.CDLL(None).malloc_zone_statistics
    except (AttributeError, OSError):
        return None
    statistics.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MallocStatistics)]
    statistics.restype = None
    result = _MallocStatistics()
    statistics(None, ctypes.byref(result))
    return int(result.size_in_use)


@dataclass(frozen=True)
class VerifiedHour:
    hour_ms: int
    first_ordinal: int
    rows: np.ndarray


class TradeStream:
    def __init__(self, ds: PairedDataset, start_ordinal: int = 1, end_ordinal: int | None = None,
                 *, threads=4, memory_limit='1GB', temp_dir: Path, batch_rows=1_000_000):
        self.ds = ds
        self.start_ordinal, self.end_ordinal = start_ordinal, end_ordinal
        self.threads, self.memory_limit = threads, memory_limit
        self.temp_dir = Path(temp_dir).resolve()
        if self.temp_dir.is_relative_to('/Volumes'):
            raise ValueError('DuckDB temp directory must be on local disk, outside /Volumes')
        if threads < 1 or batch_rows < 1:
            raise ValueError('threads and batch_rows must be positive')
        self.batch_rows = batch_rows
        if not 1 <= start_ordinal <= ds.manifest['n_trades']:
            raise ValueError('start ordinal outside dataset')
        if end_ordinal is not None and not start_ordinal <= end_ordinal <= ds.manifest['n_trades']:
            raise ValueError('end ordinal outside dataset')
        self.report = {'hours_verified': 0, 'n_verified': 0, 'overscan_n': 0, 'files_verified': 0,
                       'per_day_rss': []}
        self.settings = {}

    def _canonical(self, batch, day):
        cols = {}
        for name in batch.schema.names:
            col = batch.column(name)
            values = col.to_numpy(zero_copy_only=False)
            cols[name] = np.ma.array(values, mask=col.is_null().to_numpy(zero_copy_only=False)) if col.null_count else values
        rows = np.empty(batch.num_rows, dtype=ROW_DTYPE)
        for target, source in (('time_ms', 'trade_time_ms'), ('trade_id', 'trade_id'),
                               ('is_buyer_maker', 'is_buyer_maker')):
            rows[target] = _fetched(cols, source, day)
        rows['price_ticks'] = _on_grid(_fetched(cols, 'price', day), PRICE_SCALE, 'price')
        rows['qty_lots'] = _on_grid(_fetched(cols, 'quantity', day), QTY_SCALE, 'quantity')
        if (rows['price_ticks'] <= 0).any() or (rows['qty_lots'] <= 0).any():
            raise DataIntegrityError('trade units outside positive int64 grid')
        return rows

    def __iter__(self):
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError("trade replay requires duckdb; install the 'tuning' extra") from exc
        try:
            import psutil
        except ImportError:
            psutil = None
        process = psutil.Process() if psutil else None
        self.report['rss_kind'] = 'current' if process else 'peak'

        ds, m = self.ds, self.ds.manifest
        hours = ds.hours.sort_values('hour_ms', ignore_index=True)
        grid = hours.hour_ms.to_numpy()
        first_day, last_day = (int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp()*1000)
                               for d in m['requested_range'])
        if not np.array_equal(grid, np.arange(first_day, last_day+DAY_MS, HOUR_MS)):
            raise DataIntegrityError('hours index does not cover the requested range')
        expected_ordinals = np.r_[1, 1+np.cumsum(hours.n_trades.to_numpy()[:-1])]
        if not np.array_equal(hours.first_ordinal, expected_ordinals):
            raise DataIntegrityError('hours index ordinal discontinuity')
        containing = np.flatnonzero((hours.first_ordinal <= self.start_ordinal) &
                                   (hours.first_ordinal+hours.n_trades > self.start_ordinal))
        start_idx = 0 if self.start_ordinal == 1 else int(containing[0])
        if self.start_ordinal == int(hours.first_ordinal.iloc[start_idx]):
            day = int(grid[start_idx])//DAY_MS
            while (start_idx > 0 and int(grid[start_idx-1])//DAY_MS == day
                   and int(hours.n_trades.iloc[start_idx-1]) == 0):
                start_idx -= 1
        stop_idx = len(hours)-1
        if self.end_ordinal is not None:
            stop_idx = int(np.flatnonzero((hours.first_ordinal <= self.end_ordinal) &
                                         (hours.first_ordinal+hours.n_trades > self.end_ordinal))[0])
        files = [dict(f, abs=Path(m['source_root'])/f['path']) for f in m['files']]
        hashed, last_key = set(), None
        ordinal = int(hours.first_ordinal.iloc[start_idx])
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        # DuckDB owns a private directory; close removes its spill files even on failure.
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(dir=self.temp_dir, prefix='trade-') as spill, duckdb.connect() as con:
            con.execute('SET threads = ?', [self.threads])
            con.execute('SET memory_limit = ?', [self.memory_limit])
            con.execute('SET temp_directory = ?', [spill])
            con.execute("SET max_temp_directory_size = '20GB'")
            con.execute('SET enable_external_file_cache = false')
            for setting in ('threads', 'memory_limit', 'temp_directory', 'max_temp_directory_size',
                            'preserve_insertion_order', 'enable_external_file_cache'):
                self.settings[setting] = con.execute('SELECT current_setting(?)', [setting]).fetchone()[0]
            idx = start_idx
            while idx <= stop_idx:
                lo = int(grid[idx])
                hi = lo//DAY_MS*DAY_MS+DAY_MS
                todays = [f for f in files if f['min_time'] < hi and f['max_time'] >= lo]
                for f in todays:
                    _check_file(f, hashed)
                reader = None
                batches = iter(())
                current = np.empty(0, dtype=ROW_DTYPE)
                try:
                    reader = con.execute('''
                    SELECT trade_time_ms, trade_id, price, quantity, is_buyer_maker
                    FROM read_parquet(?, hive_partitioning = true)
                    WHERE price > 0 AND isfinite(price) AND quantity > 0 AND isfinite(quantity)
                      AND trade_time_ms IS NOT NULL AND trade_time_ms >= ? AND trade_time_ms < ?
                    ORDER BY trade_time_ms, trade_id''', [[str(f['abs']) for f in todays], lo, hi]
                    ).fetch_record_batch(self.batch_rows) if todays else None
                    batches = iter(reader) if reader is not None else iter(())
                    pos, exhausted = 0, False
                    while idx <= stop_idx and int(grid[idx]) < hi:
                        hour_ms = int(grid[idx])
                        parts = []
                        while True:
                            if pos == len(current) and not exhausted:
                                batch = next(batches, None)
                                if batch is None:
                                    exhausted = True
                                else:
                                    current = self._canonical(batch, datetime.fromtimestamp(lo/1000, timezone.utc).date())
                                    pos = 0
                                    t, ids = current['time_ms'], current['trade_id']
                                    if len(t):
                                        if last_key is not None and (int(t[0]), int(ids[0])) <= last_key:
                                            raise DataIntegrityError('trade keys not strictly increasing')
                                        if ((t[1:] < t[:-1]) | ((t[1:] == t[:-1]) & (ids[1:] <= ids[:-1]))).any():
                                            raise DataIntegrityError('trade keys not strictly increasing')
                                        last_key = int(t[-1]), int(ids[-1])
                            if pos == len(current):
                                if exhausted:
                                    break
                                continue
                            if int(current['time_ms'][pos]) >= hour_ms+HOUR_MS:
                                break
                            stop = int(np.searchsorted(current['time_ms'], hour_ms+HOUR_MS))
                            # Copy just this hour's rows; don't retain an entire batch through a view.
                            parts.append(current[pos:stop].copy())
                            pos = stop
                        rows = np.concatenate(parts) if parts else np.empty(0, dtype=ROW_DTYPE)
                        ref = hours.iloc[idx]
                        got = (hour_ms, ordinal, len(rows), sum_lots(rows['qty_lots']), hashlib.sha256(rows.tobytes()).hexdigest())
                        want = (int(ref.hour_ms), int(ref.first_ordinal), int(ref.n_trades), int(ref.sum_qty_lots), ref.digest)
                        if got != want:
                            raise DataIntegrityError(f'hour {hour_ms}: trades {got} != hours.parquet {want}')
                        self.report['hours_verified'] += 1
                        self.report['n_verified'] += len(rows)
                        if self.end_ordinal is not None:
                            self.report['overscan_n'] += max(0, ordinal+len(rows)-1-self.end_ordinal)
                        rows.flags.writeable = False
                        yield VerifiedHour(hour_ms, ordinal, rows)
                        ordinal += len(rows)
                        idx += 1
                        del rows, parts
                finally:
                    if reader is not None:
                        reader.close()
                    # Closing Arrow alone can retain an abandoned ORDER BY result.
                    # Replace it on the owning connection before the quiescent sample.
                    con.execute('SELECT 1').fetchall()
                    del reader
                    del batches, current
                    gc.collect()
                    with con.cursor() as memory:
                        duckdb_bytes = int(memory.execute(
                            'SELECT coalesce(sum(memory_usage_bytes), 0) FROM duckdb_memory()'
                        ).fetchone()[0])
                        tags = dict(memory.execute(
                            'SELECT tag, memory_usage_bytes FROM duckdb_memory() WHERE memory_usage_bytes != 0'
                        ).fetchall())
                    malloc_bytes = malloc_in_use_bytes()
                    self.report['per_day_rss'].append({'day': lo//DAY_MS,
                        'bytes': process.memory_info().rss if process else peak_rss_bytes(),
                        'rss_kind': self.report['rss_kind'], 'duckdb_memory_bytes': duckdb_bytes,
                        'duckdb_memory_tags': tags, 'malloc_in_use_bytes': malloc_bytes,
                        'unavailable': ['malloc_in_use_bytes'] if malloc_bytes is None else []})
            full = self.start_ordinal == 1 and self.end_ordinal is None
            if full:
                for f in files:
                    _check_file(f, hashed)
                if ordinal-1 != m['n_trades'] or ordinal-1 != int(hours.n_trades.sum()):
                    raise DataIntegrityError('manifest trade count mismatch')
            self.report['files_verified'] = len(hashed)
        ds.recheck()


class _Lane:
    def __init__(self, ds, reader, pipeline, ranges):
        from orderflow_system.analytics.trade_accumulator import TradeAccumulator
        from orderflow_system.pipeline import ReplayDigest, TradeDigest

        self.reader, self.pipeline = reader, pipeline
        self.iterator = iter(reader)
        self.pending = next(self.iterator, None)
        if self.pending is None:
            raise DataIntegrityError(f'{reader.info.bar_type.value}: replay window contains no bars')
        self.previous = reader.first_predecessor_seq or 0
        self.initial_previous = self.previous
        if self.previous:
            day = reader.first_predecessor_ms//DAY_MS*DAY_MS
            refs = ds.hours[ds.hours.hour_ms == day]
            if len(refs) != 1:
                raise DataIntegrityError('predecessor day missing from hour index')
            self.w = int(refs.first_ordinal.iloc[0])
        else:
            self.w = 1
        self.last_tn = reader.last_tick_number()
        cfg = pipeline.trade_config
        self.accumulator = TradeAccumulator(round(cfg.footprint_bucket*100), round(cfg.profile_bucket*100), self.w-1)
        self.accumulator.warm_end = self.previous
        self.accumulator.pending_end = self.pending.seq
        self.replay_digest, self.trade_digest = ReplayDigest(), TradeDigest(ranges=ranges)
        self.warm_n = self.warm_lots = self.bar_n = self.bar_lots = self.footprint_lots = 0
        self.final_lots = self.n_tail = self.lots_tail = self.bars = 0
        self.first_seq = self.pending.seq
        self.final_days = 0

    def reconcile(self, bar, update):
        import math
        s = update.stats
        tests = {'n': s.n == bar.seq-self.previous == bar.tick_count,
                 'open': s.open_ticks/100 == bar.open, 'high': s.high_ticks/100 == bar.high,
                 'low': s.low_ticks/100 == bar.low, 'close': s.close_ticks/100 == bar.close,
                 'close_time': s.close_time_ms == bar.timestamp_ms,
                 'ordinals': s.first_ordinal == self.previous+1 and s.last_ordinal == bar.seq,
                 'volume': math.isfinite(bar.volume) and math.isclose(s.lots/1000, bar.volume, rel_tol=1e-9),
                 'dollar': math.isfinite(bar.dollar_value) and math.isclose(s.dollar/1e5, bar.dollar_value, rel_tol=1e-9),
                 'aggressor': s.buy_lots+s.sell_lots == s.lots,
                 'footprint': sum_lots(update.footprint_levels[1])+sum_lots(update.footprint_levels[2]) == s.lots}
        for field, ok in tests.items():
            if not ok:
                raise DataIntegrityError(f'{self.reader.info.bar_type.value}: bar {bar.seq}: {field} mismatch')


class TradeReplay:
    """One verified stream feeds independent reader/accumulator/pipeline lanes.

    Iteration yields (bar_type, BarResult). Reports are available only after exhaustion.
    Optional digest ranges use inclusive global closing-trade ordinals (InfoBar.seq).
    """
    def __init__(self, ds: PairedDataset, lanes, *, temp_dir: Path, threads=4, memory_limit='1GB',
                 batch_rows=1_000_000, digest_ranges=None):
        self.ds = ds
        self.lanes = {}
        for reader, pipeline in lanes:
            bt = reader.info.bar_type.value
            if bt in self.lanes or bt not in ds.bar_types:
                raise ValueError('lanes must have unique registered bar types')
            if reader.info.parquet_path.resolve() != ds.bar_path(bt).resolve() or not reader.strict:
                raise DataIntegrityError('trade replay requires strict readers from the authenticated dataset')
            self.lanes[bt] = _Lane(ds, reader, pipeline, (digest_ranges or {}).get(bt, ()))
        if not self.lanes:
            raise ValueError('at least one lane is required')
        self.full = all(l.reader.start_ms is None and l.reader.end_ms is None for l in self.lanes.values())
        start = min(l.w for l in self.lanes.values())
        end = None if self.full else max(l.last_tn for l in self.lanes.values())
        self.stream = TradeStream(ds, start, end, temp_dir=temp_dir, threads=threads,
                                  memory_limit=memory_limit, batch_rows=batch_rows)
        self.report = None

    def __iter__(self):
        from orderflow_system.pipeline import TradeView

        daily_lots = self.ds.hours.groupby(self.ds.hours.hour_ms//DAY_MS).sum_qty_lots.sum().to_dict()
        for hour in self.stream:
            lo, rows = hour.first_ordinal, hour.rows
            for bt, lane in self.lanes.items():
                acc = lane.accumulator
                pos = max(0, lane.w-lo)
                if pos >= len(rows):
                    continue
                if acc.cursor < lane.initial_previous:
                    stop = min(len(rows), lane.initial_previous-lo+1)
                    if stop > pos:
                        part = rows[pos:stop]
                        acc.warm(lo+pos, part)
                        lane.warm_n += len(part)
                        lane.warm_lots += sum_lots(part['qty_lots'])
                        pos = stop
                while pos < len(rows) and lane.pending is not None:
                    bar = lane.pending
                    stop = min(len(rows), bar.seq-lo+1)
                    if stop <= pos:
                        raise DataIntegrityError('bar ordinal is behind the lane cursor')
                    part = rows[pos:stop]
                    acc.add(lo+pos, part)
                    lane.bar_n += len(part)
                    lane.bar_lots += sum_lots(part['qty_lots'])
                    pos = stop
                    if acc.cursor != bar.seq:
                        break
                    update = acc.close_bar(bar.seq, bar.timestamp_ms)
                    lane.reconcile(bar, update)
                    lane.footprint_lots += sum_lots(update.footprint_levels[1])+sum_lots(update.footprint_levels[2])
                    for day, _, lots in update.finalised_days:
                        total = sum_lots(lots)
                        if total != int(daily_lots[day]):
                            raise DataIntegrityError(f'{bt}: I3 day {day} volume mismatch')
                        lane.final_lots += total
                        lane.final_days += 1
                    result = lane.pipeline.process_bar(bar, update)
                    lane.replay_digest.update(result)
                    lane.trade_digest.update(result, update)
                    lane.bars += 1
                    lane.previous = bar.seq
                    lane.pending = next(lane.iterator, None)
                    if lane.pending is not None:
                        acc.pending_end = lane.pending.seq
                    yield bt, result
                if self.full and pos < len(rows) and lane.pending is None:
                    tail = rows[pos:]
                    lane.n_tail += len(tail)
                    lane.lots_tail += sum_lots(tail['qty_lots'])
        report_lanes = {}
        for bt, lane in self.lanes.items():
            if lane.pending is not None or lane.previous != lane.last_tn:
                raise DataIntegrityError(f'{bt}: bars extend beyond stream')
            developing = lane.accumulator.developing_profiles()
            developing_lots = sum(sum_lots(lots) for _, _, lots in developing)
            identities = {'I1': lane.bar_lots == lane.footprint_lots and
                          lane.bar_n == lane.last_tn-lane.initial_previous,
                          'I2': lane.final_lots+developing_lots == lane.warm_lots+lane.bar_lots,
                          'I3': True}
            if self.full:
                identities['I4'] = (lane.bar_n+lane.n_tail == self.ds.manifest['n_trades'] and
                                    lane.bar_lots+lane.lots_tail == int(self.ds.hours.sum_qty_lots.sum()))
            if not all(identities.values()):
                raise DataIntegrityError(f'{bt}: conservation failed: {identities}')
            engine = lane.pipeline.trade_view or TradeView(lane.pipeline.trade_config, lane.pipeline.config.volume_profile)
            profiles = []
            for day, ticks, lots in developing:
                p = engine.compute_profile(day, ticks, lots)
                profiles.append({'day': day, 'ticks': ticks.tolist(), 'lots': lots.tolist(),
                                 'poc': p.poc, 'vah': p.vah, 'val': p.val, 'source': p.source, 'final': False})
            report_lanes[bt] = {'bars': lane.bars, 'first_seq': lane.first_seq, 'last_seq': lane.last_tn,
                'warm_n': lane.warm_n, 'warm_lots': lane.warm_lots, 'bar_n': lane.bar_n, 'bar_lots': lane.bar_lots,
                'n_tail': lane.n_tail, 'lots_tail': lane.lots_tail, 'finalised_lots': lane.final_lots,
                'finalised_days': lane.final_days, 'developing_lots': developing_lots, 'developing_profiles': profiles,
                'domains': {'W': [lane.w, lane.initial_previous],
                            'B': [lane.initial_previous+1, lane.last_tn],
                            'T': [lane.last_tn+1, self.ds.manifest['n_trades'] if self.full else lane.last_tn]},
                'identities': identities, 'replay_digest': lane.replay_digest.hexdigest(),
                'trade_digest': lane.trade_digest.hexdigest()}
        self.ds.recheck()
        self.report = {'dataset_id': self.ds.dataset_id, 'root_hash': self.ds.root_hash,
                       'full_history': self.full, 'stream': self.stream.report, 'lanes': report_lanes,
                       'duckdb_settings': self.stream.settings}

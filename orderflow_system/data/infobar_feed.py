"""
Information-driven bar ingestion.

Reads the external Parquet dataset

    <root>/<symbol>/<bar_type>/<UTC stamp>.parquet   (+ <UTC stamp>.json sidecar)

and replays it chronologically as `InfoBar` objects. The dataset is read-only:
this module never writes to it.

Contract with the data (verified against the real files, see docs/infobars.md):
  * `Date Time` is the bar CLOSE time (UTC). A bar is therefore only ever
    delivered after it has closed -> no look-ahead by construction.
  * `Tick Number` (cumulative trade index at bar close) is strictly increasing
    and breaks ties between bars that close in the same millisecond.
  * Volume columns are PER BAR despite their "Cumulative" names
    (buy + sell == volume, and Tick Number diffs == Cumulative Ticks).
  * There is no price-level or trade-level data: only bar aggregates.

Ordering is (timestamp, tick number); both must be non-decreasing /
strictly-increasing across the whole stream or the row is rejected.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Iterator, Optional

import numpy as np

from orderflow_system.data.models import BarType, InfoBar

logger = logging.getLogger(__name__)

TS_COL = "Date Time"
COLUMNS = [
    TS_COL, "Tick Number", "Open", "High", "Low", "Close",
    "Cumulative Volume", "Cumulative Buy Volume", "Cumulative Sell Volume",
    "Cumulative Ticks", "Cumulative Dollar Value", "Threshold",
]
_VOL_REL_TOL = 1e-9   # |buy + sell - volume| <= tol * max(1, volume)
_HOUR_MS = 3_600_000


class DataIntegrityError(ValueError):
    """The dataset violates the documented schema / ordering contract."""


# ──────────────────────────────────────────────
# Dataset discovery
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetInfo:
    bar_type: BarType
    symbol: str
    parquet_path: Path
    sidecar_path: Optional[Path]
    sidecar: dict = field(default_factory=dict)
    sidecar_bytes: Optional[bytes] = None

    dataset_id: str = field(init=False)

    def __post_init__(self):
        """Freeze the identity of the discovered file and consumed metadata."""
        path = self.parquet_path.resolve()
        stat = path.stat()
        metadata = (hashlib.sha256(self.sidecar_bytes).hexdigest()
                    if self.sidecar_bytes is not None else "no-sidecar")
        identity = f"{path}|{path.stem}|{stat.st_size}|{stat.st_mtime_ns}|{metadata}"
        object.__setattr__(self, "dataset_id", hashlib.sha256(identity.encode()).hexdigest())

    @property
    def version(self) -> str:
        return self.parquet_path.stem

    @property
    def missing_hours_ms(self) -> list[int]:
        """Start (ms) of every trade-less hour the data builder reported."""
        return sorted(_parse_ts_ms(h) for h in self.sidecar.get("missing_tick_hours", []))


def find_dataset(
    root: str | Path, symbol: str, bar_type: BarType | str, version: Optional[str] = None
) -> DatasetInfo:
    """Locate the parquet (+ sidecar) for a bar type. Newest stamp wins unless pinned."""
    bar_type = BarType(bar_type)
    directory = Path(root) / symbol / bar_type.value
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Info-bar directory not found: {directory} "
            f"(is the external volume mounted?)"
        )
    files = sorted(directory.glob("*.parquet"))
    if version:
        files = [f for f in files if f.stem == version]
    if not files:
        raise FileNotFoundError(f"No parquet file in {directory} (version={version!r})")
    parquet = files[-1]   # stamps are UTC and sort lexicographically
    sidecar_path = parquet.with_suffix(".json")
    sidecar: dict = {}
    sidecar_bytes = None
    if sidecar_path.is_file():
        sidecar_bytes = sidecar_path.read_bytes()
        sidecar = json.loads(sidecar_bytes)
        _check_sidecar(sidecar, symbol, bar_type)
    else:
        sidecar_path = None
        logger.warning("No JSON sidecar next to %s", parquet)
    return DatasetInfo(bar_type, symbol, parquet, sidecar_path, sidecar, sidecar_bytes)


def _check_sidecar(sidecar: dict, symbol: str, bar_type: BarType) -> None:
    if sidecar.get("symbol") not in (None, symbol):
        raise DataIntegrityError(f"Sidecar symbol {sidecar.get('symbol')!r} != {symbol!r}")
    cfg_type = sidecar.get("config", {}).get("bar_type")
    if cfg_type not in (None, bar_type.value):
        raise DataIntegrityError(f"Sidecar bar_type {cfg_type!r} != {bar_type.value!r}")


def _parse_ts_ms(value: str | datetime) -> int:
    """ISO date / datetime -> Unix ms. Naive values are interpreted as UTC."""
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def parse_ts_ms(value: Optional[str]) -> Optional[int]:
    return None if value is None else _parse_ts_ms(value)


# ──────────────────────────────────────────────
# Reader
# ──────────────────────────────────────────────

@dataclass
class ReadReport:
    """What the reader actually saw (counted from the data, not the sidecar)."""
    rows_scanned: int = 0
    rows_yielded: int = 0
    rows_rejected: int = 0
    reject_reasons: Counter = field(default_factory=Counter)
    first_ts_ms: Optional[int] = None
    last_ts_ms: Optional[int] = None
    tied_timestamps: int = 0          # bars closing in the same ms as the previous bar
    max_gap_ms: int = 0
    bars_spanning_missing_hours: int = 0
    row_groups_read: int = 0
    row_groups_skipped: int = 0


class InfoBarReader:
    """
    Chronological iterator over one bar type's dataset.

    start_ms is inclusive, end_ms exclusive (both against bar CLOSE time).
    strict=True raises DataIntegrityError on the first malformed row;
    strict=False drops it and records the reason in `report`.
    """

    def __init__(
        self,
        info: DatasetInfo,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        strict: bool = True,
    ):
        self.info = info
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.strict = strict
        self.report = ReadReport()
        # Populated before the first yield, including when earlier groups are pruned.
        self.first_predecessor_ms: Optional[int] = None
        missing = info.missing_hours_ms
        self._missing = [(h, h + _HOUR_MS) for h in missing]

    # -- internals ---------------------------------------------------------

    def _group_in_range(self, pf, idx: int, ts_col: int) -> bool:
        if self.start_ms is None and self.end_ms is None:
            return True
        stats = pf.metadata.row_group(idx).column(ts_col).statistics
        if stats is None or not stats.has_min_max:
            return True
        if not (isinstance(stats.min, datetime) and isinstance(stats.max, datetime)):
            return True
        lo, hi = _parse_ts_ms(stats.min), _parse_ts_ms(stats.max)   # naive -> UTC
        if self.start_ms is not None and hi < self.start_ms:
            # Lenient acceptance depends on all preceding ordering decisions:
            # a locally valid row may have been rejected against an earlier group.
            # Keep that state by scanning past groups in lenient mode.
            return not self.strict
        if self.end_ms is not None and lo >= self.end_ms:
            return False
        return True

    def _fail(self, reason: str, row: int) -> None:
        if self.strict:
            raise DataIntegrityError(
                f"{self.info.parquet_path.name}: {reason} at file row {row}"
            )
        self.report.rows_rejected += 1
        self.report.reject_reasons[reason] += 1

    def __iter__(self) -> Iterator[InfoBar]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        rep = self.report
        pf = pq.ParquetFile(self.info.parquet_path)
        schema_names = pf.schema_arrow.names
        absent = [c for c in COLUMNS if c not in schema_names]
        if absent:
            raise DataIntegrityError(f"Missing columns {absent} in {self.info.parquet_path.name}")
        ts_type = pf.schema_arrow.field(TS_COL).type
        if not pa.types.is_timestamp(ts_type) or ts_type.unit not in ("s", "ms", "us", "ns"):
            raise DataIntegrityError(f"{TS_COL} must be an Arrow timestamp with s/ms/us/ns unit")
        for name in ("Tick Number", "Cumulative Ticks"):
            if not pa.types.is_integer(pf.schema_arrow.field(name).type):
                raise DataIntegrityError(f"{name} must be an Arrow integer")
        ts_col = schema_names.index(TS_COL)

        def read_group(idx):
            tbl = pf.read_row_group(idx, columns=COLUMNS)
            # Fill timestamp nulls only for conversion: they are rejected below.
            raw_ts = tbl[TS_COL].fill_null(0).cast("int64").to_numpy()
            unit = ts_type.unit
            ts = raw_ts * 1000 if unit == "s" else raw_ts // {"ms": 1, "us": 1000, "ns": 1_000_000}[unit]
            arrays = [tbl[k].to_numpy() for k in COLUMNS[1:]]
            seq, o, h, l, c, vol, bv, sv, nt, dv, th = arrays
            finite = np.isfinite(np.column_stack(arrays)).all(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                bad_price = ~((o > 0) & (h > 0) & (l > 0) & (c > 0))
                bad_ohlc = (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)
                bad_vol = (vol <= 0) | (bv < 0) | (sv < 0) | (
                    np.abs(bv + sv - vol) > _VOL_REL_TOL * np.maximum(1.0, vol))
                vwap = dv / vol
                tolerance = 1e-9 * np.maximum(np.abs(l), np.abs(h))
                bad_money = (dv <= 0) | (vwap < l - tolerance) | (vwap > h + tolerance)
                bad_ticks = nt <= 0
            reasons = np.zeros(len(ts), dtype=object)
            for mask, why in ((bad_ticks, "non_positive_ticks"), (bad_money, "bad_dollar_value"),
                              (bad_vol, "bad_volume"), (bad_ohlc, "inconsistent_ohlc"),
                              (bad_price, "non_positive_price"), (~finite, "non_finite"),
                              (tbl[TS_COL].is_null().to_numpy(), "null_timestamp"),
                              (tbl["Tick Number"].is_null().to_numpy(), "null_seq"),
                              (tbl["Cumulative Ticks"].is_null().to_numpy(), "null_tick_count")):
                reasons[mask] = why
            return ts, arrays, reasons

        prev_ts: Optional[int] = None
        prev_seq = -1
        row_base = 0
        previous_group = None
        for g in range(pf.num_row_groups):
            n_rows = pf.metadata.row_group(g).num_rows
            if not self._group_in_range(pf, g, ts_col):
                rep.row_groups_skipped += 1
                stats = pf.metadata.row_group(g).column(ts_col).statistics
                # Only groups entirely before start can supply past context.
                if (self.start_ms is not None and stats is not None and stats.has_min_max
                        and isinstance(stats.max, datetime) and _parse_ts_ms(stats.max) < self.start_ms):
                    previous_group = g
                row_base += n_rows
                continue
            if previous_group is not None:
                # Row-group pruning must not discard the preceding accepted close.
                # Usually this reads just the previous group's tail; walk backwards
                # only when that group contains no valid rows (lenient input).
                for pg in range(previous_group, -1, -1):
                    past_ts, past_arrays, past_reasons = read_group(pg)
                    accepted_ts, accepted_seq = None, -1
                    for j in range(len(past_ts)):
                        if past_reasons[j]:
                            continue
                        t, seq = int(past_ts[j]), int(past_arrays[0][j])
                        if accepted_ts is None or (t >= accepted_ts and seq > accepted_seq):
                            accepted_ts, accepted_seq = t, seq
                    if accepted_ts is not None:
                        prev_ts, prev_seq = accepted_ts, accepted_seq
                        break
                previous_group = None
            rep.row_groups_read += 1
            ts, arrays, row_reason = read_group(g)
            seq, o, h, l, c, vol, bv, sv, nt, dv, th = arrays
            for i in range(len(ts)):
                rep.rows_scanned += 1
                if row_reason[i]:
                    self._fail(row_reason[i], row_base + i)
                    continue
                t, s = int(ts[i]), int(seq[i])
                if self.end_ms is not None and t >= self.end_ms:
                    return
                if prev_ts is not None and (t < prev_ts or s <= prev_seq):
                    self._fail("out_of_order", row_base + i)
                    continue
                predecessor_ms = prev_ts
                duration = None if prev_ts is None else t - prev_ts
                spans_gap = (duration is not None and
                             any(prev_ts < hi and t > lo for lo, hi in self._missing))
                prev_ts, prev_seq = t, s
                if self.start_ms is not None and t < self.start_ms:
                    continue
                if duration is not None:
                    if duration == 0:
                        rep.tied_timestamps += 1
                    rep.max_gap_ms = max(rep.max_gap_ms, duration)
                if spans_gap:
                    rep.bars_spanning_missing_hours += 1
                if rep.first_ts_ms is None:
                    self.first_predecessor_ms = predecessor_ms
                    rep.first_ts_ms = t
                rep.last_ts_ms = t
                rep.rows_yielded += 1
                yield InfoBar(
                    timestamp_ms=t,
                    open=float(o[i]), high=float(h[i]), low=float(l[i]), close=float(c[i]),
                    volume=float(vol[i]), buy_volume=float(bv[i]), sell_volume=float(sv[i]),
                    tick_count=int(nt[i]), dollar_value=float(dv[i]), threshold=float(th[i]),
                    seq=s, bar_type=self.info.bar_type, duration_ms=duration,
                    data_gap=bool(spans_gap),
                )
            row_base += n_rows


def open_bars(
    root: str | Path,
    symbol: str,
    bar_type: BarType | str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    version: Optional[str] = None,
    strict: bool = True,
) -> InfoBarReader:
    """Convenience: discover + build a reader. `start`/`end` are ISO UTC strings."""
    info = find_dataset(root, symbol, bar_type, version)
    return InfoBarReader(info, parse_ts_ms(start), parse_ts_ms(end), strict)


# ──────────────────────────────────────────────
# Async replay driver
# ──────────────────────────────────────────────

class InfoBarReplayFeed:
    """
    Pushes bars from a reader into an async `on_bar` callback, strictly in order.

    bars_per_second > 0 paces the replay (for watching it on the dashboard);
    0 replays as fast as the consumer allows. Control is yielded to the event
    loop regularly so the dashboard stays responsive either way.
    """

    def __init__(
        self,
        reader: InfoBarReader,
        on_bar: Callable[[InfoBar], Awaitable[None]],
        bars_per_second: float = 0.0,
        yield_every: int = 256,
    ):
        self.reader = reader
        self.on_bar = on_bar
        self.bars_per_second = bars_per_second
        self.yield_every = yield_every
        self._stop = False
        self.bars_sent = 0

    async def run(self) -> int:
        interval = 1.0 / self.bars_per_second if self.bars_per_second > 0 else 0.0
        next_at = time.monotonic()
        for bar in self.reader:
            if self._stop:
                break
            await self.on_bar(bar)
            self.bars_sent += 1
            if interval:
                next_at += interval
                delay = next_at - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    next_at = time.monotonic()
            elif self.bars_sent % self.yield_every == 0:
                await asyncio.sleep(0)
        return self.bars_sent

    async def stop(self) -> None:
        self._stop = True

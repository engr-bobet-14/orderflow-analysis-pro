"""
Full-history trade alignment of a paired dataset (plan §4, §7.1): fail-fast, read-only.

One chronological pass, one UTC day at a time, over the dataset's frozen source-file list:
  * every source file is SHA-256-checked against the trade manifest when first touched;
  * every UTC hour's canonical rows (time, trade_id, price ticks, qty lots, aggressor side) must
    match hours.parquet exactly: first ordinal, count, lots and digest;
  * every bar of every bar type is rebuilt from its trade ordinals (TN[k-1], TN[k]] and compared with
    the parquet: Tick Number, Cumulative Ticks, OHLC and Date Time (= its closing trade's time)
    exactly; volume, tick-rule buy/sell volume and dollar value within 1e-9 relative (RiskLabAI tick
    rule: initial rule 0, carried across bars). Null or non-finite values fail before any tolerance;
  * a null or non-boolean trade field (trade_id, is_buyer_maker) fails before canonicalisation;
  * the unfinished tail of each bar type must hold exactly n_trades - last Tick Number trades (I4).

The first mismatch raises DataIntegrityError; nothing is skipped or repaired. Memory is bounded by one
UTC day of trades plus one Tick Number column per bar type; bars are streamed from their parquet.
"""

from __future__ import annotations

import hashlib
import resource
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from orderflow_system.data.infobar_feed import DataIntegrityError
from orderflow_system.data.paired_dataset import PRICE_SCALE, QTY_SCALE, ROW_DTYPE, PairedDataset, sha256_file

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a dev dependency
    psutil = None

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
REL_TOL = 1e-9
_GRID_TOL = 1e-6
TIME = "Date Time"     # bar close; compared as integer microseconds
EXACT = ("Cumulative Ticks", "Open", "High", "Low", "Close", TIME)
APPROX = ("Cumulative Volume", "Cumulative Buy Volume", "Cumulative Sell Volume", "Cumulative Dollar Value")
FLOATS = ("Open", "High", "Low", "Close") + APPROX
# The builder's bar schema (information-driven-bars src/info_bars.py BAR_SCHEMA). Types are checked before
# any value is read: an int64 cast keeps a timestamp's DECLARED unit, so a wrong unit would compare raw ints.
BAR_TYPES = {TIME: pa.timestamp("us", tz="UTC"), "Tick Number": pa.int64(), "Cumulative Ticks": pa.int64(),
             **{c: pa.float64() for c in FLOATS}}
_EMPTY = hashlib.sha256(b"").hexdigest()


def _day_ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def peak_rss_bytes() -> int:
    """This process's peak RSS (ru_maxrss is bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


class _BarStream:
    """Sequential reader of one bar parquet: `take(k)` returns the next k bars' columns."""

    def __init__(self, path: Path):
        self._name = path.name
        self._batches = pq.ParquetFile(path).iter_batches(batch_size=65_536, columns=list(EXACT + APPROX))
        self._buf: dict[str, np.ndarray] = {c: np.empty(0) for c in EXACT + APPROX}
        self._pos = 0

    def _column(self, batch, name: str) -> np.ndarray:
        col = batch.column(name)
        if col.null_count:
            raise DataIntegrityError(f"{self._name}: null {name} in a bar")
        return (col.cast(pa.int64()) if name == TIME else col).to_numpy()   # TIME -> epoch microseconds

    def take(self, k: int) -> dict[str, np.ndarray]:
        while len(self._buf["Open"]) - self._pos < k:
            batch = next(self._batches)
            self._buf = {c: np.concatenate([self._buf[c][self._pos:], self._column(batch, c)]) for c in self._buf}
            self._pos = 0
        out = {c: v[self._pos:self._pos + k] for c, v in self._buf.items()}
        self._pos += k
        return out


@dataclass
class _Partial:
    """Trades of the bar in progress (not yet closed) carried across day boundaries."""
    n: int = 0
    open: float = 0.0
    high: float = -np.inf
    low: float = np.inf
    vol: float = 0.0
    buy: float = 0.0
    dollar: float = 0.0

    def add(self, p, q, buy_q, pq_) -> None:
        if not len(p):
            return
        if self.n == 0:
            self.open = float(p[0])
        self.n += len(p)
        self.high, self.low = max(self.high, float(p.max())), min(self.low, float(p.min()))
        self.vol += float(q.sum())
        self.buy += float(buy_q.sum())
        self.dollar += float(pq_.sum())


class _TypeState:
    def __init__(self, bar_type: str, path: Path):
        self.bar_type = bar_type
        schema = pq.read_schema(path)
        for name, want in BAR_TYPES.items():
            got = schema.field(name).type if name in schema.names else None
            if got != want:
                raise DataIntegrityError(f"{bar_type}: column {name} has type {got}, expected {want}")
        tn = pq.read_table(path, columns=["Tick Number"]).column(0)
        if tn.null_count:
            raise DataIntegrityError(f"{bar_type}: null Tick Number in a bar")
        self.tn = tn.to_numpy().astype(np.int64)
        if len(self.tn) and (np.diff(self.tn) <= 0).any():
            raise DataIntegrityError(f"{bar_type}: Tick Number is not strictly increasing")
        self.bars = _BarStream(path)
        self.next = 0                  # index of the next bar to close
        self.carry = _Partial()

    def consume(self, o_lo: int, t, p, q, rule) -> None:
        """Trades with ordinals o_lo .. o_lo+len(p)-1 (one UTC day, times `t` in ms); close and compare
        every bar ending here."""
        o_hi = o_lo + len(p) - 1
        stop = int(np.searchsorted(self.tn, o_hi, side="right"))
        ends = self.tn[self.next:stop] - o_lo           # local index of each closing trade
        buy_q, pq_ = np.where(rule == 1, q, 0.0), p * q
        if not len(ends):
            self.carry.add(p, q, buy_q, pq_)
            return
        starts = np.concatenate([[0], ends[:-1] + 1])
        last = int(ends[-1]) + 1
        got = {
            "Cumulative Ticks": ends - starts + 1,
            "Open": p[starts].copy(), "Close": p[ends],
            "High": np.maximum.reduceat(p[:last], starts), "Low": np.minimum.reduceat(p[:last], starts),
            "Cumulative Volume": np.add.reduceat(q[:last], starts),
            "Cumulative Buy Volume": np.add.reduceat(buy_q[:last], starts),
            "Cumulative Dollar Value": np.add.reduceat(pq_[:last], starts),
            TIME: t[ends] * 1000,                       # close = time of the bar's last trade (us)
        }
        c = self.carry
        if c.n:                                          # first bar closing today began earlier
            got["Cumulative Ticks"][0] += c.n
            got["Open"][0] = c.open
            got["High"][0], got["Low"][0] = max(got["High"][0], c.high), min(got["Low"][0], c.low)
            got["Cumulative Volume"][0] += c.vol
            got["Cumulative Buy Volume"][0] += c.buy
            got["Cumulative Dollar Value"][0] += c.dollar
        got["Cumulative Sell Volume"] = got["Cumulative Volume"] - got["Cumulative Buy Volume"]
        expected = self.bars.take(len(ends))
        for col in FLOATS:   # NaN/Inf compare False against any tolerance: reject them outright
            for side, values in (("parquet", expected[col]), ("trades", got[col])):
                if not np.isfinite(values).all():
                    i = int(np.argmax(~np.isfinite(values)))
                    raise DataIntegrityError(
                        f"{self.bar_type}: bar #{self.next + i} {col}: non-finite {side} value {values[i]!r}")
        for col in EXACT + APPROX:
            e, g = expected[col], got[col]
            bad = (e != g) if col in EXACT else np.abs(e - g) > REL_TOL * np.maximum(1.0, np.abs(e))
            if bad.any():
                i = int(np.argmax(bad))
                raise DataIntegrityError(
                    f"{self.bar_type}: bar #{self.next + i} (Tick Number {int(self.tn[self.next + i])}) "
                    f"{col}: parquet {e[i]!r} != trades {g[i]!r}")
        self.next = stop
        self.carry = _Partial()
        self.carry.add(p[last:], q[last:], buy_q[last:], pq_[last:])


def _on_grid(values: np.ndarray, scale: int, what: str) -> np.ndarray:
    scaled = values * scale
    units = np.rint(scaled)
    if len(units) and np.abs(scaled - units).max() > _GRID_TOL:
        raise DataIntegrityError(f"{what} off the 1/{scale} grid")
    return units.astype(np.int64)


_FETCH_KIND = {"trade_time_ms": "i", "trade_id": "i", "price": "f", "quantity": "f", "is_buyer_maker": "b"}


def _fetched(cols: dict, name: str, day: date) -> np.ndarray:
    """A DuckDB column as a plain array; NULLs (masked entries) and wrong types fail, never coerce."""
    values = cols[name]
    if np.ma.isMaskedArray(values):
        if values.mask.any():
            raise DataIntegrityError(f"{day}: null {name} in the trade stream")
        values = values.data
    values = np.asarray(values)
    if values.dtype.kind != _FETCH_KIND[name]:
        raise DataIntegrityError(f"{day}: {name} has dtype {values.dtype}, expected kind {_FETCH_KIND[name]!r}")
    return values.astype({"i": np.int64, "f": np.float64, "b": bool}[_FETCH_KIND[name]], copy=False)


def _check_file(f: dict, hashed: set[str]) -> None:
    """SHA-256 a selected source file against the trade manifest, once."""
    if f["path"] in hashed:
        return
    if not f["abs"].is_file() or f["abs"].stat().st_size != f["size"] or sha256_file(f["abs"]) != f["sha256"]:
        raise DataIntegrityError(f"source file {f['path']} does not match the trade manifest")
    hashed.add(f["path"])


def verify_alignment(ds: PairedDataset, *, threads: int = 4) -> dict:
    """Run the full-history check; returns a report with status "passed" or raises DataIntegrityError."""
    import duckdb  # optional dependency (the `tuning` extra), needed only here

    m = ds.manifest
    source_root = Path(m["source_root"])
    files = [dict(f, abs=source_root / f["path"]) for f in m["files"]]
    hashed: set[str] = set()
    types = {bt: _TypeState(bt, ds.bar_path(bt)) for bt in ds.bar_types}
    hours = ds.hours.sort_values("hour_ms", ignore_index=True)
    first_day, last_day = (date.fromisoformat(d) for d in m["requested_range"])
    if len(hours) != ((last_day - first_day).days + 1) * 24 or int(hours.hour_ms[0]) != _day_ms(first_day):
        raise DataIntegrityError("hours.parquet does not cover the requested range hour by hour")
    next_ordinal, prev_price, prev_rule, last_key = 1, None, 0.0, None
    per_day_rss: list[int] = []
    process = psutil.Process() if psutil else None

    with duckdb.connect() as con:
        con.execute(f"SET threads = {int(threads)}")
        day, hour_row = first_day, 0
        while day <= last_day:
            lo, hi = _day_ms(day), _day_ms(day) + DAY_MS
            todays = [f for f in files if f["min_time"] < hi and f["max_time"] >= lo]
            for f in todays:
                _check_file(f, hashed)
            cols = con.execute("""
                SELECT trade_time_ms, trade_id, price, quantity, is_buyer_maker
                FROM read_parquet(?, hive_partitioning = true)
                WHERE price > 0 AND isfinite(price) AND quantity > 0 AND isfinite(quantity)
                  AND trade_time_ms IS NOT NULL AND trade_time_ms >= ? AND trade_time_ms < ?
                ORDER BY trade_time_ms, trade_id""", [[str(f["abs"]) for f in todays], lo, hi]
            ).fetchnumpy() if todays else None
            t = _fetched(cols, "trade_time_ms", day) if cols else np.empty(0, np.int64)
            n = len(t)
            rows = np.empty(n, dtype=ROW_DTYPE)
            if n:
                ids = _fetched(cols, "trade_id", day)
                prev_t = np.concatenate([[last_key[0]], t[:-1]]) if last_key else t[:-1]
                prev_i = np.concatenate([[last_key[1]], ids[:-1]]) if last_key else ids[:-1]
                cur_t, cur_i = (t, ids) if last_key else (t[1:], ids[1:])
                if ((cur_t < prev_t) | ((cur_t == prev_t) & (cur_i <= prev_i))).any():
                    raise DataIntegrityError(f"{day}: (trade_time_ms, trade_id) not strictly increasing")
                last_key = (int(t[-1]), int(ids[-1]))
                price = _fetched(cols, "price", day)
                qty = _fetched(cols, "quantity", day)
                rows["time_ms"], rows["trade_id"] = t, ids
                rows["price_ticks"] = _on_grid(price, PRICE_SCALE, f"{day}: price")
                rows["qty_lots"] = _on_grid(qty, QTY_SCALE, f"{day}: quantity")
                rows["is_buyer_maker"] = _fetched(cols, "is_buyer_maker", day)
            bounds = np.searchsorted(t, lo + HOUR_MS * np.arange(25))
            for h in range(24):
                part = rows[bounds[h]:bounds[h + 1]]
                ref = hours.iloc[hour_row]
                got = (lo + h * HOUR_MS, next_ordinal + int(bounds[h]), len(part), int(part["qty_lots"].sum()),
                       hashlib.sha256(part.tobytes()).hexdigest() if len(part) else _EMPTY)
                want = (int(ref.hour_ms), int(ref.first_ordinal), int(ref.n_trades), int(ref.sum_qty_lots), ref.digest)
                if got != want:
                    raise DataIntegrityError(f"hour {datetime.fromtimestamp(got[0] / 1000, timezone.utc)}: "
                                             f"trades (ordinal, n, lots, digest) {got[1:]} != hours.parquet {want[1:]}")
                hour_row += 1
            if n:
                diff = np.diff(price, prepend=price[0] if prev_price is None else prev_price)
                sign = np.sign(diff)
                last_nz = np.maximum.accumulate(np.where(sign != 0, np.arange(n), -1))
                rule = np.where(last_nz >= 0, sign[np.maximum(last_nz, 0)], prev_rule)
                prev_price, prev_rule = float(price[-1]), float(rule[-1])
                for st in types.values():
                    st.consume(next_ordinal, t, price, qty, rule)
                next_ordinal += n
            per_day_rss.append(process.memory_info().rss if process else peak_rss_bytes())
            del cols, rows, t
            day += timedelta(days=1)

    for f in files:   # margin-day files that added no in-range rows are part of the selection too
        _check_file(f, hashed)
    total = next_ordinal - 1
    if total != m["n_trades"] or total != int(hours.n_trades.sum()):
        raise DataIntegrityError(f"stream has {total} trades; manifest {m['n_trades']}, hours {int(hours.n_trades.sum())}")
    report_types = {}
    for bt, st in types.items():
        if st.next < len(st.tn):
            raise DataIntegrityError(f"{bt}: Tick Number {int(st.tn[st.next])} lies beyond the {total}-trade stream")
        last_tn = int(st.tn[-1]) if len(st.tn) else 0
        if st.carry.n != total - last_tn:
            raise DataIntegrityError(f"{bt}: tail holds {st.carry.n} trades, expected {total - last_tn}")
        report_types[bt] = {"bars_checked": int(len(st.tn)), "last_tick_number": last_tn, "tail_trades": st.carry.n}
    ds.recheck()
    return {
        "status": "passed", "full_history": True, "dataset_id": ds.dataset_id, "root_hash": ds.root_hash,
        "requested_range": m["requested_range"], "n_trades": total, "hours_verified": hour_row,
        "files_verified": len(hashed), "types": report_types,
        "rss": {"per_day_bytes": per_day_rss, "peak_bytes": peak_rss_bytes()},
    }

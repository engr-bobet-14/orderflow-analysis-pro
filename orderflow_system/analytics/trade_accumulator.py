"""Pure integer trade aggregation. Sub-dollar ties use exact half-even, unlike legacy floats."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from orderflow_system.data.infobar_feed import DataIntegrityError

DAY_MS = 86_400_000
MAX_INT = np.iinfo(np.int64).max


def exact_value_area(lots, value_area_pct) -> tuple[int, int, int]:
    """First-max POC and inclusive value-area indices, using exact decimal lots thresholds."""
    pct = Fraction(repr(value_area_pct))
    if not 0 < pct <= 1:
        raise ValueError('value_area_pct must be in (0, 1]')
    poc = int(np.argmax(lots))
    lo = hi = poc
    accumulated = int(lots[poc])
    total = sum(map(int, lots))
    while accumulated*pct.denominator < total*pct.numerator:
        if hi+1 == len(lots) and lo == 0:
            break
        up = int(lots[hi+1]) if hi+1 < len(lots) else -1
        down = int(lots[lo-1]) if lo > 0 else -1
        if up >= down:
            hi += 1
            accumulated += up
        else:
            lo -= 1
            accumulated += down
    return poc, lo, hi


def bucket_index(price_ticks, bucket_ticks: int):
    if not isinstance(bucket_ticks, (int, np.integer)) or bucket_ticks < 1:
        raise ValueError('bucket_ticks must be a positive integer')
    q, r = np.divmod(price_ticks, bucket_ticks)
    # Compare r with b-r to avoid overflowing 2*r.
    return q + ((r > bucket_ticks-r) | ((r == bucket_ticks-r) & (q % 2 != 0)))


def _checked(value: int) -> int:
    if value < 0 or value > MAX_INT:
        raise DataIntegrityError('trade integer overflow')
    return value


def sum_lots(values) -> int:
    if not len(values):
        return 0
    if int(values.max()) <= MAX_INT // len(values):
        return _checked(int(values.sum(dtype=np.int64)))
    return _checked(sum(map(int, values)))


def _merge(target: dict, keys, lots):
    if not len(keys):
        return
    order = np.argsort(keys, kind='stable')
    keys, lots = keys[order], lots[order]
    unique, starts = np.unique(keys, return_index=True)
    totals = np.add.reduceat(lots, starts)
    for key, total in zip(unique, totals):
        target[int(key)] = _checked(target.get(int(key), 0) + int(total))


def _array(values):
    a = np.asarray(values, dtype='<i8')
    a.flags.writeable = False
    return a


def _buckets(prices, width):
    indices = bucket_index(prices, width)
    if (indices > MAX_INT//width).any():
        raise DataIntegrityError('trade bucket integer overflow')
    return indices*width


def _profile(day, levels):
    keys = sorted(levels)
    return day, _array(keys), _array([levels[k] for k in keys])


@dataclass(frozen=True)
class TradeBarStats:
    open_ticks: int
    high_ticks: int
    low_ticks: int
    close_ticks: int
    n: int
    lots: int
    buy_lots: int
    sell_lots: int
    dollar: int
    open_time_ms: int
    close_time_ms: int
    first_ordinal: int
    last_ordinal: int

    @property
    def aggressor_delta_lots(self):
        return self.buy_lots - self.sell_lots


@dataclass(frozen=True)
class TradeBarUpdate:
    stats: TradeBarStats
    footprint_levels: tuple[np.ndarray, np.ndarray, np.ndarray]
    finalised_days: list[tuple[int, np.ndarray, np.ndarray]]


class TradeAccumulator:
    def __init__(self, fp_bucket_ticks: int, pf_bucket_ticks: int, cursor: int):
        bucket_index(0, fp_bucket_ticks)
        bucket_index(0, pf_bucket_ticks)
        self.fp_bucket_ticks, self.pf_bucket_ticks = fp_bucket_ticks, pf_bucket_ticks
        self.cursor = cursor
        # The replay driver supplies these bounds from its pending bars. A live
        # caller may leave them unset and choose when to close the open bar.
        self.warm_end: int | None = None
        self.pending_end: int | None = None
        self._days: dict[int, dict[int, int]] = {}
        self._buy, self._sell = {}, {}
        self._stats: TradeBarStats | None = None

    def _continuity(self, first_ordinal, rows, end):
        if first_ordinal != self.cursor + 1:
            raise DataIntegrityError(f'ordinal {first_ordinal} follows cursor {self.cursor}')
        if end is not None and self.cursor + len(rows) > end:
            raise DataIntegrityError('ordinal slice exceeds lane boundary')

    def _profiles(self, rows):
        days = rows['time_ms'] // DAY_MS
        buckets = _buckets(rows['price_ticks'], self.pf_bucket_ticks)
        for day in np.unique(days):
            mask = days == day
            _merge(self._days.setdefault(int(day), {}), buckets[mask], rows['qty_lots'][mask])

    def warm(self, first_ordinal, rows):
        self._continuity(first_ordinal, rows, self.warm_end)
        sum_lots(rows['qty_lots'])
        self._profiles(rows)
        self.cursor += len(rows)

    def add(self, first_ordinal, rows):
        self._continuity(first_ordinal, rows, self.pending_end)
        if not len(rows):
            return
        p, q = rows['price_ticks'], rows['qty_lots']
        lots = sum_lots(q)
        if int(p.max()) <= MAX_INT // max(1, int(q.max())):
            dollar = sum_lots(p*q)
        else:
            dollar = _checked(sum(int(x)*int(y) for x, y in zip(p, q)))
        buy = sum_lots(q[rows['is_buyer_maker'] == 0])
        old = self._stats
        self._stats = TradeBarStats(
            old.open_ticks if old else int(p[0]), max(old.high_ticks if old else 0, int(p.max())),
            min(old.low_ticks if old else int(p.min()), int(p.min())), int(p[-1]),
            (old.n if old else 0)+len(rows), _checked((old.lots if old else 0)+lots),
            _checked((old.buy_lots if old else 0)+buy),
            _checked((old.sell_lots if old else 0)+lots-buy),
            _checked((old.dollar if old else 0)+dollar),
            old.open_time_ms if old else int(rows['time_ms'][0]), int(rows['time_ms'][-1]),
            old.first_ordinal if old else first_ordinal, first_ordinal+len(rows)-1,
        )
        buckets = _buckets(p, self.fp_bucket_ticks)
        for maker, target in ((0, self._buy), (1, self._sell)):
            mask = rows['is_buyer_maker'] == maker
            _merge(target, buckets[mask], q[mask])
        self._profiles(rows)
        self.cursor += len(rows)

    def close_bar(self, tn, close_ms) -> TradeBarUpdate:
        if self.cursor != tn or self._stats is None or self._stats.close_time_ms != close_ms:
            raise DataIntegrityError('bar close does not match accumulated trades')
        keys = sorted(self._buy.keys() | self._sell.keys())
        levels = (_array(keys), _array([self._buy.get(k, 0) for k in keys]),
                  _array([self._sell.get(k, 0) for k in keys]))
        final = [_profile(d, self._days.pop(d)) for d in sorted(self._days) if d < close_ms//DAY_MS]
        result = TradeBarUpdate(self._stats, levels, final)
        self._stats, self._buy, self._sell = None, {}, {}
        return result

    def developing_profiles(self):
        return [_profile(d, levels) for d, levels in sorted(self._days.items())]

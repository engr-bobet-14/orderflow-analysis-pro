"""
Delta Engine
Computes horizontal delta, vertical delta, cumulative delta, and delta rate of change.
Core component for detecting absorption, initiative, exhaustion, and divergence.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from orderflow_system.data.models import Tick, InfoBar, Side


@dataclass
class DeltaResult:
    """Delta analysis for a single bar or tick window."""
    vertical_delta: float = 0.0          # buy_vol - sell_vol for this bar
    cumulative_delta: float = 0.0        # Running total across bars
    horizontal_delta: dict[float, float] = field(default_factory=dict)
    # per-price: buy_vol - sell_vol. EMPTY (= unknown) for info bars: only
    # real trade-level data (compute_from_ticks) can populate it.
    max_delta_price: Optional[float] = None   # Price with strongest buy delta (None if no price levels)
    min_delta_price: Optional[float] = None   # Price with strongest sell delta (None if no price levels)
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    delta_pct: float = 0.0              # delta / total_volume
    volume_rate: Optional[float] = None  # participation: volume per second (None = unknown)


class DeltaEngine:
    """
    Computes delta metrics used throughout the strategy.

    Key concepts from Fabio:
      - Horizontal delta: buy_vol - sell_vol at EACH price level within a bar (trade-level data only)
      - Vertical delta: total aggressive buys - total aggressive sells per bar
      - Cumulative delta: running sum across bars, used for divergence detection
      - Delta divergence: price makes new high but cum_delta doesn't → weakening
    """

    def __init__(self, tick_size: float = 0.1):
        self.tick_size = tick_size
        self._cumulative_delta = 0.0
        self._delta_history: list[DeltaResult] = []
        self._max_history = 500

    def reset(self):
        self._cumulative_delta = 0.0
        self._delta_history.clear()

    def compute_from_bar(self, bar: InfoBar) -> DeltaResult:
        """Vertical + cumulative delta of an info bar (no price-level delta exists)."""
        vertical_delta = bar.buy_volume - bar.sell_volume
        self._cumulative_delta += vertical_delta

        total = bar.buy_volume + bar.sell_volume
        result = DeltaResult(
            vertical_delta=vertical_delta,
            cumulative_delta=self._cumulative_delta,
            buy_volume=bar.buy_volume,
            sell_volume=bar.sell_volume,
            delta_pct=vertical_delta / total if total > 0 else 0.0,
            volume_rate=bar.volume_rate,
        )
        self._remember(result)
        return result

    def _remember(self, result: DeltaResult) -> None:
        self._delta_history.append(result)
        if len(self._delta_history) > self._max_history:
            self._delta_history = self._delta_history[-self._max_history:]

    def compute_from_ticks(
        self, ticks: list[Tick], tick_size: Optional[float] = None
    ) -> DeltaResult:
        """Compute delta from a window of ticks."""
        ts = tick_size or self.tick_size
        h_delta: dict[float, float] = defaultdict(float)
        buy_vol = 0.0
        sell_vol = 0.0

        for t in ticks:
            rounded = round(round(t.price / ts) * ts, 10)
            if t.is_buy:
                buy_vol += t.size
                h_delta[rounded] += t.size
            else:
                sell_vol += t.size
                h_delta[rounded] -= t.size

        vertical_delta = buy_vol - sell_vol
        self._cumulative_delta += vertical_delta

        max_delta_price = max(h_delta, key=lambda k: h_delta[k]) if h_delta else None
        min_delta_price = min(h_delta, key=lambda k: h_delta[k]) if h_delta else None
        total = buy_vol + sell_vol
        span_s = (ticks[-1].timestamp_ms - ticks[0].timestamp_ms) / 1000.0 if ticks else 0.0

        result = DeltaResult(
            vertical_delta=vertical_delta,
            cumulative_delta=self._cumulative_delta,
            volume_rate=total / max(span_s, 1.0) if ticks else None,
            horizontal_delta=dict(h_delta),
            max_delta_price=max_delta_price,
            min_delta_price=min_delta_price,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            delta_pct=vertical_delta / total if total > 0 else 0.0,
        )

        self._remember(result)
        return result

    @property
    def cumulative_delta(self) -> float:
        return self._cumulative_delta

    @property
    def history(self) -> list[DeltaResult]:
        return self._delta_history

    def get_delta_roc(self, lookback: int = 5) -> float:
        """Rate of change of vertical delta over last N bars."""
        if len(self._delta_history) < lookback:
            return 0.0
        recent = [d.vertical_delta for d in self._delta_history[-lookback:]]
        if len(recent) < 2:
            return 0.0
        # Simple slope via linear regression
        x = np.arange(len(recent), dtype=float)
        y = np.array(recent, dtype=float)
        if np.std(x) == 0:
            return 0.0
        slope = float(np.polyfit(x, y, 1)[0])
        return slope

    def get_volume_trend(self, lookback: int = 5) -> float:
        """Slope of participation rate (volume/sec) over last N bars.
        Declining = exhaustion clue. Rate, not raw volume: volume/dollar bars
        have near-constant volume, their effort shows up as time compression."""
        if len(self._delta_history) < lookback:
            return 0.0
        recent = [d.volume_rate for d in self._delta_history[-lookback:]]
        if any(r is None for r in recent):
            return 0.0      # unknown rate in the window: no trend claimed
        x = np.arange(len(recent), dtype=float)
        y = np.array(recent, dtype=float)
        if np.std(x) == 0:
            return 0.0
        slope = float(np.polyfit(x, y, 1)[0])
        return slope

    def detect_delta_peaks(
        self, lookback: int = 20
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """
        Find local peaks and troughs in cumulative delta for divergence detection.
        Returns (peaks, troughs) as lists of (index, value).
        """
        if len(self._delta_history) < 3:
            return [], []

        history = self._delta_history[-lookback:]
        cd = [d.cumulative_delta for d in history]
        peaks = []
        troughs = []

        for i in range(1, len(cd) - 1):
            if cd[i] > cd[i - 1] and cd[i] > cd[i + 1]:
                peaks.append((i, cd[i]))
            elif cd[i] < cd[i - 1] and cd[i] < cd[i + 1]:
                troughs.append((i, cd[i]))

        return peaks, troughs

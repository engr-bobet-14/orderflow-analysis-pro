"""
Initiative Auction Detector
Detects aggressive momentum with follow-through — effort + result aligned.

From Fabio:
  "Strong delta and a candle that closes on the upside — delta is leading the price.
   This is the best example of aggressive momentum — initiative auction."

  "Constant aggression of the buyer, consistent pressure on the upside,
   one-side imbalance prints... you can use as a really strong point to join
   the trend when it's developing and you have a strong delta."

Logic:
  HIGH effort + HIGH result + directional alignment = INITIATIVE AUCTION
  - Strong delta in one direction
  - Bar closes in same direction as delta
  - Participation above average (volume per second accelerating)
  - One-sided imbalance prints in the footprint (only with real price levels)
"""

from __future__ import annotations

from collections import deque

from typing import Optional

from orderflow_system.data.models import InfoBar, Signal, SignalType, Side
from orderflow_system.analytics.delta import DeltaResult
from orderflow_system.analytics.footprint import FootprintBar
from orderflow_system.config.settings import InitiativeConfig


class InitiativeDetector:
    """
    Detects initiative auction patterns.

    Used for:
    1. Break-even trigger (first initiative after absorption → move SL to BE)
    2. Trailing trigger (each new initiative print → trail SL)
    3. Trend joining signal (strong initiative = join the move)
    """

    def __init__(self, config: InitiativeConfig, tick_size: float = 0.1):
        self.config = config
        self.tick_size = tick_size
        # Last 50 KNOWN participation rates, rather than a fixed bar window.
        # First/gap bars contribute nothing and never evict a known rate.
        self._avg_rate_window: list[float] = []
        self._max_window = 50
        self._signal_history: deque[Signal] = deque(maxlen=5000)

    def check_bar(
        self,
        bar: InfoBar,
        delta: DeltaResult,
        footprint: FootprintBar,
    ) -> Optional[Signal]:
        """Check a completed bar for initiative auction pattern."""
        if bar.volume == 0:
            return None

        # Average the PRIOR known participation rates (the current known rate
        # is added afterwards so it does not dilute its own comparison).
        rate = bar.volume_rate
        if rate is None:
            return None     # first/gap bar: unknown participation contributes nothing
        prior = self._avg_rate_window
        avg_rate = sum(prior) / len(prior) if prior else rate
        self._avg_rate_window.append(rate)
        if len(self._avg_rate_window) > self._max_window:
            self._avg_rate_window = self._avg_rate_window[-self._max_window:]

        # ── Check criteria ──

        # 1. Strong delta (scale-free: share of the bar's volume)
        pct = abs(delta.delta_pct)
        if pct < self.config.min_delta_pct:
            return None

        # 2. Participation acceleration (bar closed faster than recent average)
        accel = rate / avg_rate if avg_rate > 0 else 1.0
        if accel < self.config.volume_acceleration_min:
            return None

        # 3. Price displacement (bar body must be meaningful), basis points
        body_bps = bar.body_size / bar.close * 1e4
        if body_bps < self.config.min_body_bps:
            return None

        # 4. Delta and price must be directionally aligned
        delta_bullish = delta.vertical_delta > 0
        bar_bullish = bar.is_green

        if self.config.delta_price_alignment and delta_bullish != bar_bullish:
            return None

        # ── Direction and signal ──
        direction = Side.BUY if delta_bullish else Side.SELL

        # 5. One-sided imbalance prints (bonus strength) — only with real price levels
        imbalance_count = 0
        if footprint.has_price_levels:
            imbalances = footprint.imbalance_levels(threshold=3.0)
            dir_str = "buy" if delta_bullish else "sell"
            imbalance_count = sum(1 for _, d in imbalances if d == dir_str)

        # Compute strength score
        strength = min(100.0, (
            (pct / self.config.min_delta_pct) * 20               # Delta strength
            + accel * 15                                         # Participation acceleration
            + (body_bps / self.config.min_body_bps) * 5          # Price follow-through
            + imbalance_count * 10                               # Imbalance bonus
        ))

        signal = Signal(
            timestamp_ms=bar.timestamp_ms,
            signal_type=SignalType.INITIATIVE,
            direction=direction,
            price_level=bar.close,
            strength=strength,
            details={
                "delta": delta.vertical_delta,
                "delta_pct": round(delta.delta_pct, 4),
                "volume": bar.volume,
                "avg_rate": round(avg_rate, 2),
                "vol_acceleration": round(accel, 2),
                "body_bps": round(body_bps, 2),
                "imbalance_levels": imbalance_count if footprint.has_price_levels else None,
                "bar_close": "green" if bar.is_green else "red",
            },
        )
        self._signal_history.append(signal)
        return signal

    @property
    def signal_history(self) -> list[Signal]:
        return self._signal_history

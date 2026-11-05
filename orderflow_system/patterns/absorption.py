"""
Absorption Detector
Detects when aggressive orders are absorbed by passive liquidity — no price movement.

From Fabio:
  "High effort from the buyers that received zero reward. This is the textbook
   example for absorption — positive delta but negative closure."

  "72 + 61 + 60 + 62 = ~300 contracts on this horizontal level... all this effort
   is being absorbed. This is a perfect example of absorption."

Logic:
  Effort (aggressive volume)  vs  Result (price displacement)
  HIGH effort + LOW result = ABSORPTION → Entry signal

  On info bars the effort is the bar's delta (|buy - sell| / volume) and the
  result is the bar's close vs open. The level-based method (repeated attempts
  at one price) needs real price-level volume and stays inert on info bars.
"""

from __future__ import annotations

from collections import deque

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from orderflow_system.data.models import (
    InfoBar, Signal, SignalType, Side,
)
from orderflow_system.analytics.footprint import FootprintBar, clipped_trade_bounds, validate_trade_bucket_ticks
from orderflow_system.analytics.delta import DeltaResult
from orderflow_system.config.settings import AbsorptionConfig, TradeSignalConfig
from orderflow_system.utils.clock import Clock, RealClock


@dataclass
class AbsorptionEvent:
    """Tracks absorption building at a price level."""
    price: float
    aggressive_volume: float = 0.0
    price_displacement: float = 0.0
    attempts: int = 0
    absorbing_side: str = ""     # 'buyers_absorbing' or 'sellers_absorbing'
    first_seen_ms: int = 0
    last_seen_ms: int = 0


class AbsorptionDetector:
    """
    Detects absorption patterns in real-time.

    Two detection methods:
    1. Per-bar: strong delta but the bar closes in the opposite direction
       (positive delta + negative close = buyers absorbed = bearish absorption)
    2. Level-based: aggressive volume accumulates at a price level with no
       displacement. Requires a footprint with real price levels; skipped otherwise.

    Multiple attempts at the same level increase confidence.
    """

    def __init__(self, config: AbsorptionConfig, tick_size: float = 0.1, clock: Optional[Clock] = None,
                 *, signals: Optional[TradeSignalConfig] = None):
        self.config = config
        self.signals = signals or TradeSignalConfig()
        self.tick_size = tick_size
        self._clock = clock or RealClock()
        self._active_absorptions: dict[float, AbsorptionEvent] = {}
        self._signal_history: deque[Signal] = deque(maxlen=5000)
        self._cleanup_period_ms = 60_000  # Clean stale events every 60 seconds

    def check_bar(
        self,
        bar: InfoBar,
        footprint: FootprintBar,
        delta: DeltaResult,
        current_price: float,
        *,
        trade_footprint: Optional[FootprintBar] = None,
        trade_bucket_ticks: Optional[int] = None,
    ) -> Optional[Signal]:
        """
        Check a completed bar for absorption.

        Absorption bar signatures:
        - HIGH delta in one direction but bar closes in the OPPOSITE direction
          → Positive delta (buy pressure) + red bar = sellers absorbing the buys
          → Negative delta (sell pressure) + green bar = buyers absorbing the sells
        - High volume at a specific level with no price movement through it
          (only when real price-level data exists)

        A2's trade_footprint must originate from a TradeAccumulator built with
        round(trade_config.footprint_bucket*100), as _Lane does. The pipeline
        supplies trade_bucket_ticks from its TradeFlowConfig for geometry.
        With A2 enabled and a trade footprint supplied, A2 replaces the legacy
        M2 bar-footprint path (inert for info bars).
        """
        if self.signals.absorption_levels and trade_footprint is None:
            raise ValueError('A2 requires a trade footprint')
        if trade_footprint is not None:
            validate_trade_bucket_ticks(trade_bucket_ticks)
        if bar.volume == 0:
            return None

        # ── Method 1: Delta vs Close Mismatch ──
        signal = self._check_delta_close_mismatch(bar, delta, footprint)
        if signal:
            return signal

        if self.signals.absorption_levels and trade_footprint is not None:
            return self._check_trade_absorption(bar, trade_footprint, trade_bucket_ticks)

        # ── Method 2: Level-based absorption ──
        return self._check_level_absorption(bar, footprint, current_price)

    def _check_delta_close_mismatch(
        self,
        bar: InfoBar,
        delta: DeltaResult,
        footprint: FootprintBar,
    ) -> Optional[Signal]:
        """
        Fabio's textbook absorption:
        "Positive delta but negative closure" = aggressive buyers absorbed by passive sellers.
        The opposite direction wins.
        """
        pct = abs(delta.delta_pct)
        if pct < self.config.min_delta_pct:
            return None

        # "No reward": the bar must close against the delta by a real margin.
        adverse_bps = (bar.open - bar.close) / bar.close * 1e4
        strength = min(100.0, (pct / self.config.min_delta_pct) * 40)
        details = {
            "type": "delta_close_mismatch",
            "delta": delta.vertical_delta,
            "delta_pct": round(delta.delta_pct, 4),
            "aggressive_buy_vol": delta.buy_volume,
            "aggressive_sell_vol": delta.sell_volume,
        }

        # Positive delta (buy pressure) but bearish bar
        if delta.vertical_delta > 0 and adverse_bps >= self.config.min_adverse_body_bps:
            # Buyers were absorbed → bearish signal
            details["bar_close"] = "bearish"
            return self._create_signal(
                bar=bar,
                direction=Side.SELL,
                price_level=bar.high,  # Absorption happened at the high
                strength=strength,
                details=details,
            )

        # Negative delta (sell pressure) but bullish bar
        if delta.vertical_delta < 0 and -adverse_bps >= self.config.min_adverse_body_bps:
            # Sellers were absorbed → bullish signal
            details["bar_close"] = "bullish"
            return self._create_signal(
                bar=bar,
                direction=Side.BUY,
                price_level=bar.low,  # Absorption happened at the low
                strength=strength,
                details=details,
            )

        return None

    def _check_trade_absorption(self, bar, footprint, width) -> Optional[Signal]:
        cfg = self.signals
        volume = sum(lv.total_volume for lv in footprint.levels.values())
        if volume <= 0 or bar.range_size <= 0:
            return None
        candidates = []
        for centre, lv in sorted(footprint.levels.items()):
            total = lv.total_volume
            if total <= 0 or total / volume < cfg.min_level_share:
                continue
            lo, hi = clipped_trade_bounds(bar, centre, width)
            if lo > hi:
                continue
            share = total / volume
            for direction, inside, dominant, level, travel in (
                (Side.SELL, lo >= bar.high - cfg.level_zone_frac * bar.range_size,
                 lv.ask_volume, lo, bar.high - lo),
                (Side.BUY, hi <= bar.low + cfg.level_zone_frac * bar.range_size,
                 lv.bid_volume, hi, hi - bar.low),
            ):
                side_ratio = dominant / total
                reject = ((level - bar.close) if direction == Side.SELL else (bar.close - level)) / bar.close * 1e4
                through = travel / level * 1e4
                if (not inside or side_ratio < cfg.min_side_ratio or reject < cfg.min_reject_bps
                        or through > cfg.max_through_bps):
                    continue
                details = dict(type='trade_level_absorption', delta_source='aggressor', bucket_centre=centre,
                               level_lo=lo, level_hi=hi, share=share, side_ratio=side_ratio,
                               reject_bps=reject, through_bps=through, bucket=width / 100)
                strength = min(100.0, 30 * share / cfg.min_level_share + 20 * reject / cfg.min_reject_bps)
                candidates.append((share, -travel, centre, direction, level, strength, details))
        if not candidates or len({c[3] for c in candidates}) > 1:
            return None
        _, _, _, direction, level, strength, details = max(candidates, key=lambda c: c[:3])
        return self._create_signal(bar, direction, level, strength, details)

    def _check_level_absorption(
        self,
        bar: InfoBar,
        footprint: FootprintBar,
        current_price: float,
    ) -> Optional[Signal]:
        """
        Check for absorption at specific price levels within the footprint.
        High volume at a level + price didn't break through = absorption.
        Needs real price-level volume; info bars have none, so nothing is inferred.
        """
        if not footprint.has_price_levels:
            return None

        now_ms = self._clock.now_ms()
        tick_size = self.tick_size

        for price, lv in footprint.levels.items():
            total = lv.total_volume
            if total < self.config.big_trade_filter:
                continue

            # Check: high volume at this level but price displaced little
            price_disp = abs(current_price - price) / max(tick_size, 0.01)
            effort_high = total >= self.config.min_aggressive_volume
            result_low = price_disp <= self.config.max_price_displacement_ticks

            if effort_high and result_low:
                # Track repeated absorption
                rounded = round(price, 4)
                if rounded not in self._active_absorptions:
                    self._active_absorptions[rounded] = AbsorptionEvent(
                        price=rounded,
                        first_seen_ms=now_ms,
                    )

                event = self._active_absorptions[rounded]
                event.aggressive_volume += total
                event.price_displacement = price_disp
                event.attempts += 1
                event.last_seen_ms = now_ms

                # Determine who is absorbing
                if lv.ask_volume > lv.bid_volume:
                    event.absorbing_side = "sellers_absorbing"
                    direction = Side.SELL
                else:
                    event.absorbing_side = "buyers_absorbing"
                    direction = Side.BUY

                # Signal threshold: enough volume or repeated attempts
                if (
                    event.aggressive_volume >= self.config.min_aggressive_volume
                    and event.attempts >= self.config.min_attempts
                ):
                    strength = min(
                        100.0,
                        (event.aggressive_volume / self.config.min_aggressive_volume) * 30
                        + event.attempts * 15,
                    )
                    signal = self._create_signal(
                        bar=bar,
                        direction=direction,
                        price_level=price,
                        strength=strength,
                        details={
                            "type": "level_absorption",
                            "total_aggressive_volume": event.aggressive_volume,
                            "attempts": event.attempts,
                            "absorbing_side": event.absorbing_side,
                            "duration_ms": now_ms - event.first_seen_ms,
                        },
                    )
                    # Reset after signal
                    del self._active_absorptions[rounded]
                    return signal

        # Cleanup stale events
        self._cleanup_stale(now_ms)
        return None

    def _cleanup_stale(self, now_ms: int):
        """Remove absorption events that are too old."""
        stale_threshold = now_ms - (self.config.rolling_window_seconds * 3 * 1000)
        stale_keys = [
            k for k, v in self._active_absorptions.items()
            if v.last_seen_ms < stale_threshold
        ]
        for k in stale_keys:
            del self._active_absorptions[k]

    def _create_signal(
        self,
        bar: InfoBar,
        direction: Side,
        price_level: float,
        strength: float,
        details: dict,
    ) -> Signal:
        signal = Signal(
            timestamp_ms=bar.timestamp_ms,
            signal_type=SignalType.ABSORPTION,
            direction=direction,
            price_level=price_level,
            strength=strength,
            details=details,
        )
        self._signal_history.append(signal)
        return signal

    @property
    def active_absorptions(self) -> dict[float, AbsorptionEvent]:
        return self._active_absorptions

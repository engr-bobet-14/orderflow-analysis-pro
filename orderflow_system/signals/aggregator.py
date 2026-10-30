"""
Signal Aggregator & State Machine
Combines all pattern signals with volume profile context into actionable trade alerts.

Implements Fabio's execution model as a state machine:
  WATCHING → ABSORPTION_DETECTED → POSITION_OPEN → BREAK_EVEN → TRAILING → CLOSED

Signal weighting:
  - Absorption: 30% (primary entry)
  - Delta/Divergence: 25% (confirmation)
  - Volume Profile context: 25% (level qualification)
  - Initiative/Sweep: 20% (BE trigger / trail)
"""

from __future__ import annotations

from collections import deque

import logging
from dataclasses import dataclass, field
from typing import Optional

from orderflow_system.data.models import (
    Signal, SignalType, Side, TradeState, TradePhase, InfoBar, ExecutionLevel, ExecutionSnapshot,
)
from orderflow_system.signals.profile_framing import DailyBias, QualifiedLevel, LevelType
from orderflow_system.config.settings import BiasDirection
from orderflow_system.utils.clock import Clock, RealClock

logger = logging.getLogger(__name__)


@dataclass
class AggregatedSignal:
    """Weighted combination of multiple signals at a qualified level."""
    timestamp_ms: int
    direction: Side
    composite_score: float = 0.0       # 0-100
    qualified_level: Optional[QualifiedLevel] = None
    signals: list[Signal] = field(default_factory=list)
    action: str = ""                    # 'enter', 'break_even', 'trail', 'exit', 'stop', 'target', 'alert_only'
    suggested_sl: float = 0.0
    suggested_tp: float = 0.0
    notes: str = ""
    bar_seq: int = 0                    # InfoBar.seq of the bar that produced this
    execution: ExecutionSnapshot = field(init=False)


class SignalAggregator:
    """
    Combines pattern signals with profile context and manages the trade state machine.

    Flow:
    1. Profile framing qualifies levels and sets daily bias
    2. When price reaches a qualified level, enter WATCHING state
    3. Absorption at the level → ENTRY signal (composite score must pass threshold)
    4. Initiative auction after entry → BREAK EVEN trigger
    5. Subsequent initiative prints → TRAIL stop
    6. Exhaustion or divergence → EXIT / reduce
    """

    def __init__(
        self,
        min_composite_score: float = 60.0,
        signal_cooldown_seconds: float = 60.0,
        price_proximity_pct: float = 0.002,  # 0.2% proximity to qualified level
        clock: Optional[Clock] = None,
    ):
        self.min_composite_score = min_composite_score
        self.signal_cooldown_seconds = signal_cooldown_seconds
        self.price_proximity_pct = price_proximity_pct
        self._clock = clock or RealClock()
        self._active_trades: dict[str, TradeState] = {}  # instrument → trade
        self._watched_levels: dict[str, list[QualifiedLevel]] = {}  # instrument → watched levels
        self._last_signal_time: dict[str, int] = {}      # instrument → timestamp_ms
        self._signal_history: deque[AggregatedSignal] = deque(maxlen=5000)
        self._next_trade_id = 1
        self._execution_levels: dict[str, ExecutionLevel] = {}

    def process_signal(
        self,
        instrument: str,
        signal: Signal,
        bias: Optional[DailyBias],
        current_price: float,
        recent_bars: list[InfoBar],
    ) -> Optional[AggregatedSignal]:
        """
        Process a new pattern signal against the current bias and trade state.
        Returns an aggregated signal if action is needed.
        """
        now_ms = self._clock.now_ms()

        # Cooldown check
        last_ts = self._last_signal_time.get(instrument, 0)
        if now_ms - last_ts < self.signal_cooldown_seconds * 1000:
            return None

        active_trade = self._active_trades.get(instrument)

        # ── State machine routing ──

        if active_trade is None or active_trade.phase == TradePhase.CLOSED:
            # No active trade — check for new entry
            return self._check_new_entry(
                instrument, signal, bias, current_price, now_ms
            )

        elif active_trade.phase == TradePhase.WATCHING:
            # Watching a qualified level — look for absorption or sweep
            if signal.signal_type == SignalType.ABSORPTION:
                return self._handle_absorption_at_level(
                    instrument, signal, bias, active_trade, current_price, now_ms
                )
            elif signal.signal_type == SignalType.SWEEP:
                # Sweep at watched level — generate alert but don't enter
                return self._handle_sweep_at_level(
                    instrument, signal, bias, active_trade, current_price, now_ms
                )

        elif active_trade.phase == TradePhase.POSITION_OPEN:
            # Position open, waiting for BE trigger
            if signal.signal_type == SignalType.INITIATIVE:
                return self._handle_initiative_for_be(
                    instrument, signal, active_trade, current_price, now_ms
                )
            elif signal.signal_type in (SignalType.EXHAUSTION, SignalType.DIVERGENCE):
                return self._handle_exit_warning(
                    instrument, signal, active_trade, current_price, now_ms
                )

        elif active_trade.phase in (TradePhase.BREAK_EVEN, TradePhase.TRAILING):
            # Trailing — update trail or detect exit
            if signal.signal_type == SignalType.INITIATIVE:
                return self._handle_initiative_for_trail(
                    instrument, signal, active_trade, recent_bars, current_price, now_ms
                )
            elif signal.signal_type in (SignalType.EXHAUSTION, SignalType.DIVERGENCE):
                return self._handle_exit_warning(
                    instrument, signal, active_trade, current_price, now_ms
                )

        return None

    def set_watching(
        self, instrument: str, level: QualifiedLevel, direction: Side
    ):
        """Begin watching a qualified level for entry signals."""
        # Track multiple watched levels per instrument (don't overwrite)
        if instrument not in self._watched_levels:
            self._watched_levels[instrument] = []

        # Avoid duplicate levels (same price within 0.01%)
        for existing in self._watched_levels[instrument]:
            if existing.direction == level.direction and abs(existing.price - level.price) / max(level.price, 1) < 0.0001:
                return  # Already watching this level

        self._watched_levels[instrument].append(level)

        # Only create WATCHING trade if no active trade yet
        active = self._active_trades.get(instrument)
        if active is None or active.phase == TradePhase.CLOSED:
            trade = TradeState(
                instrument=instrument,
                direction=direction,
                phase=TradePhase.WATCHING,
                qualified_level=level.price,
            )
            self._active_trades[instrument] = trade
            self._execution_levels.pop(instrument, None)
            logger.info(
                f"[{instrument}] WATCHING {level.level_type.value} "
                f"@ {level.price:.2f} for {'LONG' if direction == Side.BUY else 'SHORT'}"
            )

    def reset_watching(self, instrument: str):
        """Drop watched levels (e.g. at a session rollover). An open position is untouched."""
        self._watched_levels.pop(instrument, None)
        active = self._active_trades.get(instrument)
        if active is not None and active.phase in (TradePhase.WATCHING, TradePhase.CLOSED):
            del self._active_trades[instrument]
            self._execution_levels.pop(instrument, None)

    def _check_new_entry(
        self,
        instrument: str,
        signal: Signal,
        bias: Optional[DailyBias],
        current_price: float,
        now_ms: int,
    ) -> Optional[AggregatedSignal]:
        """Check if a new signal qualifies for entry at a profile level."""
        if signal.signal_type != SignalType.ABSORPTION:
            return None  # Only absorption triggers new entries

        nearest = self._nearest_entry_level(instrument, signal.direction, bias, current_price)
        if nearest is None:
            return None

        # Compute composite score
        score = self._compute_composite_score(signal, nearest, bias)

        if score < self.min_composite_score:
            return None

        # Create trade state
        trade = TradeState(
            instrument=instrument,
            direction=signal.direction,
            qualified_level=nearest.price,
        )
        trade.advance_to_absorption(signal)

        # Compute SL/TP and advance to position
        sl, tp = self._compute_sl_tp(signal.direction, nearest, bias, current_price)
        trade.advance_to_position(current_price, sl, tp, entry_time_ms=now_ms)
        self._active_trades[instrument] = trade

        agg = AggregatedSignal(
            timestamp_ms=now_ms,
            direction=signal.direction,
            composite_score=score,
            qualified_level=nearest,
            signals=[signal],
            action="enter",
            suggested_sl=sl,
            suggested_tp=tp,
            notes=(
                f"ENTRY SIGNAL: Absorption at {nearest.level_type.value} "
                f"({nearest.price:.2f}). Score: {score:.0f}. "
                f"SL: {sl:.2f}, TP: {tp:.2f}"
            ),
        )
        return self._record_action(instrument, trade, agg)

    def _handle_absorption_at_level(
        self,
        instrument: str,
        signal: Signal,
        bias: Optional[DailyBias],
        trade: TradeState,
        current_price: float,
        now_ms: int,
    ) -> Optional[AggregatedSignal]:
        """Handle absorption signal while watching a level."""
        nearest_level = self._nearest_entry_level(instrument, signal.direction, bias, current_price)
        if nearest_level is None:
            return None

        score = self._compute_composite_score(signal, nearest_level, bias)

        if score < self.min_composite_score:
            # Do NOT advance the phase on a rejected attempt -- the trade
            # stays WATCHING so a later qualifying absorption signal can
            # still be evaluated. Advancing here before this check used to
            # permanently wedge the state machine in ABSORPTION_DETECTED,
            # since process_signal has no routing branch to recover from it.
            return None

        trade.direction = signal.direction
        trade.qualified_level = nearest_level.price
        trade.advance_to_absorption(signal)

        sl, tp = self._compute_sl_tp(
            signal.direction, nearest_level, bias, current_price
        )

        # Advance to position with SL/TP
        trade.advance_to_position(current_price, sl, tp, entry_time_ms=now_ms)

        agg = AggregatedSignal(
            timestamp_ms=now_ms,
            direction=signal.direction,
            composite_score=score,
            qualified_level=nearest_level,
            signals=[signal],
            action="enter",
            suggested_sl=sl,
            suggested_tp=tp,
            notes=(
                f"ENTRY: Absorption confirmed at watched level "
                f"{nearest_level.price:.2f}. Attempts: {len(trade.absorption_signals)}"
            ),
        )
        return self._record_action(instrument, trade, agg)

    def _handle_initiative_for_be(
        self,
        instrument: str,
        signal: Signal,
        trade: TradeState,
        current_price: float,
        now_ms: int,
    ) -> Optional[AggregatedSignal]:
        """Initiative after entry → move to break even."""
        if signal.direction != trade.direction:
            return None

        if not self._protective_stop(trade.direction, trade.entry_price, current_price):
            return None

        trade.advance_to_break_even(signal)

        agg = AggregatedSignal(
            timestamp_ms=now_ms,
            direction=trade.direction,
            composite_score=signal.strength,
            signals=[signal],
            action="break_even",
            notes=(
                f"BREAK EVEN: Initiative auction confirmed. "
                f"Move SL to entry {trade.break_even_price:.2f}"
            ),
        )
        return self._record_action(instrument, trade, agg)

    def _handle_initiative_for_trail(
        self,
        instrument: str,
        signal: Signal,
        trade: TradeState,
        recent_bars: list[InfoBar],
        current_price: float,
        now_ms: int,
    ) -> Optional[AggregatedSignal]:
        """Subsequent initiative prints → trail stop."""
        if signal.direction != trade.direction:
            return None

        # Trail to the low of the initiative bar (for longs) or high (for shorts)
        if recent_bars:
            last_bar = recent_bars[-1]
            if trade.direction == Side.BUY:
                new_trail = last_bar.low
            else:
                new_trail = last_bar.high
        else:
            new_trail = signal.price_level

        # Validate the resulting stop, including the existing trailing bound.
        new_trail = max(trade.trail_stop, new_trail) if trade.direction == Side.BUY else min(trade.trail_stop, new_trail)
        tightens = new_trail > trade.trail_stop if trade.direction == Side.BUY else new_trail < trade.trail_stop
        if not tightens or not self._protective_stop(trade.direction, new_trail, current_price):
            return None

        trade.update_trail(new_trail, signal)

        agg = AggregatedSignal(
            timestamp_ms=now_ms,
            direction=trade.direction,
            composite_score=signal.strength,
            signals=[signal],
            action="trail",
            notes=(
                f"TRAIL: New initiative print. "
                f"Move SL to {trade.trail_stop:.2f}"
            ),
        )
        return self._record_action(instrument, trade, agg)

    def _handle_exit_warning(
        self,
        instrument: str,
        signal: Signal,
        trade: TradeState,
        current_price: float,
        now_ms: int,
    ) -> Optional[AggregatedSignal]:
        """Exhaustion or divergence → warning to exit/tighten."""
        # Only warn if signal is AGAINST current trade direction
        if signal.direction == trade.direction:
            return None  # Same direction exhaustion/divergence = less relevant

        action = "exit_warning"
        if signal.strength >= 70:
            action = "exit"
            # Auto-close trade on strong exit signal
            trade.close_trade(current_price, f"{signal.signal_type.value} exit (strength {signal.strength:.0f})", exit_time_ms=now_ms)

        agg = AggregatedSignal(
            timestamp_ms=now_ms,
            direction=trade.direction,
            composite_score=signal.strength,
            signals=[signal],
            action=action,
            notes=(
                f"{'EXIT' if action == 'exit' else 'WARNING'}: "
                f"{signal.signal_type.value} detected against position. "
                f"Strength: {signal.strength:.0f}"
            ),
        )
        return self._record_action(instrument, trade, agg)

    def _handle_sweep_at_level(
        self,
        instrument: str,
        signal: Signal,
        bias: Optional[DailyBias],
        trade: TradeState,
        current_price: float,
        now_ms: int,
    ) -> Optional[AggregatedSignal]:
        """Handle sweep signal while watching a level — alert only, adds context."""
        agg = AggregatedSignal(
            timestamp_ms=now_ms,
            direction=signal.direction,
            composite_score=signal.strength,
            signals=[signal],
            action="alert_only",
            notes=(
                f"SWEEP detected near watched level @ {trade.qualified_level:.2f}. "
                f"Strength: {signal.strength:.0f} — watch for absorption follow-up"
            ),
        )
        return self._record_action(instrument, trade, agg)

    def _compute_composite_score(
        self,
        signal: Signal,
        level: Optional[QualifiedLevel],
        bias: Optional[DailyBias],
    ) -> float:
        """
        Weighted composite score:
          Absorption: 30%
          Delta/Divergence: 25%
          VP context (level strength): 25%
          Initiative/Sweep: 20%
        """
        score = 0.0

        # Signal strength component (30-40% depending on type)
        if signal.signal_type == SignalType.ABSORPTION:
            score += signal.strength * 0.30
        elif signal.signal_type in (SignalType.DIVERGENCE,):
            score += signal.strength * 0.25
        elif signal.signal_type == SignalType.INITIATIVE:
            score += signal.strength * 0.20
        elif signal.signal_type == SignalType.SWEEP:
            score += signal.strength * 0.20
        else:
            score += signal.strength * 0.15

        # Volume profile context (25%)
        if level:
            score += level.strength * 0.25

        # Bias alignment (remaining %)
        if bias:
            bias_aligned = (
                (bias.direction == BiasDirection.LONG and signal.direction == Side.BUY)
                or (bias.direction == BiasDirection.SHORT and signal.direction == Side.SELL)
            )
            if bias_aligned:
                score += bias.confidence * 0.20
            elif bias.direction == BiasDirection.WARNING:
                score -= 10  # Penalty for trading against warning

        return min(100.0, max(0.0, score))

    def _compute_sl_tp(
        self,
        direction: Side,
        level: Optional[QualifiedLevel],
        bias: Optional[DailyBias],
        current_price: float,
    ) -> tuple[float, float]:
        """Compute suggested stop loss and take profit."""
        # Default: 0.3% SL, 0.6% TP
        if direction == Side.BUY:
            default = (current_price * 0.997, current_price * 1.006)
        else:
            default = (current_price * 1.003, current_price * 0.994)
        if bias is None:
            return default

        if direction == Side.BUY:
            # SL below VAL or absorption zone
            sl = bias.val - (bias.vah - bias.val) * 0.1
            # TP at VAH
            tp = bias.vah
            valid = sl < current_price < tp
        else:
            # SL above VAH
            sl = bias.vah + (bias.vah - bias.val) * 0.1
            # TP at VAL
            tp = bias.val
            valid = tp < current_price < sl

        # Profile levels on the wrong side of price (e.g. entry beyond VAH) give
        # a stop/target that is already hit; fall back to the percentage default.
        return (sl, tp) if valid else default

    def _nearest_level(
        self, levels: list[QualifiedLevel], direction: Side, price: float
    ) -> Optional[QualifiedLevel]:
        """Nearest direction-matching level strictly within the entry radius."""
        candidates = [level for level in levels
                      if level.direction == direction
                      and abs(price - level.price) / max(price, 1.0) < self.price_proximity_pct]
        return min(candidates, key=lambda level: abs(price - level.price), default=None)

    def _nearest_entry_level(
        self, instrument: str, direction: Side, bias: Optional[DailyBias], price: float
    ) -> Optional[QualifiedLevel]:
        nearest = self._nearest_level(self._watched_levels.get(instrument, []), direction, price)
        if nearest is None and bias is not None:
            nearest = self._nearest_level(bias.qualified_levels, direction, price)
        return nearest

    @staticmethod
    def _protective_stop(direction: Side, stop: float, price: float) -> bool:
        return stop < price if direction == Side.BUY else stop > price

    def _record_action(
        self, instrument: str, trade: TradeState, agg: AggregatedSignal, bar_seq: Optional[int] = None
    ) -> AggregatedSignal:
        if trade.trade_id == 0:
            trade.trade_id = self._next_trade_id
            self._next_trade_id += 1
        if agg.qualified_level is not None:
            level = agg.qualified_level
            self._execution_levels[instrument] = ExecutionLevel(
                price=level.price, direction=level.direction, level_type=level.level_type.value,
                strength=level.strength, source_dates=tuple(level.source_dates), notes=level.notes,
            )
        elif instrument not in self._execution_levels and trade.qualified_level:
            self._execution_levels[instrument] = ExecutionLevel(trade.qualified_level, trade.direction)
        if bar_seq is None:
            bar_seq = agg.signals[0].details.get("bar_seq", 0) if agg.signals else 0
        agg.bar_seq = bar_seq
        closed = agg.action in ("stop", "target", "exit")
        direction = agg.direction if agg.action == "alert_only" and trade.phase == TradePhase.WATCHING else trade.direction
        agg.execution = ExecutionSnapshot(
            trade_id=trade.trade_id, instrument=instrument, direction=direction,
            phase_after=trade.phase, entry_price=trade.entry_price,
            stop_loss=trade.stop_loss, take_profit=trade.take_profit,
            rr_ratio=trade.rr_ratio,
            fill_price=trade.exit_price if closed else trade.entry_price if agg.action == "enter" else None,
            exit_reason=agg.action if closed else None, pnl=trade.pnl_ticks if closed else None,
            qualified_level=self._execution_levels.get(instrument),
            bar_seq=bar_seq, timestamp_ms=agg.timestamp_ms,
        )
        if agg.action not in ("stop", "target"):
            self._last_signal_time[instrument] = agg.timestamp_ms
        self._signal_history.append(agg)
        return agg

    def check_exits(self, instrument: str, bar: InfoBar) -> Optional[AggregatedSignal]:
        """
        Close an open trade if the just-closed bar traded through its stop or target.

        Call this with each new bar BEFORE feeding that bar's pattern signals,
        so a trade opened at the close of bar N is only exposed from bar N+1
        (no same-bar fills). A bar only gives high/low, not their order, so if
        it spans both stop and target the stop is assumed to fill first
        (pessimistic; the intrabar sequence is unknown and is not guessed).
        """
        trade = self._active_trades.get(instrument)
        if trade is None or trade.phase not in (
            TradePhase.POSITION_OPEN, TradePhase.BREAK_EVEN, TradePhase.TRAILING
        ):
            return None

        long = trade.direction == Side.BUY
        stop_hit = bar.low <= trade.stop_loss if long else bar.high >= trade.stop_loss
        target_hit = bar.high >= trade.take_profit if long else bar.low <= trade.take_profit
        if not (stop_hit or target_hit):
            return None

        if stop_hit:
            action = "stop"
            price = min(trade.stop_loss, bar.open) if long else max(trade.stop_loss, bar.open)
        else:
            action, price = "target", trade.take_profit
        trade.close_trade(price, f"{action} hit", exit_time_ms=bar.timestamp_ms)
        agg = AggregatedSignal(
            timestamp_ms=bar.timestamp_ms,
            direction=trade.direction,        # the position that closed, not the closing side
            composite_score=0.0,
            action=action,
            notes=f"{'LONG' if long else 'SHORT'} {action.upper()} HIT @ {price:.2f} (PnL {trade.pnl_ticks:+.2f})",
        )
        return self._record_action(instrument, trade, agg, bar.seq)

    def get_active_trade(self, instrument: str) -> Optional[TradeState]:
        return self._active_trades.get(instrument)

    def close_trade(self, instrument: str, exit_price: float, reason: str = "", *, exit_time_ms: int):
        trade = self._active_trades.get(instrument)
        if trade:
            trade.close_trade(exit_price, reason, exit_time_ms=exit_time_ms)

    @property
    def signal_history(self) -> list[AggregatedSignal]:
        return self._signal_history

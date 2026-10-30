"""
Info-bar processing pipeline (synchronous, I/O free, deterministic).

    InfoBar -> delta -> footprint -> pattern detectors -> signal aggregator
                         \\-> UTC-day session profile -> profile framing (bias)

One pipeline instance handles one instrument + one bar type. It contains no
network, database or wall-clock access: every time-dependent component runs on
an `EventClock` driven by the bars' own close timestamps, so replaying the same
bars always produces the same signals and live/backtest share one code path.

No look-ahead:
  * a bar is processed only after it closed (its timestamp is its close);
  * a trade opened on bar N can only be stopped/targeted from bar N+1;
  * a session (UTC day of bar close) profile is finalised by the FIRST bar of the
    next session and only then feeds the bias used by that and later bars.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from orderflow_system.analytics.delta import DeltaEngine, DeltaResult
from orderflow_system.analytics.footprint import FootprintBar, FootprintEngine
from orderflow_system.analytics.volume_profile import VolumeProfileEngine
from orderflow_system.config.settings import InstrumentConfig
from orderflow_system.data.models import BarType, InfoBar, Signal, VolumeProfileResult
from orderflow_system.patterns.absorption import AbsorptionDetector
from orderflow_system.patterns.divergence import DivergenceDetector
from orderflow_system.patterns.exhaustion import ExhaustionDetector
from orderflow_system.patterns.initiative import InitiativeDetector
from orderflow_system.signals.aggregator import AggregatedSignal, SignalAggregator
from orderflow_system.signals.profile_framing import DailyBias, ProfileFramingEngine
from orderflow_system.utils.clock import EventClock

logger = logging.getLogger(__name__)

_DAY_MS = 86_400_000
_AUTO_WATCH_MIN_STRENGTH = 50


def utc_date(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


@dataclass
class BarResult:
    """Everything one bar produced, for the I/O layer (DB, alerts, dashboard)."""
    bar: InfoBar
    delta: DeltaResult
    footprint: FootprintBar
    signals: list[Signal] = field(default_factory=list)            # raw detector output
    aggregated: list[AggregatedSignal] = field(default_factory=list)  # entries/exits/BE/trail/stop/target
    closed_profile: Optional[VolumeProfileResult] = None           # set on the bar that opens a new session
    bias: Optional[DailyBias] = None                               # bias computed at that rollover
    gap_skipped: bool = False                                      # detector output discarded (outage)


class ReplayDigest:
    """Order-sensitive hash of everything a replay emitted — equal digests mean identical replays."""

    def __init__(self):
        self._h = hashlib.sha256()
        self.bars = 0

    def update(self, r: BarResult) -> None:
        self.bars += 1
        h = self._h
        h.update(f"B|{r.bar.timestamp_ms}|{r.bar.seq}|{r.delta.cumulative_delta:.9f}".encode())
        for s in r.signals:
            h.update(f"S|{s.signal_type.value}|{s.direction.value}|{s.price_level:.6f}|{s.strength:.9f}".encode())
        for a in r.aggregated:
            h.update(f"A|{a.timestamp_ms}|{a.action}|{a.direction.value}|{a.composite_score:.9f}"
                     f"|{a.suggested_sl:.6f}|{a.suggested_tp:.6f}".encode())
            h.update(repr(a.execution).encode())
        if r.closed_profile is not None:
            p = r.closed_profile
            h.update(f"P|{p.session_date}|{p.poc}|{p.vah}|{p.val}|{p.shape}|{p.total_volume:.9f}".encode())

    def hexdigest(self) -> str:
        return self._h.hexdigest()


class InfoBarPipeline:
    def __init__(
        self,
        config: InstrumentConfig,
        bar_type: Optional[BarType] = None,
        history_bars: int = 5000,
    ):
        self.config = config
        self.symbol = config.instrument.value
        self.bar_type = bar_type
        self.clock = EventClock()

        self.vp_engine = VolumeProfileEngine(config.volume_profile)
        self.delta_engine = DeltaEngine(tick_size=config.tick_size)
        self.footprint_engine = FootprintEngine(tick_size=config.tick_size)

        self.absorption = AbsorptionDetector(config.absorption, tick_size=config.tick_size, clock=self.clock)
        self.initiative = InitiativeDetector(config.initiative, tick_size=config.tick_size)
        self.exhaustion = ExhaustionDetector(config.exhaustion)
        self.divergence = DivergenceDetector(config.divergence)
        self.detectors = [self.absorption, self.initiative, self.exhaustion, self.divergence]

        self.profile_framing = ProfileFramingEngine(config.volume_profile)
        self.aggregator = SignalAggregator(
            min_composite_score=config.risk.min_composite_score,
            signal_cooldown_seconds=config.risk.signal_cooldown_seconds,
            clock=self.clock,
        )

        self.bars: deque[InfoBar] = deque(maxlen=history_bars)
        # Chart x-value per retained bar (see _display_second): fixed when the bar
        # arrives, so a bar's value never depends on which window is requested.
        self._display: deque[int] = deque(maxlen=history_bars)
        self._last_display = 0
        self._recent: deque[InfoBar] = deque(maxlen=10)
        self._session_day: Optional[int] = None
        self._session_date: str = ""
        self._session_bars: list[InfoBar] = []
        self._last_ts: Optional[int] = None

        self.bars_processed = 0
        self.gap_bars_skipped = 0
        self.signal_counts: Counter = Counter()
        self.action_counts: Counter = Counter()
        self.sessions_closed = 0

    # ── main entry ──────────────────────────────────────────

    def process_bar(self, bar: InfoBar) -> BarResult:
        if self._last_ts is not None and bar.timestamp_ms < self._last_ts:
            raise ValueError(
                f"Out-of-order bar: {bar.timestamp_ms} < {self._last_ts}. "
                f"Bars must be processed chronologically."
            )
        self._last_ts = bar.timestamp_ms
        self.clock.set(bar.timestamp_ms)

        closed_profile, bias = self._roll_session(bar)

        # Exits first: a trade opened on this bar's close must not fill on this bar.
        aggregated: list[AggregatedSignal] = []
        exit_agg = self.aggregator.check_exits(self.symbol, bar)
        if exit_agg:
            aggregated.append(exit_agg)

        delta = self.delta_engine.compute_from_bar(bar)
        footprint = self.footprint_engine.build_from_bar(bar)
        self.bars.append(bar)
        self._display.append(self._display_second(bar))
        self._recent.append(bar)
        self._session_bars.append(bar)
        self.bars_processed += 1

        result = BarResult(
            bar=bar, delta=delta, footprint=footprint, aggregated=aggregated,
            closed_profile=closed_profile, bias=bias,
        )

        # Every detector sees every bar, keeping price/delta/rate histories aligned.
        signals = self._detect(bar, delta, footprint)
        if bar.data_gap:
            # Gap rates are unknown and any emitted patterns are discarded.
            self.gap_bars_skipped += 1
            result.gap_skipped = True
            # UI histories must expose only accepted patterns as well.
            for detector in self.detectors:
                history = detector._signal_history
                if history and any(history[-1] is signal for signal in signals):
                    history.pop()
        else:
            for sig in signals:
                sig.details["bar_seq"] = bar.seq   # which bar produced it (timestamps can tie)
            result.signals = signals
            recent5 = list(self._recent)[-5:]
            for sig in signals:
                self.signal_counts[sig.signal_type.value] += 1
                agg = self.aggregator.process_signal(
                    instrument=self.symbol,
                    signal=sig,
                    bias=self.profile_framing.current_bias,
                    current_price=bar.close,
                    recent_bars=recent5,
                )
                if agg:
                    aggregated.append(agg)

        # Applies to every aggregate of this bar, including a stop/target on a gap bar.
        for agg in aggregated:
            agg.bar_seq = bar.seq
            self.action_counts[agg.action] += 1
        return result

    # ── internals ───────────────────────────────────────────

    def _display_second(self, bar: InfoBar) -> int:
        """
        Strictly increasing chart x-value (seconds): max(close second, previous + 1).
        Several bars can close in the same second, the chart needs unique x.
        DISPLAY ONLY — never used by analytics; exact time is `timestamp_ms`.
        """
        t = max(bar.timestamp_ms // 1000, self._last_display + 1)
        self._last_display = t
        return t

    def _detect(self, bar: InfoBar, delta: DeltaResult, footprint: FootprintBar) -> list[Signal]:
        signals: list[Signal] = []
        recent = list(self._recent)
        for sig in (
            self.absorption.check_bar(bar, footprint, delta, bar.close),
            self.initiative.check_bar(bar, delta, footprint),
            self.exhaustion.check_bar(bar, delta, self.delta_engine, footprint, recent),
            self.divergence.check_bar(bar, self.delta_engine),
        ):
            if sig:
                signals.append(sig)
        return signals

    def _roll_session(self, bar: InfoBar) -> tuple[Optional[VolumeProfileResult], Optional[DailyBias]]:
        """Finalise the previous UTC-day profile when a bar closes in a new day."""
        day = bar.timestamp_ms // _DAY_MS
        if self._session_day is None:
            self._session_day, self._session_date = day, utc_date(bar.timestamp_ms)
            return None, None
        if day == self._session_day:
            return None, None

        profile = self.vp_engine.compute_from_bars(self._session_bars, session_date=self._session_date)
        self._session_bars = []
        self._session_day, self._session_date = day, utc_date(bar.timestamp_ms)
        if profile.total_volume <= 0:
            return None, None

        self.profile_framing.add_profile(profile)
        last_close = self._recent[-1].close if self._recent else bar.open
        bias = self.profile_framing.analyze(last_close)
        self.sessions_closed += 1

        # New session -> new levels. Open positions are untouched.
        self.aggregator.reset_watching(self.symbol)
        for level in bias.qualified_levels:
            if level.strength >= _AUTO_WATCH_MIN_STRENGTH:
                self.aggregator.set_watching(self.symbol, level, level.direction)
        return profile, bias

    # ── read-only views (dashboard / alerts) ────────────────

    @property
    def current_price(self) -> float:
        return self.bars[-1].close if self.bars else 0.0

    @property
    def session_date(self) -> str:
        return self._session_date

    def current_session_profile(self) -> VolumeProfileResult:
        """Profile of the session in progress (built from bars already closed)."""
        return self.vp_engine.compute_from_bars(self._session_bars, session_date=self._session_date)

    def get_recent_bars(self, n: int) -> list[InfoBar]:
        return list(self.bars)[-n:]

    def get_recent_display_times(self, n: int) -> list[int]:
        """Chart x-values aligned 1:1 with get_recent_bars(n)."""
        return list(self._display)[-n:]

    @property
    def stats(self) -> dict:
        last = self.bars[-1] if self.bars else None
        return {
            "symbol": self.symbol,
            "bar_type": self.bar_type.value if self.bar_type else None,
            "price": self.current_price,
            "bars": self.bars_processed,
            "last_bar_ms": last.timestamp_ms if last else None,
            "session": self._session_date,
            "sessions_closed": self.sessions_closed,
            "cum_delta": self.delta_engine.cumulative_delta,
            "signals": dict(self.signal_counts),
            "actions": dict(self.action_counts),
            "gap_bars_skipped": self.gap_bars_skipped,
        }

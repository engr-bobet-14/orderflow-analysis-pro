"""
Injectable event clock.

`RealClock` preserves existing live-trading behavior (wall-clock time) and
is every constructor's default, so live semantics are unchanged. Historical
replay / backtest callers inject an `EventClock` instead and advance it to
each candle's own timestamp as they step through history, so cooldowns and
staleness windows are computed against market time rather than however
fast the replay happens to execute.
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now_ms(self) -> int: ...


class RealClock:
    """Wall-clock time. Default for every live code path."""

    def now_ms(self) -> int:
        return int(time.time() * 1000)


class EventClock:
    """Reports whichever timestamp was last set via `.set()`. Historical/
    backtest callers set this to the current candle's `timestamp_ms` as
    they replay."""

    def __init__(self, now_ms: int = 0):
        self._now_ms = now_ms

    def now_ms(self) -> int:
        return self._now_ms

    def set(self, now_ms: int) -> None:
        self._now_ms = now_ms

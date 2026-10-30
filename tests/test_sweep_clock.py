"""Historical L2 snapshots use event time throughout sweep and tracker queries."""

import pytest

from orderflow_system.analytics.footprint import FootprintBar
from orderflow_system.analytics.orderbook import OrderbookTracker
from orderflow_system.config.settings import SweepConfig
from orderflow_system.data.models import InfoBar, OrderbookLevel, OrderbookSnapshot, Side
from orderflow_system.patterns.sweep import SweepDetector
from orderflow_system.utils.clock import EventClock, RealClock


HISTORICAL_MS = 1_700_000_000_000


def consume_levels(tracker, timestamp_ms, side="ask"):
    bids = [OrderbookLevel(99 - i, 2) for i in range(3)]
    asks = [OrderbookLevel(101 + i, 2) for i in range(3)]
    tracker.update(OrderbookSnapshot(timestamp_ms - 1, bids, asks))
    tracker.update(OrderbookSnapshot(timestamp_ms,
                                     bids if side == "ask" else [],
                                     asks if side == "bid" else []))


def sweep_bar(timestamp_ms):
    return InfoBar(timestamp_ms, 100, 104, 96, 102, volume=10)


@pytest.mark.parametrize("side", ["ask", "bid"])
def test_tracker_queries_and_cutoff_follow_its_injected_event_clock(side):
    clock = EventClock(HISTORICAL_MS)
    tracker = OrderbookTracker(clock=clock)
    consume_levels(tracker, HISTORICAL_MS, side)
    clock.set(HISTORICAL_MS + 1000)
    assert len(tracker.get_recent_consumptions(1000, side)) == 3
    assert tracker.count_swept_levels(1000, side) == 3
    assert tracker.total_consumed_volume(1000, side) == 6
    clock.set(HISTORICAL_MS + 1001)
    assert tracker.get_recent_consumptions(1000, side) == []
    assert tracker.count_swept_levels(1000, side) == 0
    assert tracker.total_consumed_volume(1000, side) == 0


@pytest.mark.parametrize("side,direction", [("ask", Side.BUY), ("bid", Side.SELL)])
def test_sweep_passes_event_clock_through_external_tracker_and_uses_event_cooldown(
    monkeypatch, side, direction,
):
    # A default-clock tracker deliberately proves the detector passes its own clock
    # through both count/volume queries, rather than relying on constructor agreement.
    tracker = OrderbookTracker()
    clock = EventClock(HISTORICAL_MS)
    detector = SweepDetector(SweepConfig(), clock=clock)

    def no_wall_clock(self):
        raise AssertionError("Historical sweep must not query RealClock")

    monkeypatch.setattr(RealClock, "now_ms", no_wall_clock)
    consume_levels(tracker, HISTORICAL_MS, side)
    first = detector.check(tracker, sweep_bar(clock.now_ms()), FootprintBar())
    assert first is not None and first.direction == direction
    assert first.timestamp_ms == HISTORICAL_MS
    assert first.details["levels_swept"] == 3
    assert first.details["total_volume_consumed"] == 6
    assert detector._last_signal_ms == HISTORICAL_MS

    clock.set(HISTORICAL_MS + 1000)
    consume_levels(tracker, clock.now_ms(), side)
    assert detector.check(tracker, sweep_bar(clock.now_ms()), FootprintBar()) is None
    assert len(detector._signal_history) == 1

    # Fresh historical snapshots at the exact cooldown boundary permit another signal.
    clock.set(HISTORICAL_MS + 5000)
    consume_levels(tracker, clock.now_ms(), side)
    second = detector.check(tracker, sweep_bar(clock.now_ms()), FootprintBar())
    assert second is not None and second.direction == direction
    assert second.timestamp_ms == clock.now_ms()
    assert second.details["total_volume_consumed"] == 6
    assert len(detector._signal_history) == 2

    # Even with an elapsed cooldown, aged consumptions must no longer trigger.
    clock.set(HISTORICAL_MS + 10_000)
    assert detector.check(tracker, sweep_bar(clock.now_ms()), FootprintBar()) is None


def test_live_constructors_keep_real_clock_defaults():
    assert isinstance(OrderbookTracker()._clock, RealClock)
    assert isinstance(SweepDetector(SweepConfig())._clock, RealClock)

"""
Integration tests — engines, detectors, database, working on synthetic data.
Run: pytest orderflow_system/test_integration.py
"""

import os
import sys

import pytest

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orderflow_system.analytics.delta import DeltaEngine, DeltaResult
from orderflow_system.analytics.footprint import FootprintBar, FootprintEngine
from orderflow_system.analytics.volume_profile import VolumeProfileEngine
from orderflow_system.config.settings import get_btcusd_config
from orderflow_system.data.database import Database
from orderflow_system.data.models import (
    BarType, InfoBar, Side, Signal, SignalType, Tick, VolumeProfileResult,
)
from orderflow_system.patterns.absorption import AbsorptionDetector
from orderflow_system.patterns.initiative import InitiativeDetector
from orderflow_system.signals.profile_framing import ProfileFramingEngine


def make_bar(ts=1_000_000, o=100.0, h=101.0, l=99.0, c=100.5, buy=60.0, sell=40.0,
             dur=10_000, seq=1, **kw) -> InfoBar:
    return InfoBar(
        timestamp_ms=ts, open=o, high=h, low=l, close=c, volume=buy + sell,
        buy_volume=buy, sell_volume=sell, tick_count=100,
        dollar_value=(buy + sell) * (h + l) / 2, seq=seq,
        bar_type=BarType.DOLLAR_IMBALANCE, duration_ms=dur, **kw,
    )


def test_volume_profile_from_ticks():
    """VP from real trade prices: POC at the heaviest price."""
    engine = VolumeProfileEngine(get_btcusd_config().volume_profile)
    ticks = []
    for i in range(100):
        ticks.append(Tick(1000 + i, 18500.0, 10.0, Side.BUY))
    for i in range(50):
        ticks.append(Tick(2000 + i, 18490.0, 5.0, Side.SELL))
        ticks.append(Tick(3000 + i, 18510.0, 5.0, Side.BUY))
    for i in range(5):
        ticks.append(Tick(4000 + i, 18470.0, 1.0, Side.SELL))
        ticks.append(Tick(5000 + i, 18530.0, 1.0, Side.BUY))

    vp = engine.compute_from_ticks(ticks, session_date="2026-02-12")
    assert vp.poc == 18500.0
    assert vp.val <= vp.poc <= vp.vah
    assert vp.source == "ticks"


def test_volume_profile_from_bars_uses_vwap_not_range():
    """Bar volume lands at the bar's VWAP only; it is never spread over high-low."""
    engine = VolumeProfileEngine(get_btcusd_config().volume_profile)
    wide = make_bar(o=100, h=200, l=50, c=120, buy=70, sell=30)   # huge range
    wide.dollar_value = 100 * 150.0                               # vwap = 150
    vp = engine.compute_from_bars([wide], session_date="d")
    assert vp.source == "bar_vwap"
    assert list(vp.volume_at_price) == [150.0]
    assert vp.volume_at_price[150.0] == 100.0


def test_delta_engine_bar():
    engine = DeltaEngine(tick_size=0.01)
    r1 = engine.compute_from_bar(make_bar(buy=140, sell=60))
    r2 = engine.compute_from_bar(make_bar(buy=10, sell=30))
    assert r1.vertical_delta == 80
    assert r2.cumulative_delta == 60
    assert r1.horizontal_delta == {}                 # no price-level data -> unknown, not zeros
    assert r1.max_delta_price is None and r1.min_delta_price is None


def test_delta_engine_ticks_keeps_horizontal_delta():
    engine = DeltaEngine(tick_size=1.0)
    r = engine.compute_from_ticks([Tick(1, 100.0, 5, Side.BUY), Tick(2, 101.0, 3, Side.SELL)])
    assert r.horizontal_delta == {100.0: 5, 101.0: -3}
    assert r.max_delta_price == 100.0 and r.min_delta_price == 101.0


def test_footprint_of_info_bar_has_no_levels():
    engine = FootprintEngine()
    fp = engine.build_from_bar(make_bar(buy=70, sell=30))
    assert not fp.has_price_levels
    assert fp.levels == {}
    assert fp.total_buy_volume == 70 and fp.total_sell_volume == 30 and fp.delta == 40
    assert fp.imbalance_levels() == []
    assert engine.get_aggressive_volume_at_level(100.0) is None   # unknown, not (0, 0)
    assert engine.count_consecutive_imbalances("buy") is None


def test_footprint_from_ticks_has_levels():
    engine = FootprintEngine(tick_size=1.0)
    fp = engine.build_from_ticks([Tick(1, 100.0, 5, Side.BUY), Tick(2, 100.0, 2, Side.SELL)])
    assert fp.has_price_levels
    assert fp.levels[100.0].ask_volume == 5 and fp.levels[100.0].bid_volume == 2


def test_absorption_delta_close_mismatch_bearish():
    """Strong buy delta but the bar closes down -> buyers absorbed -> SELL."""
    detector = AbsorptionDetector(get_btcusd_config().absorption)
    bar = make_bar(o=100.0, h=100.5, l=99.0, c=99.5, buy=80, sell=20)   # -50 bps body, delta +60%
    delta = DeltaEngine().compute_from_bar(bar)
    signal = detector.check_bar(bar, FootprintEngine().build_from_bar(bar), delta, bar.close)
    assert signal is not None
    assert signal.direction == Side.SELL
    assert signal.price_level == bar.high


def test_absorption_needs_real_adverse_move():
    """A doji after strong delta is not absorption (no meaningful opposite close)."""
    detector = AbsorptionDetector(get_btcusd_config().absorption)
    bar = make_bar(o=100.0, h=100.5, l=99.5, c=99.9995, buy=80, sell=20)   # 0.5 bps down < 1 bps threshold
    delta = DeltaEngine().compute_from_bar(bar)
    assert detector.check_bar(bar, FootprintEngine().build_from_bar(bar), delta, bar.close) is None


def test_absorption_level_method_inert_without_price_levels():
    detector = AbsorptionDetector(get_btcusd_config().absorption)
    bar = make_bar(buy=51, sell=49)                      # weak delta -> method 1 off
    delta = DeltaEngine().compute_from_bar(bar)
    assert detector.check_bar(bar, FootprintEngine().build_from_bar(bar), delta, bar.close) is None
    assert detector.active_absorptions == {}


def test_initiative_detection_uses_participation_rate():
    detector = InitiativeDetector(get_btcusd_config().initiative)
    engine = DeltaEngine()
    fp = FootprintBar()
    for i in range(10):   # baseline: slow bars (60 s)
        b = make_bar(ts=i * 60_000, o=100 + i, h=101 + i, l=99 + i, c=100.4 + i,
                     buy=50, sell=50, dur=60_000, seq=i)
        detector.check_bar(b, engine.compute_from_bar(b), fp)

    fast = make_bar(ts=700_000, o=110.0, h=112.0, l=109.9, c=111.9, buy=170, sell=30,
                    dur=5_000, seq=99)   # same volume class, 12x faster, strong delta, up bar
    signal = detector.check_bar(fast, engine.compute_from_bar(fast), fp)
    assert signal is not None
    assert signal.direction == Side.BUY
    assert signal.details["imbalance_levels"] is None    # unknown without price levels


def test_initiative_rejects_misaligned_delta():
    detector = InitiativeDetector(get_btcusd_config().initiative)
    engine = DeltaEngine()
    for i in range(5):
        b = make_bar(ts=i * 60_000, dur=60_000, seq=i)
        detector.check_bar(b, engine.compute_from_bar(b), FootprintBar())
    down = make_bar(ts=400_000, o=112.0, h=112.1, l=109.0, c=109.2, buy=170, sell=30, dur=5_000, seq=50)
    assert detector.check_bar(down, engine.compute_from_bar(down), FootprintBar()) is None


def test_profile_framing():
    """Daily bias from profile shapes."""
    engine = ProfileFramingEngine()
    vp1 = VolumeProfileResult(
        session_date="2026-02-10", poc=18550.0, vah=18560.0, val=18520.0,
        total_volume=10000, shape="p_shape", poc_position_pct=0.75,
        volume_at_price={18520.0: 500, 18530.0: 800, 18540.0: 1200, 18550.0: 3000, 18560.0: 2500},
    )
    engine.add_profile(vp1)
    assert engine.analyze(18555.0).direction.value == "long"

    vp2 = VolumeProfileResult(
        session_date="2026-02-11", poc=18580.0, vah=18600.0, val=18550.0,
        total_volume=12000, shape="p_shape", poc_position_pct=0.70,
        volume_at_price={18550.0: 600, 18560.0: 1000, 18570.0: 1500, 18580.0: 4000,
                         18590.0: 3000, 18600.0: 2000},
    )
    engine.add_profile(vp2)
    bias2 = engine.analyze(18585.0)
    assert bias2.qualified_levels


@pytest.mark.asyncio
async def test_database_ticks_and_info_bars(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    await db.connect()

    await db.insert_ticks_batch("BTCUSDT", [
        Tick(1000000, 18500.0, 10.0, Side.BUY, "t1"),
        Tick(1000001, 18500.5, 5.0, Side.SELL, "t2"),
    ])
    assert len(await db.get_ticks("BTCUSDT", 999999, 1000002)) == 2

    bars = [make_bar(ts=1000 + i, seq=i + 1) for i in range(3)]
    await db.insert_bars_batch("BTCUSDT", BarType.DOLLAR_IMBALANCE, bars)
    await db.insert_bars_batch("BTCUSDT", BarType.DOLLAR_IMBALANCE, bars)   # idempotent
    got = await db.get_bars("BTCUSDT", BarType.DOLLAR_IMBALANCE, 0, 10_000)
    assert [b.seq for b in got] == [1, 2, 3]
    assert got[0].dollar_value == bars[0].dollar_value
    assert await db.get_bars("BTCUSDT", BarType.TICK_RUN, 0, 10_000) == []
    assert [b.duration_ms for b in got] == [b.duration_ms for b in bars]     # round-trips for replay parity
    gapped = make_bar(ts=5000, seq=9, data_gap=True, dur=None)
    await db.insert_bars_batch("BTCUSDT", BarType.DOLLAR_IMBALANCE, [gapped])
    back = (await db.get_bars("BTCUSDT", BarType.DOLLAR_IMBALANCE, 4000, 6000))[0]
    assert back.data_gap is True and back.duration_ms is None and back.volume_rate is None

    sig = Signal(timestamp_ms=1, signal_type=SignalType.ABSORPTION, direction=Side.BUY, price_level=1.0)
    for _ in range(2):                                                       # re-running a replay
        await db.insert_signal("BTCUSDT", sig, bar_type=BarType.DOLLAR_IMBALANCE)
    cur = await db._db.execute("SELECT COUNT(*) FROM signals")
    assert (await cur.fetchone())[0] == 1

    vp = VolumeProfileResult(session_date="2026-01-01", poc=1, vah=2, val=0, total_volume=3,
                             volume_at_price={1.0: 3.0}, source="bar_vwap")
    await db.insert_volume_profile("BTCUSDT", vp, bar_type=BarType.DOLLAR_IMBALANCE)
    back = await db.get_volume_profiles("BTCUSDT", bar_type=BarType.DOLLAR_IMBALANCE)
    await db.insert_volume_profile("BTCUSDT", vp, bar_type=BarType.DOLLAR_IMBALANCE)   # same session again
    back = await db.get_volume_profiles("BTCUSDT", bar_type=BarType.DOLLAR_IMBALANCE)
    assert len(back) == 1 and back[0].source == "bar_vwap"
    assert await db.get_volume_profiles("BTCUSDT", bar_type=BarType.TICK_RUN) == []
    await db.close()

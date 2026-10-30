"""Pipeline guarantees: chronological, deterministic, no look-ahead, honest about missing data."""

import hashlib
import random

import pytest

from orderflow_system.config.settings import get_btcusd_config
from orderflow_system.data.models import BarType, InfoBar, Side, Signal, SignalType, TradePhase
from orderflow_system.pipeline import InfoBarPipeline, utc_date
from orderflow_system.signals.aggregator import SignalAggregator

DAY = 86_400_000
T0 = 1_672_531_200_000   # 2023-01-01T00:00:00Z


def synth_bars(n=4000, seed=7, step_ms=45_000, price0=20000.0):
    """Reproducible random bars with directional pressure and varying speed."""
    rnd = random.Random(seed)
    bars, price, ts, seq = [], price0, T0, 0
    for i in range(n):
        o = price
        drift = rnd.gauss(0, 12)
        c = max(1.0, o + drift)
        h, l = max(o, c) + abs(rnd.gauss(0, 4)), min(o, c) - abs(rnd.gauss(0, 4))
        vol = rnd.uniform(5, 40)
        pressure = 0.5 + max(-0.45, min(0.45, drift / 40 + rnd.gauss(0, 0.2)))
        buy = vol * pressure
        dur = int(step_ms * rnd.uniform(0.1, 3.0))
        ts += dur
        seq += rnd.randint(50, 500)
        bars.append(InfoBar(
            timestamp_ms=ts, open=o, high=h, low=l, close=c, volume=vol, buy_volume=buy,
            sell_volume=vol - buy, tick_count=100, dollar_value=vol * (h + l) / 2,
            threshold=1.0, seq=seq, bar_type=BarType.DOLLAR_IMBALANCE,
            duration_ms=None if i == 0 else dur,
        ))
        price = c
    return bars


def run(bars, **kw):
    p = InfoBarPipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE, **kw)
    results = [p.process_bar(b) for b in bars]
    return p, results


def fingerprint(results):
    h = hashlib.sha256()
    for r in results:
        for s in r.signals:
            h.update(f"S|{s.timestamp_ms}|{s.signal_type.value}|{s.direction.value}|{s.price_level:.6f}|{s.strength:.6f}".encode())
        for a in r.aggregated:
            h.update(f"A|{a.timestamp_ms}|{a.action}|{a.composite_score:.6f}|{a.suggested_sl:.6f}|{a.suggested_tp:.6f}".encode())
        if r.closed_profile:
            p = r.closed_profile
            h.update(f"P|{p.session_date}|{p.poc}|{p.vah}|{p.val}|{p.shape}".encode())
    return h.hexdigest()


def test_synthetic_stream_actually_exercises_the_pipeline():
    p, results = run(synth_bars(6000, seed=3))
    assert p.stats["signals"], "no detector fired; tests below would be vacuous"
    assert p.sessions_closed >= 1
    assert any(r.aggregated for r in results)


def test_replay_is_deterministic():
    bars = synth_bars()
    _, a = run(bars)
    _, b = run(bars)
    assert fingerprint(a) == fingerprint(b)


def test_no_lookahead_prefix_invariance():
    """Results for the first N bars must not depend on what comes after them."""
    bars = synth_bars(3000)
    _, full = run(bars)
    for cut in (500, 1234, 2500):
        _, prefix = run(bars[:cut])
        assert fingerprint(prefix) == fingerprint(full[:cut]), f"output at bar<{cut} changed with later data"


def test_future_bars_cannot_change_past_signals_even_when_wild():
    bars = synth_bars(1500)
    wild = bars[:1000] + [
        InfoBar(timestamp_ms=b.timestamp_ms, open=1, high=10**6, low=1, close=10**6,
                volume=b.volume, buy_volume=b.volume, sell_volume=0, tick_count=1,
                dollar_value=b.volume * 10**6, seq=b.seq, duration_ms=b.duration_ms)
        for b in bars[1000:]
    ]
    _, a = run(bars)
    _, b = run(wild)
    assert fingerprint(a[:1000]) == fingerprint(b[:1000])


def test_out_of_order_bar_raises():
    p = InfoBarPipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE)
    bars = synth_bars(5)
    p.process_bar(bars[2])
    with pytest.raises(ValueError, match="Out-of-order"):
        p.process_bar(bars[1])


def test_same_millisecond_bars_are_accepted():
    p = InfoBarPipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE)
    a, b = synth_bars(2)
    b.timestamp_ms = a.timestamp_ms
    p.process_bar(a)
    p.process_bar(b)
    assert p.bars_processed == 2


def test_session_profile_closes_only_when_first_bar_of_next_utc_day_arrives():
    bars = synth_bars(3000, step_ms=60_000)
    p = InfoBarPipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE)
    closed_at = []
    for i, b in enumerate(bars):
        r = p.process_bar(b)
        if r.closed_profile:
            closed_at.append((i, r.closed_profile))
    assert closed_at
    for i, vp in closed_at:
        prev_day = utc_date(bars[i - 1].timestamp_ms)
        assert vp.session_date == prev_day                      # the day that just ended
        assert utc_date(bars[i].timestamp_ms) != prev_day       # triggered by the new day's first bar
        in_day = sum(b.volume for b in bars[:i] if utc_date(b.timestamp_ms) == prev_day)
        assert vp.total_volume == pytest.approx(in_day)         # nothing from the new day leaked in
        assert vp.source == "bar_vwap"


def test_bar_closing_just_after_midnight_belongs_to_new_session():
    p = InfoBarPipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE)
    b1, b2, b3 = synth_bars(3)
    b1.timestamp_ms, b2.timestamp_ms, b3.timestamp_ms = T0 + DAY - 2000, T0 + DAY - 1, T0 + DAY + 1
    p.process_bar(b1); p.process_bar(b2)
    r = p.process_bar(b3)
    assert r.closed_profile is not None and r.closed_profile.session_date == "2023-01-01"
    assert r.closed_profile.total_volume == pytest.approx(b1.volume + b2.volume)


def test_gap_bars_update_state_but_skip_detectors():
    bars = synth_bars(400)
    p, base = run(bars)
    gapped = []
    for i, b in enumerate(bars):
        g = InfoBar(timestamp_ms=b.timestamp_ms, open=b.open, high=b.high, low=b.low, close=b.close,
                    volume=b.volume, buy_volume=b.buy_volume, sell_volume=b.sell_volume,
                    tick_count=b.tick_count, dollar_value=b.dollar_value, threshold=b.threshold,
                    seq=b.seq, bar_type=b.bar_type, duration_ms=b.duration_ms, data_gap=(i % 10 == 0))
        gapped.append(g)
    pg, res = run(gapped)
    assert pg.gap_bars_skipped == 40
    assert all(not r.signals for r in res if r.gap_skipped)
    assert pg.bars_processed == 400
    assert pg.delta_engine.cumulative_delta == pytest.approx(p.delta_engine.cumulative_delta)


def test_info_bar_derived_values_are_honest():
    b = synth_bars(2)[1]
    assert b.vwap == pytest.approx(b.dollar_value / b.volume)
    assert b.delta_pct == pytest.approx((b.buy_volume - b.sell_volume) / b.volume)
    assert b.volume_rate == pytest.approx(b.volume / max(b.duration_ms / 1000, 1.0))
    first = synth_bars(1)[0]
    assert first.duration_ms is None and first.volume_rate is None     # unknown stays unknown


def test_first_bar_does_not_poison_participation_baseline():
    """Unknown rate (first bar) must not seed the initiative baseline or count as a rate."""
    p = InfoBarPipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE)
    first = synth_bars(1)[0]
    p.process_bar(first)
    assert p.initiative._avg_rate_window == []
    assert p.delta_engine.history[0].volume_rate is None
    assert p.delta_engine.get_volume_trend(lookback=1) == 0.0


def test_stop_on_gap_bar_is_still_attributed_and_counted():
    from orderflow_system.data.models import TradeState
    p = InfoBarPipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE)
    bars = synth_bars(3)
    p.process_bar(bars[0])
    t = TradeState(instrument="BTCUSDT", direction=Side.BUY)
    t.advance_to_position(bars[0].close, bars[0].close - 1, bars[0].close + 1e6, entry_time_ms=bars[0].timestamp_ms)
    p.aggregator._active_trades["BTCUSDT"] = t
    gap = bars[1]
    gap.low = bars[0].close - 100
    gap.data_gap = True
    r = p.process_bar(gap)
    assert r.gap_skipped and r.signals == []
    assert [a.action for a in r.aggregated] == ["stop"]
    assert r.aggregated[0].bar_seq == gap.seq and p.stats["actions"]["stop"] == 1
    assert r.aggregated[0].direction == Side.BUY                 # the position that closed


def test_display_time_is_window_independent_and_strictly_increasing():
    bars = synth_bars(300)
    for i in (50, 51, 52):
        bars[i].timestamp_ms = bars[49].timestamp_ms          # burst in one ms
    p, _ = run(bars)
    full = p.get_recent_display_times(300)
    assert all(b > a for a, b in zip(full, full[1:]))
    assert p.get_recent_display_times(40) == full[-40:]       # same bar -> same x in any window
    assert all(t >= b.timestamp_ms // 1000 for t, b in zip(full, p.get_recent_bars(300)))


def test_exhaustion_strength_is_scale_free():
    """Same shape at 1000x the volume must give the same strength (bar types differ in absolute size)."""
    def run_scaled(k):
        p = InfoBarPipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE)
        out = []
        for b in synth_bars(3000, seed=11):
            b = InfoBar(timestamp_ms=b.timestamp_ms, open=b.open, high=b.high, low=b.low, close=b.close,
                        volume=b.volume * k, buy_volume=b.buy_volume * k, sell_volume=b.sell_volume * k,
                        tick_count=b.tick_count, dollar_value=b.dollar_value * k, seq=b.seq,
                        bar_type=b.bar_type, duration_ms=b.duration_ms)
            out += [round(s.strength, 6) for s in p.process_bar(b).signals if s.signal_type == SignalType.EXHAUSTION]
        return out
    a, b = run_scaled(1), run_scaled(1000)
    assert a and a == b


def test_pipeline_never_builds_price_level_data():
    p, results = run(synth_bars(600))
    assert all(not r.footprint.has_price_levels and r.delta.horizontal_delta == {} for r in results)


# ── aggregator exit logic ──

def _open_long(agg, price=100.0, sl=99.0, tp=102.0, ts=1_000):
    from orderflow_system.data.models import TradeState
    t = TradeState(instrument="X", direction=Side.BUY)
    t.advance_to_position(price, sl, tp, entry_time_ms=ts)
    agg._active_trades["X"] = t
    return t


def _bar(ts, o, h, l, c):
    return InfoBar(timestamp_ms=ts, open=o, high=h, low=l, close=c, volume=1, buy_volume=.5, sell_volume=.5, seq=ts)


def test_stop_hit_closes_trade_at_stop():
    agg = SignalAggregator()
    t = _open_long(agg)
    out = agg.check_exits("X", _bar(2000, 100, 100.5, 98.5, 99.0))
    assert out.action == "stop" and t.phase == TradePhase.CLOSED and t.pnl_ticks == pytest.approx(-1.0)


def test_target_hit_closes_trade_at_target():
    agg = SignalAggregator()
    t = _open_long(agg)
    out = agg.check_exits("X", _bar(2000, 100, 102.5, 99.5, 102.0))
    assert out.action == "target" and t.pnl_ticks == pytest.approx(2.0)


def test_bar_spanning_stop_and_target_assumes_stop_first():
    agg = SignalAggregator()
    t = _open_long(agg)
    out = agg.check_exits("X", _bar(2000, 100, 103, 98, 100))
    assert out.action == "stop" and t.pnl_ticks < 0


def test_inside_bar_changes_nothing():
    agg = SignalAggregator()
    t = _open_long(agg)
    assert agg.check_exits("X", _bar(2000, 100, 101, 99.5, 100.5)) is None
    assert t.phase == TradePhase.POSITION_OPEN


def test_trade_opened_on_a_bar_is_not_exited_by_that_same_bar():
    """Pipeline runs exits BEFORE new entries, so an entry at bar N's close is exposed from N+1."""
    _, results = run(synth_bars(6000, seed=3))
    entries = 0
    for r in results:
        actions = [a.action for a in r.aggregated]
        if "enter" in actions:
            entries += 1
            assert not {"stop", "target"} & set(actions[actions.index("enter"):])
    assert entries > 0


def test_wrong_side_profile_levels_fall_back_to_percentage_stops():
    from orderflow_system.signals.profile_framing import DailyBias
    agg = SignalAggregator()
    bias = DailyBias(date="d", poc=100, vah=100, val=90)       # price (120) is above VAH
    sl, tp = agg._compute_sl_tp(Side.BUY, None, bias, 120.0)
    assert sl < 120.0 < tp                                     # a usable long, not an already-hit target


def test_flagged_high_stays_in_divergence_history_and_aligns_delta(monkeypatch):
    """An 11-bar window cannot drop an outage high while retaining its delta."""
    bars = synth_bars(11)
    for bar in bars:
        bar.open, bar.close, bar.low, bar.high = 100, 100, 99, 101
    bars[1].high, bars[1].data_gap = 120, True
    bars[-1].high = 110
    p = InfoBarPipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE)
    monkeypatch.setattr(p.delta_engine, "detect_delta_peaks", lambda lookback: ([(2, 100), (7, 0)], []))
    for bar in bars:
        result = p.process_bar(bar)
    assert len(p.divergence._price_history) == len(p.delta_engine.history) == 11
    assert p.divergence._price_history[-10][1] == 120
    assert all(s.signal_type != SignalType.DIVERGENCE for s in result.signals)


def test_post_gap_exhaustion_waits_until_unknown_rate_leaves_window(monkeypatch):
    bars = synth_bars(12)
    bars[4].data_gap = True
    p = InfoBarPipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE)
    checked = []
    def bullish(window, current, *args):
        checked.append(current.seq)
        return None
    monkeypatch.setattr(p.exhaustion, "_check_bullish_exhaustion", bullish)
    for bar in bars:
        p.process_bar(bar)
    n = p.config.exhaustion.min_bars_declining
    assert bars[4].volume_rate is None
    assert all(bar.seq not in checked for bar in bars[4:4+n+1])
    assert bars[4+n+1].seq in checked


def test_gap_signals_are_discarded_but_all_detector_histories_advance(monkeypatch):
    p = InfoBarPipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE)
    seen = []
    for detector in p.detectors:
        def check(*args, detector=detector):
            seen.append(detector)
            sig = Signal(args[0].timestamp_ms, SignalType.DIVERGENCE, Side.SELL, args[0].high)
            detector._signal_history.append(sig)
            return sig
        monkeypatch.setattr(detector, "check_bar", check)
    bar = synth_bars(2)[1]
    bar.data_gap = True
    result = p.process_bar(bar)
    assert len(seen) == 4 and result.gap_skipped and not result.signals
    assert all(not d._signal_history for d in p.detectors)
    assert not result.aggregated and p.signal_counts == {}


def test_shared_ledger_audit_checks_snapshots_and_rejects_invalid_level_direction():
    from dataclasses import replace
    from orderflow_system.validate_infobars import LedgerAudit
    pipe, results = run(synth_bars(6000, seed=3))
    audit = LedgerAudit(pipe.aggregator.price_proximity_pct)
    for result in results:
        audit.process_result(result)
    assert audit.ledger.entries and audit.ledger.closes
    assert audit.report()["ok"], audit.report()
    result = next(result for result in results if any(a.action == "enter" for a in result.aggregated))
    aggregate = next(a for a in result.aggregated if a.action == "enter")
    opposite = Side.SELL if aggregate.execution.direction == Side.BUY else Side.BUY
    aggregate.execution = replace(aggregate.execution,
                                  qualified_level=replace(aggregate.execution.qualified_level, direction=opposite))
    invalid = LedgerAudit(pipe.aggregator.price_proximity_pct)
    invalid.process_result(result)
    assert invalid.report()["violations"]["entry_level_direction"] == 1


@pytest.mark.parametrize("direction,action,prices", [
    (Side.BUY, "stop", (98, 98.5, 97, 98)),
    (Side.SELL, "stop", (102, 103, 101.5, 102)),
    (Side.BUY, "target", (103, 104, 102.5, 103.5)),
    (Side.SELL, "target", (97, 97.5, 96, 97)),
])
def test_ledger_audit_keeps_protective_gap_fills_and_counts_gap_targets(direction, action, prices):
    from orderflow_system.data.models import TradeState
    from orderflow_system.validate_infobars import LedgerAudit
    pipe = InfoBarPipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE)
    pipe.process_bar(_bar(1000, 100, 101, 99.5, 100))
    trade = TradeState(instrument=pipe.symbol, direction=direction)
    sl, tp = (99, 102) if direction == Side.BUY else (101, 98)
    trade.advance_to_position(100, sl, tp, entry_time_ms=1000)
    pipe.aggregator._active_trades[pipe.symbol] = trade
    gap = _bar(2000, *prices)
    gap.data_gap = True
    result = pipe.process_bar(gap)
    assert [aggregate.action for aggregate in result.aggregated] == [action]
    audit = LedgerAudit(pipe.aggregator.price_proximity_pct)
    audit.process_result(result)
    report = audit.report()
    assert report["ok"], report
    assert report["counters"]["protective_gap_exits"] == 1
    assert report["counters"]["target_gap_through"] == (action == "target")

"""Trade actions must use the selected direction and attainable execution prices."""

from dataclasses import FrozenInstanceError

import pytest

from orderflow_system.data.models import InfoBar, Side, Signal, SignalType, TradePhase, TradeState
from orderflow_system.signals.aggregator import SignalAggregator
from orderflow_system.signals.profile_framing import DailyBias, LevelType, QualifiedLevel
from orderflow_system.utils.clock import EventClock


def bar(seq=2, o=100, h=100.5, l=99.5, c=100):
    return InfoBar(timestamp_ms=100_000 + seq, seq=seq, open=o, high=h, low=l, close=c)


def signal(kind=SignalType.ABSORPTION, side=Side.BUY, price=100, seq=2, strength=100):
    return Signal(100_000 + seq, kind, side, price, strength, {"bar_seq": seq})


def aggregator():
    return SignalAggregator(min_composite_score=0, signal_cooldown_seconds=0, clock=EventClock(100_002))


def level(side=Side.BUY, price=100):
    return QualifiedLevel(price, LevelType.VAL if side == Side.BUY else LevelType.VAH, side, strength=100)


def opened(agg, side=Side.BUY, sl=None, tp=None):
    trade = TradeState("X", side)
    trade.advance_to_position(100, sl if sl is not None else (99 if side == Side.BUY else 101),
                              tp if tp is not None else (102 if side == Side.BUY else 98), entry_time_ms=100_001)
    agg._active_trades["X"] = trade
    return trade


def test_absorption_chooses_matching_watched_direction_before_entering():
    agg = aggregator()
    agg.set_watching("X", level(Side.SELL, 100.05), Side.SELL)
    agg.set_watching("X", level(Side.BUY, 100), Side.BUY)
    out = agg.process_signal("X", signal(), None, 100, [bar()])
    trade = agg.get_active_trade("X")
    assert out.action == "enter"
    assert trade.direction == out.direction == out.qualified_level.direction == Side.BUY
    assert trade.qualified_level == out.qualified_level.price == 100
    assert trade.stop_loss < trade.entry_price < trade.take_profit
    assert out.execution.direction == out.execution.qualified_level.direction == Side.BUY


@pytest.mark.parametrize("watching", [False, True])
def test_distant_absorption_is_rejected_on_every_entry_route(watching):
    agg = aggregator()
    lv = level(price=95)
    if watching:
        agg.set_watching("X", lv, Side.BUY)
    assert agg.process_signal("X", signal(), DailyBias("d", qualified_levels=[lv]), 100, [bar()]) is None
    trade = agg.get_active_trade("X")
    assert trade is None or trade.phase == TradePhase.WATCHING


def test_nearest_entry_search_filters_direction_before_distance():
    agg = aggregator()
    bias = DailyBias("d", qualified_levels=[level(Side.SELL, 100), level(Side.BUY, 100.05)])
    out = agg.process_signal("X", signal(), bias, 100, [bar()])
    assert out.action == "enter" and out.qualified_level.price == 100.05


def test_watched_fallback_uses_a_nearby_matching_bias_level():
    agg = aggregator()
    agg.set_watching("X", level(price=90), Side.BUY)
    out = agg.process_signal("X", signal(), DailyBias("d", qualified_levels=[level(price=100.05)]), 100, [bar()])
    assert out.action == "enter" and out.qualified_level.price == 100.05


@pytest.mark.parametrize("side,opposite,close,extreme,pnl", [
    (Side.BUY, Side.SELL, 100.25, 102, .25),
    (Side.SELL, Side.BUY, 99.75, 98, .25),
])
def test_pattern_exit_fills_at_actionable_close(side, opposite, close, extreme, pnl):
    agg = aggregator()
    trade = opened(agg, side)
    sig = signal(SignalType.EXHAUSTION, opposite, extreme)
    out = agg.process_signal("X", sig, None, close, [bar(c=close)])
    assert out.action == "exit" and out.direction == side
    assert trade.exit_price == out.execution.fill_price == close
    assert trade.pnl_ticks == out.execution.pnl == pytest.approx(pnl)
    assert trade.exit_time_ms == out.execution.timestamp_ms == 100_002
    assert sig.price_level == extreme


@pytest.mark.parametrize("side,ohlc,expected", [
    (Side.BUY, (98, 98.5, 97, 98.2), 98),
    (Side.SELL, (102, 103, 101.5, 102.2), 102),
])
def test_opening_gap_stop_uses_open(side, ohlc, expected):
    agg = aggregator()
    trade = opened(agg, side)
    out = agg.check_exits("X", bar(o=ohlc[0], h=ohlc[1], l=ohlc[2], c=ohlc[3]))
    assert out.action == "stop" and out.execution.fill_price == expected
    assert trade.exit_price == expected and trade.exit_time_ms == 100_002
    assert out.execution.pnl == -2


@pytest.mark.parametrize("side,ohlc,target", [
    (Side.BUY, (103, 104, 102.5, 103.5), 102),
    (Side.SELL, (97, 97.5, 96, 96.5), 98),
])
def test_opening_gap_target_keeps_resting_limit_price(side, ohlc, target):
    agg = aggregator()
    opened(agg, side)
    out = agg.check_exits("X", bar(o=ohlc[0], h=ohlc[1], l=ohlc[2], c=ohlc[3]))
    assert out.action == "target" and out.execution.fill_price == target
    assert out.execution.pnl == 2


@pytest.mark.parametrize("side,ohlc,valid_close", [
    (Side.BUY, (99.75, 99.9, 99.7, 99.8), 100.1),
    (Side.SELL, (100.25, 100.3, 100.1, 100.2), 99.9),
])
def test_break_even_is_deferred_until_entry_is_protective(side, ohlc, valid_close):
    agg = aggregator()
    trade = opened(agg, side, sl=99.7 if side == Side.BUY else 100.3)
    initial_stop = trade.stop_loss
    initiative = signal(SignalType.INITIATIVE, side)
    b = bar(o=ohlc[0], h=ohlc[1], l=ohlc[2], c=ohlc[3])
    assert agg.process_signal("X", initiative, None, b.close, [b]) is None
    assert trade.phase == TradePhase.POSITION_OPEN and trade.stop_loss == initial_stop
    out = agg.process_signal("X", initiative, None, valid_close, [bar(c=valid_close)])
    assert out.action == "break_even" and out.execution.fill_price is None
    assert trade.phase == TradePhase.BREAK_EVEN and trade.stop_loss == 100


@pytest.mark.parametrize("side,price,extreme", [(Side.BUY, 100.1, 100.2), (Side.SELL, 99.9, 99.8)])
def test_trail_skips_a_stop_on_the_wrong_side_of_price(side, price, extreme):
    agg = aggregator()
    trade = opened(agg, side)
    initiative = signal(SignalType.INITIATIVE, side, price=extreme)
    trade.advance_to_break_even(initiative)
    assert agg.process_signal("X", initiative, None, price, []) is None
    assert trade.phase == TradePhase.BREAK_EVEN and trade.stop_loss == 100


def test_execution_snapshot_is_deeply_immutable_and_survives_replacement():
    agg = aggregator()
    lv = level()
    entry = agg.process_signal("X", signal(seq=1), DailyBias("d", qualified_levels=[lv]), 100, [bar(seq=1)])
    close = agg.check_exits("X", bar(seq=2, l=99))
    replacement = agg.process_signal("X", signal(seq=2), DailyBias("d", qualified_levels=[lv]), 100, [bar()])
    assert close.execution.trade_id == entry.execution.trade_id
    assert replacement.execution.trade_id > close.execution.trade_id
    assert close.execution.phase_after == TradePhase.CLOSED
    assert replacement.execution.phase_after == TradePhase.POSITION_OPEN
    lv.price = 200
    lv.source_dates.append("future")
    assert entry.execution.qualified_level.price == close.execution.qualified_level.price == 100
    assert entry.execution.qualified_level.source_dates == ()
    with pytest.raises(FrozenInstanceError):
        close.execution.fill_price = 200
    with pytest.raises(FrozenInstanceError):
        close.execution.qualified_level.price = 200


@pytest.mark.parametrize("side,extreme,close", [(Side.BUY, 99.9, 100.5), (Side.SELL, 100.1, 99.5)])
def test_trail_no_op_does_not_advance_phase_or_record_action(side, extreme, close):
    agg = aggregator()
    trade = opened(agg, side)
    initiative = signal(SignalType.INITIATIVE, side, price=extreme)
    trade.advance_to_break_even(initiative)
    before_signals = list(trade.initiative_signals)
    before_history = list(agg.signal_history)
    initiative_bar = bar(h=100.6 if side == Side.BUY else extreme,
                         l=extreme if side == Side.BUY else 99.4, c=close)
    assert agg.process_signal("X", initiative, None, close, [initiative_bar]) is None
    assert trade.phase == TradePhase.BREAK_EVEN
    assert trade.trail_stop == trade.stop_loss == 100
    assert trade.initiative_signals == before_signals
    assert list(agg.signal_history) == before_history
    assert "X" not in agg._last_signal_time

    # A later strictly tighter stop is accepted, keeping the original bracket R:R.
    initiative.price_level = 100.1 if side == Side.BUY else 99.9
    update = agg.process_signal("X", initiative, None, close, [])
    assert update.action == "trail" and trade.phase == TradePhase.TRAILING
    assert trade.stop_loss == initiative.price_level
    assert update.execution.rr_ratio == 2
    assert agg.process_signal("X", initiative, None, close, []) is None
    assert len(agg.signal_history) == 1


def test_trade_transitions_require_explicit_event_timestamps():
    trade = TradeState("X", Side.BUY)
    with pytest.raises(TypeError, match="entry_time_ms"):
        trade.advance_to_position(100, 99, 102)
    assert trade.phase == TradePhase.WATCHING
    trade.advance_to_position(100, 99, 102, entry_time_ms=123)
    with pytest.raises(TypeError, match="exit_time_ms"):
        trade.close_trade(101, "test")
    assert trade.phase == TradePhase.POSITION_OPEN
    agg = aggregator()
    agg._active_trades["X"] = trade
    with pytest.raises(TypeError, match="exit_time_ms"):
        agg.close_trade("X", 101, "test")
    assert trade.phase == TradePhase.POSITION_OPEN
    agg.close_trade("X", 101, "test", exit_time_ms=456)
    assert trade.entry_time_ms == 123 and trade.exit_time_ms == 456
    assert trade.exit_price == 101 and trade.pnl_ticks == 1


def _timed_absorption(agg, clock, bias, side, seq):
    sig = Signal(clock.now_ms(), SignalType.ABSORPTION, side, 100, 100, {"bar_seq": seq})
    return agg.process_signal("X", sig, bias, 100, [])


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
@pytest.mark.parametrize("action", ["stop", "target"])
def test_protective_close_does_not_restart_signal_cooldown(side, action):
    clock = EventClock(100_000)
    agg = SignalAggregator(min_composite_score=0, signal_cooldown_seconds=30, clock=clock)
    bias = DailyBias("d", qualified_levels=[level(side)])
    assert _timed_absorption(agg, clock, bias, side, 1).action == "enter"
    assert agg._last_signal_time["X"] == 100_000
    clock.set(160_000)
    ohlc = {
        (Side.BUY, "stop"): (100, 100.1, 99.6, 99.8),
        (Side.SELL, "stop"): (100, 100.4, 99.9, 100.2),
        (Side.BUY, "target"): (100, 100.8, 99.9, 100.7),
        (Side.SELL, "target"): (100, 100.1, 99.2, 99.3),
    }[side, action]
    close_bar = InfoBar(clock.now_ms(), *ohlc, seq=2)
    assert agg.check_exits("X", close_bar).action == action
    assert agg._last_signal_time["X"] == 100_000
    clock.set(161_000)
    replacement = _timed_absorption(agg, clock, bias, side, 3)
    assert replacement.action == "enter"
    assert agg._last_signal_time["X"] == 161_000


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
@pytest.mark.parametrize("strength,action", [(100, "exit"), (60, "exit_warning")])
def test_pattern_exit_actions_keep_signal_cooldown(side, strength, action):
    clock = EventClock(100_000)
    agg = SignalAggregator(min_composite_score=0, signal_cooldown_seconds=30, clock=clock)
    bias = DailyBias("d", qualified_levels=[level(side)])
    _timed_absorption(agg, clock, bias, side, 1)
    clock.set(160_000)
    opposite = Side.SELL if side == Side.BUY else Side.BUY
    sig = Signal(clock.now_ms(), SignalType.DIVERGENCE, opposite, 102, strength, {"bar_seq": 2})
    close_price = 100.2 if side == Side.BUY else 99.8
    assert agg.process_signal("X", sig, bias, close_price, []).action == action
    assert agg._last_signal_time["X"] == 160_000
    clock.set(161_000)
    if action == "exit":
        assert _timed_absorption(agg, clock, bias, side, 3) is None
        clock.set(190_000)
        assert _timed_absorption(agg, clock, bias, side, 4).action == "enter"
    else:
        initiative = Signal(clock.now_ms(), SignalType.INITIATIVE, side, close_price, 100, {"bar_seq": 3})
        assert agg.process_signal("X", initiative, bias, close_price, []) is None
        clock.set(190_000)
        assert agg.process_signal("X", initiative, bias, close_price, []).action == "break_even"


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_snapshots_keep_original_bracket_rr_through_management_and_close(monkeypatch, side):
    agg = aggregator()
    sl, tp = (99, 102) if side == Side.BUY else (101, 98)
    monkeypatch.setattr(agg, "_compute_sl_tp", lambda *_: (sl, tp))
    entry = agg.process_signal("X", signal(side=side), DailyBias("d", qualified_levels=[level(side)]), 100, [])
    assert entry.execution.entry_price == 100
    assert entry.execution.stop_loss == sl and entry.execution.take_profit == tp
    close_price, trail = (101, 100.5) if side == Side.BUY else (99, 99.5)
    initiative = signal(SignalType.INITIATIVE, side, price=trail)
    be = agg.process_signal("X", initiative, None, close_price, [])
    trailing = agg.process_signal("X", initiative, None, close_price, [])
    opposite = Side.SELL if side == Side.BUY else Side.BUY
    exit_signal = signal(SignalType.DIVERGENCE, opposite)
    closed = agg.process_signal("X", exit_signal, None, close_price, [])
    assert [a.action for a in (entry, be, trailing, closed)] == ["enter", "break_even", "trail", "exit"]
    assert all(a.execution.rr_ratio == 2 for a in (entry, be, trailing, closed))
    assert agg.get_active_trade("X").rr_ratio == 2


@pytest.mark.parametrize("phase", [TradePhase.WATCHING, TradePhase.CLOSED])
def test_reset_watching_removes_non_open_trade_and_cached_level(phase):
    agg = aggregator()
    lv = level()
    agg.set_watching("X", lv, Side.BUY)
    # An alert captures the tentative level before the reset.
    agg.process_signal("X", signal(SignalType.SWEEP), None, 100, [])
    trade = agg.get_active_trade("X")
    if phase == TradePhase.CLOSED:
        trade.advance_to_position(100, 99, 102, entry_time_ms=100_001)
        trade.close_trade(101, "test", exit_time_ms=100_002)
    assert "X" in agg._watched_levels and "X" in agg._execution_levels
    history = list(agg.signal_history)
    agg.reset_watching("X")
    assert agg.get_active_trade("X") is None
    assert "X" not in agg._watched_levels and "X" not in agg._execution_levels
    assert list(agg.signal_history) == history
    # Repeated reset remains harmless when a rollover produces no new levels.
    agg.reset_watching("X")
    assert agg.get_active_trade("X") is None


@pytest.mark.parametrize("phase", [TradePhase.POSITION_OPEN, TradePhase.BREAK_EVEN, TradePhase.TRAILING])
def test_reset_watching_preserves_open_position_and_execution_level(phase):
    from dataclasses import asdict

    agg = aggregator()
    lv = level()
    entry = agg.process_signal("X", signal(), DailyBias("d", qualified_levels=[lv]), 100, [])
    trade = agg.get_active_trade("X")
    trade.phase = phase
    agg._watched_levels["X"] = [lv]
    before = asdict(trade)
    cached_level = agg._execution_levels["X"]
    agg.reset_watching("X")
    assert agg.get_active_trade("X") is trade and asdict(trade) == before
    assert agg._execution_levels["X"] is cached_level is entry.execution.qualified_level
    assert "X" not in agg._watched_levels


@pytest.mark.parametrize("watched_side,sweep_side", [(Side.BUY, Side.SELL), (Side.SELL, Side.BUY)])
def test_opposing_sweep_alert_snapshot_uses_signal_direction_without_entering(watched_side, sweep_side):
    from dataclasses import asdict
    from types import SimpleNamespace
    from orderflow_system.validate_infobars import LedgerAudit

    agg = aggregator()
    agg.set_watching("X", level(watched_side), watched_side)
    trade = agg.get_active_trade("X")
    before = asdict(trade)
    sweep = signal(SignalType.SWEEP, sweep_side)
    current_bar = bar()
    alert = agg.process_signal("X", sweep, None, current_bar.close, [current_bar])
    assert alert.action == "alert_only"
    assert alert.direction == alert.execution.direction == sweep.direction == sweep_side
    assert alert.execution.fill_price is alert.execution.pnl is alert.execution.exit_reason is None
    assert alert.execution.phase_after == TradePhase.WATCHING
    assert trade.phase == TradePhase.WATCHING and trade.direction == watched_side
    # Allocating the action's trade identity must not alter position state.
    after = asdict(trade)
    assert after.pop("trade_id") > 0
    before.pop("trade_id")
    assert after == before
    audit = LedgerAudit(agg.price_proximity_pct)
    audit.process_result(SimpleNamespace(bar=current_bar, signals=[sweep], aggregated=[alert]))
    assert audit.report()["ok"], audit.report()
    assert audit.report()["counters"]["aggregates"] == 1
    assert audit.ledger.records == []

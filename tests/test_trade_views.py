"""Current trade views and captured action broadcasts share one stable schema."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from orderflow_system.config.settings import InfoBarConfig, get_btcusd_config
from orderflow_system.dashboard import app as dash
from orderflow_system.data.models import InfoBar, Side, Signal, SignalType, TradePhase
from orderflow_system.main import OrderflowSystem
from orderflow_system.signals.profile_framing import DailyBias, LevelType, QualifiedLevel
from test_fanout import CapturingManager

DAY = 86_400_000
T0 = 1_672_531_200_000


def bar(seq, timestamp, o=100, h=100.1, l=99.9, c=100):
    return InfoBar(timestamp, o, h, l, c, volume=10, buy_volume=5,
                   sell_volume=5, dollar_value=c * 10, seq=seq, duration_ms=1000)


@pytest.fixture
def view_system(monkeypatch):
    config = get_btcusd_config()
    config.risk = replace(config.risk, min_composite_score=0, signal_cooldown_seconds=0)
    system = OrderflowSystem([config], InfoBarConfig(bars_per_second=1000))
    system.ws_manager = CapturingManager()
    system.reader = SimpleNamespace(first_predecessor_ms=None)
    pipeline = system.pipelines[system.symbol]
    monkeypatch.setattr(pipeline, "_detect", lambda *_: [])
    dash.set_system(system)
    try:
        yield TestClient(dash.app), system, pipeline
    finally:
        dash.set_system(None)


def assert_current_views(client, system, phase, ratio):
    rest = client.get(f"/api/trade/{system.symbol}").json()
    strategy = client.get(f"/api/strategy-status/{system.symbol}").json()["trade"]
    assert strategy == rest
    assert rest["phase"] == rest["phase_after"] == phase
    assert rest["rr_ratio"] == ratio
    with client.websocket_connect("/ws") as ws:
        initial = [ws.receive_json() for _ in range(3)]
        state = next(message["data"] for message in initial if message["channel"] == "trade_state")
        assert state == rest
    return rest


@pytest.mark.parametrize("previous_close", [False, True], ids=["no-entry", "after-close"])
def test_rollover_views_show_current_watching_and_reset(view_system, monkeypatch, previous_close):
    client, system, pipeline = view_system
    if previous_close:
        level = QualifiedLevel(100, LevelType.VAL, Side.BUY, 100)
        pipeline.profile_framing._bias_history.append(DailyBias("d", qualified_levels=[level]))
        monkeypatch.setattr(pipeline, "_detect", lambda b, *_: [
            Signal(b.timestamp_ms, SignalType.ABSORPTION, Side.BUY, 100, 100)
        ] if b.seq == 1 else [])
    pipeline.process_bar(bar(1, T0 + DAY - 10_000))
    if previous_close:
        result = pipeline.process_bar(bar(2, T0 + DAY - 5000, l=99.5))
        assert result.aggregated[0].execution.phase_after == TradePhase.CLOSED
    result = pipeline.process_bar(bar(3, T0 + DAY + 5000))
    assert not result.aggregated
    current = assert_current_views(client, system, "watching", 0)
    assert current["entry_price"] == 0
    assert current["qualified_level"] == 100
    assert client.get(f"/api/strategy-status/{system.symbol}").json()["overall"] == "WATCHING"
    # Resetting WATCHING clears the current view; old CLOSED actions cannot reappear.
    pipeline.aggregator.reset_watching(system.symbol)
    reset = assert_current_views(client, system, "none", None)
    assert reset.keys() == current.keys()


@pytest.mark.parametrize("level_mode", ["empty", "below-watch-strength"])
def test_closed_trade_rollover_without_watchable_levels_clears_all_current_views(
    view_system, monkeypatch, level_mode,
):
    client, system, pipeline = view_system
    level = QualifiedLevel(100, LevelType.VAL, Side.BUY, 100)
    pipeline.profile_framing._bias_history.append(DailyBias("d", qualified_levels=[level]))
    monkeypatch.setattr(pipeline, "_detect", lambda b, *_: [
        Signal(b.timestamp_ms, SignalType.ABSORPTION, Side.BUY, 100, 100)
    ] if b.seq == 1 else [])
    pipeline.process_bar(bar(1, T0 + DAY - 10_000))
    closed = pipeline.process_bar(bar(2, T0 + DAY - 5000, l=99.5))
    old_action, = closed.aggregated
    old_snapshot = old_action.execution
    assert old_action.action == "stop" and old_snapshot.phase_after == TradePhase.CLOSED
    old_data = asdict(old_snapshot)
    assert_current_views(client, system, "closed", old_snapshot.rr_ratio)

    analyze = pipeline.profile_framing.analyze

    def no_watchable_levels(*args, **kwargs):
        bias = analyze(*args, **kwargs)
        if level_mode == "empty":
            bias.qualified_levels = []
        else:
            for qualified in bias.qualified_levels:
                qualified.strength = 49
        return bias

    monkeypatch.setattr(pipeline.profile_framing, "analyze", no_watchable_levels)
    rollover = pipeline.process_bar(bar(3, T0 + DAY + 5000))
    assert rollover.closed_profile is not None and pipeline.sessions_closed == 1
    assert rollover.bias is pipeline.profile_framing.current_bias
    assert all(qualified.strength < 50 for qualified in rollover.bias.qualified_levels)
    assert not rollover.aggregated
    current = assert_current_views(client, system, "none", None)
    instruments = client.get("/api/instruments").json()
    assert instruments[0]["symbol"] == system.symbol
    assert instruments[0]["trade_phase"] == "none"
    with client.websocket_connect("/ws") as ws:
        initial = [ws.receive_json() for _ in range(3)]
        stats = next(message["data"] for message in initial if message["channel"] == "stats")
        trade = next(message["data"] for message in initial if message["channel"] == "trade_state")
        assert stats["trade_phase"] == "none"
        assert trade == current
    assert pipeline.aggregator.get_active_trade(system.symbol) is None
    assert pipeline.aggregator.signal_history[-1] is old_action
    assert old_action.execution is old_snapshot
    assert asdict(old_snapshot) == old_data


@pytest.mark.asyncio
async def test_original_bracket_ratio_survives_every_view_and_broadcast(view_system, monkeypatch):
    client, system, pipeline = view_system
    level = QualifiedLevel(100, LevelType.VAL, Side.BUY, 100)
    pipeline.profile_framing._bias_history.append(DailyBias("d", qualified_levels=[level]))
    monkeypatch.setattr(pipeline.aggregator, "_compute_sl_tp", lambda *_: (99, 102))
    kinds = {1: (SignalType.ABSORPTION, Side.BUY), 2: (SignalType.INITIATIVE, Side.BUY),
             3: (SignalType.INITIATIVE, Side.BUY), 4: (SignalType.DIVERGENCE, Side.SELL)}
    monkeypatch.setattr(pipeline, "_detect", lambda b, *_: [
        Signal(b.timestamp_ms, *kinds[b.seq], b.close, 100)
    ])
    bars = [bar(1, T0 + 1000), bar(2, T0 + 2000, o=100.2, h=100.7, l=100.1, c=100.5),
            bar(3, T0 + 3000, o=100.5, h=101, l=100.25, c=100.8),
            bar(4, T0 + 4000, o=100.8, h=101, l=100.5, c=100.75)]
    for current_bar, phase in zip(bars, ("position_open", "break_even", "trailing", "closed")):
        await system._on_bar(current_bar)
        current = assert_current_views(client, system, phase, 2)
        state = [m["data"] for m in system.ws_manager.messages if m["channel"] == "trade_state"][-1]
        assert state.keys() == current.keys()
        assert state["phase"] == phase and state["rr_ratio"] == 2
        assert state["rr_ratio"] == pipeline.aggregator.signal_history[-1].execution.rr_ratio
    states = [m["data"] for m in system.ws_manager.messages if m["channel"] == "trade_state"]
    assert [state["stop_loss"] for state in states] == [99, 100, 100.25, 100.25]
    assert [state["rr_ratio"] for state in states] == [2, 2, 2, 2]

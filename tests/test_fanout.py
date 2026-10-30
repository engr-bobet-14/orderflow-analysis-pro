"""Replay fan-out and Telegram formatting consume immutable execution snapshots."""

import json
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from conftest import make_rows, write_dataset
from orderflow_system.alerts.telegram_bot import TelegramAlertBot
from orderflow_system.config.settings import InfoBarConfig, get_btcusd_config
from orderflow_system.dashboard.websocket_manager import WebSocketManager, _serialize
from orderflow_system.dashboard.trade_serializer import serialize_trade
from orderflow_system.data.models import InfoBar, Side, Signal, SignalType, TradePhase
from orderflow_system.ledger import TradeLedger
from orderflow_system.main import OrderflowSystem
from orderflow_system.signals.aggregator import SignalAggregator
from orderflow_system.signals.profile_framing import DailyBias, LevelType, QualifiedLevel
from orderflow_system.utils.clock import EventClock


class CapturingManager(WebSocketManager):
    def __init__(self):
        super().__init__()
        self.messages = []

    async def broadcast(self, channel, data, symbol=""):
        self.messages.append(json.loads(json.dumps({
            "channel": channel.value, "symbol": symbol, "data": _serialize(data),
        })))


@pytest.mark.asyncio
async def test_paced_replay_broadcasts_actions_and_old_trade_before_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr("orderflow_system.main.DASHBOARD.enabled", False)
    rows = make_rows(6, t0="2023-01-01T23:59:55", step_s=1, price0=100)
    ohlc = [(100, 100.1, 99.9, 100), (100, 100.2, 99.6, 100),
            (100, 100.1, 99.2, 99.5), (100, 100.1, 99.9, 100),
            (100, 100.4, 99.9, 100.2), (100.2, 100.3, 100.1, 100.2)]
    for seq, (row, (o, h, l, c)) in enumerate(zip(rows, ohlc), 1):
        row.update(seq=seq, nt=1, o=o, h=h, l=l, c=c, dv=row["vol"] * c)
    write_dataset(tmp_path, rows)
    config = get_btcusd_config()
    config.risk = replace(config.risk, min_composite_score=0, signal_cooldown_seconds=0)
    system = OrderflowSystem([config], InfoBarConfig(root=str(tmp_path), bars_per_second=100_000))
    system.telegram.initialize = AsyncMock()
    manager = CapturingManager()
    system.ws_manager = manager
    pipeline = system.pipelines[system.symbol]
    levels = [QualifiedLevel(100, LevelType.VAL, Side.BUY, 100),
              QualifiedLevel(100, LevelType.VAH, Side.SELL, 100)]
    pipeline.profile_framing._bias_history.append(DailyBias("d", qualified_levels=levels))
    emitted = {1: (SignalType.ABSORPTION, Side.BUY), 2: (SignalType.ABSORPTION, Side.SELL),
               4: (SignalType.ABSORPTION, Side.BUY), 5: (SignalType.DIVERGENCE, Side.SELL)}

    def detect(bar, *_):
        if bar.seq not in emitted:
            return []
        kind, side = emitted[bar.seq]
        return [Signal(bar.timestamp_ms, kind, side, bar.high, 100)]

    monkeypatch.setattr(pipeline, "_detect", detect)
    ledger = TradeLedger()
    process_bar = pipeline.process_bar

    def record(bar):
        result = process_bar(bar)
        ledger.process_result(result)
        return result

    monkeypatch.setattr(pipeline, "process_bar", record)
    await system.start()
    assert system.replay_finished and pipeline.bars_processed == 6
    assert {m["channel"] for m in manager.messages} == {"signal", "trade_state", "bias", "volume_profile", "stats"}
    actions = [m["data"] for m in manager.messages if m["channel"] == "signal"]
    assert [a["action"] for a in actions] == ["enter", "stop", "enter", "target", "enter", "exit"]
    stop, replacement = actions[1:3]
    assert stop["bar_seq"] == replacement["bar_seq"] == 2
    assert stop["direction"] == stop["execution"]["direction"] == "buy"
    assert replacement["direction"] == replacement["execution"]["direction"] == "sell"
    assert stop["execution"]["trade_id"] != replacement["execution"]["trade_id"]
    assert stop["execution"]["phase_after"] == "closed"
    assert replacement["execution"]["phase_after"] == "position_open"
    assert actions[3]["direction"] == "sell" and actions[3]["execution"]["fill_price"] == 99.4
    states = [m["data"] for m in manager.messages if m["channel"] == "trade_state"]
    assert states[1] == serialize_trade(ledger.by_seq[2][0].execution)
    assert states[2] == serialize_trade(ledger.by_seq[2][1].execution)
    from fastapi.testclient import TestClient
    from orderflow_system.dashboard import app as dash
    dash.set_system(system)
    try:
        current = TestClient(dash.app).get(f"/api/trade/{system.symbol}").json()
        assert all(state.keys() == current.keys() for state in states)
        assert states[1]["phase"] == "closed" and states[2]["phase"] == "position_open"
    finally:
        dash.set_system(None)
    assert ledger.by_seq[2][0].execution.trade_id == states[1]["trade_id"]
    assert ledger.by_seq[2][0].execution.phase_after == TradePhase.CLOSED
    assert ledger.by_seq[2][1].execution.trade_id == states[2]["trade_id"]
    assert all(task.done() for task in system._tasks)


def alert_actions():
    timestamp = 1_672_531_200_000
    clock = EventClock(timestamp)
    agg = SignalAggregator(min_composite_score=0, signal_cooldown_seconds=0, clock=clock)
    lv = QualifiedLevel(100.05, LevelType.VAL, Side.BUY, 100)
    bias = DailyBias("d", qualified_levels=[lv])
    actions = []
    for seq, action in enumerate(("stop", "target", "exit"), 1):
        clock.set(timestamp + seq * 1000)
        entry = agg.process_signal("X", Signal(clock.now_ms(), SignalType.ABSORPTION, Side.BUY, 99.5, 100,
                                                {"bar_seq": seq * 2 - 1}), bias, 100, [])
        if seq == 1:
            actions.append(entry)
        clock.set(timestamp + seq * 1000 + 500)
        if action == "exit":
            close = agg.process_signal("X", Signal(clock.now_ms(), SignalType.EXHAUSTION, Side.SELL, 102, 100,
                                                    {"bar_seq": seq * 2}), bias, 100.2, [])
        else:
            bar = InfoBar(clock.now_ms(), 100, 100.8 if action == "target" else 100.1,
                          99.9 if action == "target" else 99.6, 100, seq=seq * 2)
            close = agg.check_exits("X", bar)
        actions.append(close)
    return actions


@pytest.mark.asyncio
async def test_telegram_formats_execution_prices_and_bar_close_without_network():
    bot = TelegramAlertBot("fake-token", "fake-chat")
    bot._bot = type("FakeBot", (), {"send_message": AsyncMock()})()
    actions = alert_actions()
    assert [a.action for a in actions] == ["enter", "stop", "target", "exit"]
    for action in actions:
        await bot.send_signal_alert("X", action)
    messages = [call.kwargs["text"] for call in bot._bot.send_message.await_args_list]
    assert len(messages) == 4
    assert "Entry: ~100.00 (bar close)" in messages[0]
    assert "ABSORPTION @ 99.50" in messages[0]
    assert "Entry: ~99.50" not in messages[0]
    assert "STOP HIT" in messages[1] and "TARGET HIT" in messages[2] and "EXIT SIGNAL" in messages[3]
    assert "Score:" not in messages[1] and "Score:" not in messages[2]
    assert "Exit: 100.20" in messages[3] and "EXHAUSTION @ 102.00" in messages[3]
    for action, message in zip(actions, messages):
        close = datetime.fromtimestamp(action.timestamp_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        assert f"Bar close: {close} UTC" in message
        assert "🟢 LONG" in message

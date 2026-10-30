"""All replay consumers use the same ordered, sequence-keyed execution ledger."""

import pytest

from orderflow_system.config.settings import get_btcusd_config
from orderflow_system.data.models import BarType, InfoBar, Side, Signal, SignalType, TradePhase
from orderflow_system.ledger import TradeLedger
from orderflow_system.pipeline import InfoBarPipeline
from orderflow_system.signals.profile_framing import DailyBias, LevelType, QualifiedLevel


def replay():
    pipeline = InfoBarPipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE)
    pipeline.aggregator.min_composite_score = 0
    pipeline.aggregator.signal_cooldown_seconds = 0
    levels = [QualifiedLevel(100, LevelType.VAL, Side.BUY, 100),
              QualifiedLevel(100, LevelType.VAH, Side.SELL, 100)]
    pipeline.profile_framing._bias_history.append(DailyBias("d", qualified_levels=levels))
    emitted = {
        1: (SignalType.ABSORPTION, Side.BUY, 100),
        2: (SignalType.ABSORPTION, Side.SELL, 100),
        4: (SignalType.ABSORPTION, Side.BUY, 100),
        5: (SignalType.DIVERGENCE, Side.SELL, 100.4),
    }

    def detect(bar, *_):
        if bar.seq not in emitted:
            return []
        kind, side, reference = emitted[bar.seq]
        return [Signal(bar.timestamp_ms, kind, side, reference, 100)]

    pipeline._detect = detect
    ohlc = [(100, 100.1, 99.9, 100), (100, 100.2, 99.6, 100),
            (100, 100.1, 99.2, 99.5), (100, 100.1, 99.9, 100),
            (100, 100.4, 99.9, 100.2)]
    ledger, results = TradeLedger(), []
    for seq, (o, h, l, c) in enumerate(ohlc, 1):
        # Tied timestamps must not collapse bars or resolve fills by timestamp.
        bar = InfoBar(100_002, o, h, l, c, volume=1, buy_volume=.5,
                      sell_volume=.5, dollar_value=c, seq=seq, duration_ms=0)
        result = pipeline.process_bar(bar)
        results.append(result)
        ledger.process_result(result)
    return ledger, results


def test_ledger_records_pipeline_fills_and_preserves_action_order():
    ledger, results = replay()
    assert [(r.bar_seq, r.action) for r in ledger.records] == [
        (1, "enter"), (2, "stop"), (2, "enter"), (3, "target"), (4, "enter"), (5, "exit")]
    assert [r.execution.fill_price for r in ledger.records] == pytest.approx([100, 99.7, 100, 99.4, 100, 100.2])
    assert [r.execution.pnl for r in ledger.closes] == pytest.approx([-.3, .6, .2])
    assert [r.execution.exit_reason for r in ledger.closes] == ["stop", "target", "exit"]
    assert len(ledger.entries) == len(ledger.closes) == 3
    assert [r.action for r in ledger.by_seq[2]] == ["stop", "enter"]
    old, replacement = [r.execution for r in ledger.by_seq[2]]
    assert old.trade_id != replacement.trade_id
    assert old.direction == Side.BUY and replacement.direction == Side.SELL
    assert old.phase_after == TradePhase.CLOSED and replacement.phase_after == TradePhase.POSITION_OPEN
    assert results[1].aggregated[0].execution is old
    assert old.stop_loss == pytest.approx(99.7) and replacement.stop_loss == pytest.approx(100.3)
    assert all(record.execution.qualified_level.price == 100 for record in ledger.entries)


def test_ledger_is_deterministic_and_uses_immutable_action_data():
    first, results = replay()
    second, _ = replay()
    assert first.records == second.records
    before = list(first.records)
    for result in results:
        result.bar.close = 1
        for aggregate in result.aggregated:
            if aggregate.qualified_level:
                aggregate.qualified_level.price = 1
    assert first.records == before
    assert first.closes[-1].execution.fill_price == 100.2

"""Runs the real pipeline on the real external dataset (read-only). Skipped when the volume is absent."""

from pathlib import Path

import pytest

from orderflow_system.data.models import BarType
from orderflow_system.validate_infobars import validate

ROOT = "/Volumes/Market-Data/info-bars"
pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not Path(ROOT, "BTCUSDT").is_dir(), reason="info-bar dataset not mounted"),
]


@pytest.mark.parametrize("bar_type", list(BarType), ids=lambda b: b.value)
def test_pipeline_on_real_bars(bar_type):
    out = validate(ROOT, "BTCUSDT", bar_type, "2023-11-19", "2023-11-24")   # spans the Nov-21 data outage
    assert out["bars"] > 500
    assert out["checks"], out
    assert len(out["start_independence"]["starts"]) == 3
    assert out["start_independence"]["data_gap"] is True
    assert out["ok"], out


def test_dollar_imbalance_direction_and_real_pattern_exit_regressions():
    import copy
    from orderflow_system.config.settings import get_btcusd_config
    from orderflow_system.data.infobar_feed import open_bars
    from orderflow_system.data.models import Side, SignalType, TradeState
    from orderflow_system.ledger import TradeLedger
    from orderflow_system.pipeline import InfoBarPipeline
    from orderflow_system.validate_infobars import LedgerAudit

    pipe = InfoBarPipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE)
    audit = LedgerAudit(pipe.aggregator.price_proximity_pct)
    encountered = set()
    entered_trade_id = None
    isolated_exit = None
    for bar in open_bars(ROOT, "BTCUSDT", BarType.DOLLAR_IMBALANCE,
                         start="2023-01-01", end="2023-03-16"):
        if bar.seq == 3403653210:
            # Earlier lifecycle corrections remove the original position before
            # this sequence. Keep its real detector history and test the actual
            # bar against an isolated open-position fixture through the pipeline.
            isolated = copy.deepcopy(pipe)
            entry = pipe.current_price
            trade = TradeState(instrument=pipe.symbol, direction=Side.BUY, qualified_level=entry)
            trade.advance_to_position(entry, bar.low * .99, bar.high * 1.01,
                                      entry_time_ms=pipe.bars[-1].timestamp_ms)
            isolated.aggregator._active_trades[pipe.symbol] = trade
            isolated_result = isolated.process_bar(bar)
            isolated_ledger = TradeLedger()
            isolated_ledger.process_result(isolated_result)
            isolated_exit = next(r.execution for r in isolated_ledger.closes if r.action == "exit")
            assert isolated_exit.fill_price == bar.close != bar.high
            assert isolated_exit.pnl == pytest.approx(bar.close - entry)
        result = pipe.process_bar(bar)
        audit.process_result(result)
        if bar.seq in (3236808279, 3403653210):
            encountered.add(bar.seq)
        if bar.seq == 3236808279:
            entry = next(a.execution for a in result.aggregated if a.action == "enter")
            assert entry.direction == entry.qualified_level.direction == Side.BUY
            assert pipe.aggregator.get_active_trade(pipe.symbol).direction == Side.BUY
            entered_trade_id = entry.trade_id
        if bar.seq == 3403653210:
            assert any(s.signal_type == SignalType.DIVERGENCE and s.direction == Side.SELL for s in result.signals)
            assert not result.aggregated  # the corrected natural replay has no position here
    assert encountered == {3236808279, 3403653210}
    assert isolated_exit is not None
    close = next(record.execution for record in audit.ledger.closes if record.execution.trade_id == entered_trade_id)
    assert close.pnl == pytest.approx(close.fill_price - close.entry_price)
    assert audit.report()["ok"], audit.report()


def test_tick_run_real_distant_absorption_is_not_entered():
    from orderflow_system.config.settings import get_btcusd_config
    from orderflow_system.data.infobar_feed import open_bars
    from orderflow_system.ledger import TradeLedger
    from orderflow_system.pipeline import InfoBarPipeline

    pipe = InfoBarPipeline(get_btcusd_config(), BarType.TICK_RUN)
    ledger = TradeLedger()
    encountered = False
    for bar in open_bars(ROOT, "BTCUSDT", BarType.TICK_RUN, start="2023-11-19", end="2023-11-24"):
        encountered |= bar.seq == 4304211644
        ledger.process_result(pipe.process_bar(bar))
    assert encountered, "the reviewed sequence must actually occur in this dataset version"
    assert ledger.entries, "the replay must exercise entry proximity"
    for record in ledger.entries:
        execution = record.execution
        assert abs(execution.entry_price - execution.qualified_level.price) / execution.entry_price < .002

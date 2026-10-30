"""One trade payload schema for current views and captured execution actions."""

from orderflow_system.dashboard.websocket_manager import _serialize
from orderflow_system.data.models import ExecutionSnapshot, TradePhase, TradeState


def serialize_trade(trade: TradeState | ExecutionSnapshot | None, instrument: str = "") -> dict:
    """Serialize current state or an action snapshot without recalculating its bracket ratio."""
    if isinstance(trade, ExecutionSnapshot):
        phase = trade.phase_after
        level = trade.qualified_level.price if trade.qualified_level is not None else None
        fill_price, reason, pnl = trade.fill_price, trade.exit_reason, trade.pnl
        bar_seq, timestamp_ms = trade.bar_seq, trade.timestamp_ms
    elif trade is not None:
        phase = trade.phase
        level = trade.qualified_level
        closed = phase == TradePhase.CLOSED
        fill_price = trade.exit_price if closed else trade.entry_price if phase == TradePhase.POSITION_OPEN else None
        reason = trade.notes if closed else None
        pnl = trade.pnl_ticks if closed else None
        bar_seq = None  # Current state need not originate from a detector action.
        timestamp_ms = trade.exit_time_ms if closed else trade.entry_time_ms or None
    else:
        phase, level, fill_price, reason, pnl, bar_seq, timestamp_ms = "none", None, None, None, None, None, None
    return _serialize({
        "trade_id": trade.trade_id if trade is not None else None,
        "instrument": trade.instrument if trade is not None else instrument,
        "direction": trade.direction if trade is not None else None,
        "phase": phase,
        "phase_after": phase,
        "entry_price": trade.entry_price if trade is not None else None,
        "stop_loss": trade.stop_loss if trade is not None else None,
        "take_profit": trade.take_profit if trade is not None else None,
        "rr_ratio": trade.rr_ratio if trade is not None else None,
        "fill_price": fill_price,
        "exit_reason": reason,
        "pnl": pnl,
        "pnl_ticks": pnl,
        "qualified_level": level,
        "bar_seq": bar_seq,
        "timestamp_ms": timestamp_ms,
    })

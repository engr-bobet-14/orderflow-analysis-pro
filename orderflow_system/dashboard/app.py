"""
FastAPI Dashboard Application
REST endpoints + WebSocket for the real-time orderflow trading terminal.

Everything is served from the info-bar pipeline: the x-axis is bar sequence, not
clock time. Endpoints take `bars=N` (most recent N info bars). Bars are
never re-aggregated into fixed time buckets.

Endpoints:
  GET  /api/instruments          — Active instruments with current stats
  GET  /api/bars/{symbol}        — Recent info bars (OHLC, volume, delta, ...)
  GET  /api/volume-profile/{sym} — Session VP (POC/VAH/VAL; bar-VWAP price axis)
  GET  /api/bias/{symbol}        — Daily bias and qualified levels
  GET  /api/signals/{symbol}     — Signal history
  GET  /api/markers/{symbol}     — Chart markers for the recent bars
  GET  /api/trade/{symbol}       — Active trade state
  GET  /api/delta/{symbol}       — Cumulative delta over recent bars
  GET  /api/footprint/{symbol}   — Bar-level buy/sell volume (no price levels exist)
  GET  /api/stats                — System-wide stats
  WS   /ws                       — Real-time stream
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from orderflow_system.dashboard.websocket_manager import WebSocketManager, _serialize
from orderflow_system.dashboard.trade_serializer import serialize_trade
from orderflow_system.data.models import Side, TradePhase

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# FastAPI app — created here, system ref set at startup
# ──────────────────────────────────────────────

app = FastAPI(
    title="Orderflow Trading Dashboard",
    description="Real-time orderflow analysis terminal",
    version="1.0.0",
)

# WebSocket manager — shared with main.py
ws_manager = WebSocketManager()

# Reference to OrderflowSystem — set by main.py before server starts
_system = None


def set_system(system):
    """Set the OrderflowSystem reference. Called by main.py on startup."""
    global _system
    _system = system


def get_system():
    """Get the OrderflowSystem reference."""
    return _system



# ──────────────────────────────────────────────
# Static files (frontend)
# ──────────────────────────────────────────────

# Disable browser caching for static files during development
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheMiddleware)

_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
async def index():
    """Serve the dashboard HTML."""
    index_path = os.path.join(_static_dir, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return JSONResponse({"error": "Dashboard frontend not found"}, status_code=404)


# ──────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────

@app.get("/api/instruments")
async def get_instruments():
    """List active instruments with current stats."""
    system = get_system()
    if not system:
        return []

    result = []
    for sym, pipeline in system.pipelines.items():
        stats = pipeline.stats
        trade = pipeline.aggregator.get_active_trade(sym)
        stats["trade_phase"] = trade.phase.value if trade else "none"
        stats["trade_direction"] = trade.direction.value if trade else "none"
        result.append(stats)

    return result


# ── Trade Scanner — ranked strategy status for ALL pairs ──

# Priority scores for strategy overall statuses (higher = closer to trade)
_STATUS_PRIORITY = {
    "IN_TRADE": 100,
    "TRAILING": 95,
    "BREAK_EVEN": 90,
    "ENTRY_READY": 85,
    "AT_LEVEL_SCANNING": 70,
    "WATCHING": 60,
    "WAITING_FOR_PRICE": 40,
    "NO_LEVELS": 20,
    "NO_DATA": 10,
    "OFFLINE": 5,
    "IDLE": 0,
}


@app.get("/api/scanner")
async def get_scanner():
    """Return strategy status for ALL pairs, ranked by trade proximity."""
    system = get_system()
    if not system:
        return []

    results = []
    for sym in system.pipelines:
        try:
            status = await get_strategy_status(sym)
            overall = status.get("overall", "IDLE")
            priority = _STATUS_PRIORITY.get(overall, 0)

            # Boost priority if there are more completed steps
            steps = status.get("steps", [])
            completed = sum(
                1 for s in steps
                if s.get("status") in ("completed", "triggered", "active")
            )
            priority += completed * 3  # up to +18 for 6 steps

            results.append({
                "symbol": sym,
                "overall": overall,
                "reason": status.get("reason", ""),
                "priority": priority,
                "steps_done": completed,
                "steps_total": len(steps),
                "bias_direction": status.get("bias_direction", "neutral"),
                "bias_confidence": status.get("bias_confidence", 0),
                "current_price": status.get("current_price", 0),
            })
        except Exception:
            results.append({
                "symbol": sym,
                "overall": "OFFLINE",
                "reason": "Error",
                "priority": 0,
                "steps_done": 0,
                "steps_total": 6,
                "bias_direction": "neutral",
                "bias_confidence": 0,
                "current_price": 0,
            })

    # Sort by priority descending
    results.sort(key=lambda x: x["priority"], reverse=True)
    return results


# ── Chart Markers — signal events formatted for TradingView ──

def _signal_to_marker(sig, is_buy: bool, action: str, time_s: int) -> dict:
    """Convert a signal to TradingView marker format."""
    sig_type = sig.signal_type.value if hasattr(sig.signal_type, 'value') else str(sig.signal_type)

    if action == "enter":
        return {"time": time_s, "position": "belowBar" if is_buy else "aboveBar",
                "color": "#00e676" if is_buy else "#ff1744", "shape": "circle",
                "text": "ENTRY \u25b2" if is_buy else "ENTRY \u25bc"}
    elif action == "break_even":
        return {"time": time_s, "position": "aboveBar", "color": "#42a5f5",
                "shape": "square", "text": "BE"}
    elif action == "trail":
        return {"time": time_s, "position": "aboveBar", "color": "#26c6da",
                "shape": "square", "text": "TRAIL"}
    elif action == "exit":
        return {"time": time_s, "position": "aboveBar", "color": "#ff1744",
                "shape": "circle", "text": "EXIT"}
    elif action == "exit_warning":
        return {"time": time_s, "position": "aboveBar", "color": "#ffeb3b",
                "shape": "circle", "text": "WARN"}
    else:
        if sig_type == "absorption":
            return {"time": time_s, "position": "belowBar" if is_buy else "aboveBar",
                    "color": "#26a69a" if is_buy else "#ef5350",
                    "shape": "arrowUp" if is_buy else "arrowDown", "text": "ABS"}
        elif sig_type == "initiative_auction":
            return {"time": time_s, "position": "belowBar" if is_buy else "aboveBar",
                    "color": "#66bb6a" if is_buy else "#ffa726",
                    "shape": "arrowUp" if is_buy else "arrowDown", "text": "INIT"}
        elif sig_type == "book_sweep":
            return {"time": time_s, "position": "aboveBar", "color": "#ab47bc",
                    "shape": "arrowDown", "text": "SWEEP"}
        elif sig_type == "exhaustion":
            return {"time": time_s, "position": "aboveBar", "color": "#ffeb3b",
                    "shape": "circle", "text": "EXHAUST"}
        elif sig_type == "delta_divergence":
            return {"time": time_s, "position": "aboveBar", "color": "#ff9800",
                    "shape": "circle", "text": "DIV"}
        else:
            return {"time": time_s, "position": "aboveBar", "color": "#9e9e9e",
                    "shape": "circle", "text": sig_type[:5]}


def _recent_bars(pipeline, count: int, bars: Optional[int]):
    n = bars if bars else count
    return pipeline.get_recent_bars(n)


@app.get("/api/markers/{symbol}")
async def get_markers(
    symbol: str,
    bars: int = Query(default=500, ge=1, le=5000),
    limit: int = Query(default=200, le=500),
):
    """Chart markers for signal events within the most recent `bars` bars."""
    system = get_system()
    if not system:
        return []

    pipeline = system.pipelines.get(symbol)
    if not pipeline:
        return []

    window = pipeline.get_recent_bars(bars)
    if not window:
        return []
    time_by_seq = dict(zip((b.seq for b in window), pipeline.get_recent_display_times(bars)))

    raw_signals = []
    for det in pipeline.detectors:
        raw_signals.extend(det._signal_history)

    markers = []
    consumed = set()
    # Each aggregate owns its action and consumed signals. Several actions may
    # share a bar, including an old trade's stop followed by a replacement entry.
    for agg in pipeline.aggregator.signal_history:
        seq = agg.execution.bar_seq
        if seq not in time_by_seq:
            continue
        consumed.update(id(sig) for sig in agg.signals)
        is_buy = agg.execution.direction == Side.BUY
        if agg.action in ("stop", "target"):
            markers.append({
                "time": time_by_seq[seq],
                "position": "belowBar" if is_buy else "aboveBar",
                "color": "#f85149" if agg.action == "stop" else "#3fb950",
                "shape": "square", "text": agg.action.upper(),
            })
        elif agg.signals:
            markers.append(_signal_to_marker(agg.signals[0], is_buy, agg.action, time_by_seq[seq]))

    for sig in raw_signals:
        seq = sig.details.get("bar_seq")
        time_s = time_by_seq.get(seq)
        if time_s is None or id(sig) in consumed:
            continue   # outside the displayed window
        markers.append(_signal_to_marker(sig, sig.direction == Side.BUY, "alert_only", time_s))

    markers.sort(key=lambda x: x["time"])
    return markers[-limit:]


@app.get("/api/bars/{symbol}")
async def get_bars(
    symbol: str,
    count: int = Query(default=500, le=5000),
    bars: Optional[int] = Query(default=None, ge=1, le=5000, description="Most recent N info bars"),
):
    """Most recent info bars, oldest first. Bars are never re-aggregated."""
    system = get_system()
    if not system:
        return []

    pipeline = system.pipelines.get(symbol)
    if not pipeline:
        return JSONResponse({"error": f"Unknown symbol: {symbol}"}, status_code=404)

    window = _recent_bars(pipeline, count, bars)
    result = []
    for c, t in zip(window, pipeline.get_recent_display_times(len(window))):
        result.append({
            "time": t,
            "time_ms": c.timestamp_ms,
            "seq": c.seq,
            "open": round(c.open, 6),
            "high": round(c.high, 6),
            "low": round(c.low, 6),
            "close": round(c.close, 6),
            "volume": round(c.volume, 4),
            "buy_volume": round(c.buy_volume, 4),
            "sell_volume": round(c.sell_volume, 4),
            "delta": round(c.delta, 4),
            "tick_count": c.tick_count,
            "dollar_value": round(c.dollar_value, 2),
            "threshold": c.threshold,
            "duration_ms": c.duration_ms,
            "data_gap": c.data_gap,
        })
    return result


@app.get("/api/volume-profile/{symbol}")
async def get_volume_profile(
    symbol: str,
    days: int = Query(default=5, le=30),
    bars: Optional[int] = Query(default=None, ge=1, le=5000, description="Profile of the most recent N bars"),
):
    """
    Volume profile. Price axis = each info bar's VWAP (the data has no price-level
    volume); `source` says so. With `bars` it profiles that many recent bars,
    otherwise it returns the most recently completed UTC-day sessions.
    """
    system = get_system()
    if not system:
        return []

    pipeline = system.pipelines.get(symbol)
    if not pipeline:
        return JSONResponse({"error": f"Unknown symbol: {symbol}"}, status_code=404)

    if bars:
        window = pipeline.get_recent_bars(bars)
        if not window:
            return []
        profiles = [pipeline.vp_engine.compute_from_bars(window, session_date=f"last_{len(window)}_bars")]
    else:
        profiles = pipeline.profile_framing.profile_history[-days:]
        if pipeline.bars_processed:
            profiles = list(profiles) + [pipeline.current_session_profile()]

    return [
        {
            "session_date": vp.session_date,
            "poc": vp.poc,
            "vah": vp.vah,
            "val": vp.val,
            "total_volume": vp.total_volume,
            "shape": vp.shape,
            "poc_position_pct": vp.poc_position_pct,
            "lvn_levels": vp.lvn_levels,
            "source": vp.source,
            "volume_at_price": {
                str(round(price, 6)): round(vol, 4)
                for price, vol in sorted(vp.volume_at_price.items())
            },
        }
        for vp in profiles
    ]


@app.get("/api/bias/{symbol}")
async def get_bias(symbol: str):
    """Get current daily bias and qualified levels."""
    system = get_system()
    if not system:
        return {"direction": "neutral", "confidence": 0, "qualified_levels": [], "notes": "System not running"}

    pipeline = system.pipelines.get(symbol)
    if not pipeline:
        return JSONResponse({"error": f"Unknown symbol: {symbol}"}, status_code=404)

    bias = pipeline.profile_framing.current_bias
    if not bias:
        return {
            "direction": "neutral",
            "confidence": 0,
            "qualified_levels": [],
            "notes": "No profile data yet",
        }

    return _serialize(bias)


@app.get("/api/signals/{symbol}")
async def get_signals(symbol: str, limit: int = Query(default=50, le=200)):
    """Get aggregated signal history (newest first)."""
    system = get_system()
    pipeline = system.pipelines.get(symbol) if system else None
    if not pipeline:
        return []
    return [_serialize(sig) for sig in list(pipeline.aggregator.signal_history)[::-1][:limit]]


@app.get("/api/trade/{symbol}")
async def get_trade(symbol: str):
    """Get active trade state for an instrument."""
    system = get_system()
    pipeline = system.pipelines.get(symbol) if system else None
    trade = pipeline.aggregator.get_active_trade(symbol) if pipeline else None
    return serialize_trade(trade, instrument=symbol)


@app.get("/api/strategy-status/{symbol}")
async def get_strategy_status(symbol: str):
    """
    Full Fabio methodology status: what the system is doing and WHY.
    Returns a step-by-step checklist of the strategy pipeline.
    """
    system = get_system()
    if not system:
        return _empty_strategy_status(symbol, "System not running")

    pipeline = system.pipelines.get(symbol)
    if not pipeline:
        return _empty_strategy_status(symbol, f"Unknown symbol: {symbol}")

    aggregator = pipeline.aggregator
    trade = aggregator.get_active_trade(symbol)
    bias = pipeline.profile_framing.current_bias
    current_price = pipeline.current_price
    absorption_signals = trade.absorption_signals if trade else []
    initiative_signals = trade.initiative_signals if trade else []
    rr_ratio = trade.rr_ratio if trade else 0

    # ── Step 1: Profile Framing ──
    step_profile = {
        "step": 1,
        "name": "Profile Framing",
        "status": "inactive",
        "icon": "📊",
        "detail": "No profile data yet",
        "sub": [],
    }
    if bias:
        step_profile["status"] = "active"
        step_profile["detail"] = bias.notes or f"{bias.profile_shape} profile"
        step_profile["sub"] = [
            f"Shape: {bias.profile_shape}",
            f"Direction: {bias.direction.value if hasattr(bias.direction, 'value') else bias.direction}",
            f"Confidence: {bias.confidence:.0f}%",
            f"POC: {bias.poc:.2f}" if bias.poc else "POC: --",
            f"VAH: {bias.vah:.2f}" if bias.vah else "VAH: --",
            f"VAL: {bias.val:.2f}" if bias.val else "VAL: --",
        ]
        if bias.merged_vah:
            step_profile["sub"].append(f"Merged VAH: {bias.merged_vah:.2f}")
        if bias.merged_val:
            step_profile["sub"].append(f"Merged VAL: {bias.merged_val:.2f}")

    # ── Step 2: Qualified Levels ──
    step_levels = {
        "step": 2,
        "name": "Qualified Levels",
        "status": "inactive",
        "icon": "🎯",
        "detail": "No levels qualified",
        "sub": [],
    }
    nearest_level = None
    nearest_dist = float("inf")
    if bias and bias.qualified_levels:
        step_levels["status"] = "active"
        step_levels["detail"] = f"{len(bias.qualified_levels)} level(s) qualified"
        for lv in bias.qualified_levels:
            dist_pct = abs(current_price - lv.price) / max(current_price, 1) * 100 if current_price > 0 else 0
            proximity = "AT LEVEL" if dist_pct < 0.2 else f"{dist_pct:.2f}% away"
            dir_label = "LONG" if lv.direction.value == "buy" else "SHORT"
            step_levels["sub"].append(
                f"{lv.level_type.value.upper()} @ {lv.price:.2f} → {dir_label} | Str: {lv.strength:.0f}% | {proximity}"
            )
            if current_price > 0 and abs(current_price - lv.price) < nearest_dist:
                nearest_dist = abs(current_price - lv.price)
                nearest_level = lv

    # ── Step 3: Price at Level? ──
    at_level = False
    proximity_pct = aggregator.price_proximity_pct
    step_price = {
        "step": 3,
        "name": "Price at Level",
        "status": "inactive",
        "icon": "📍",
        "detail": "Waiting for price to reach a qualified level",
        "sub": [],
    }
    if current_price > 0 and nearest_level:
        dist_pct = abs(current_price - nearest_level.price) / max(current_price, 1)
        if dist_pct < proximity_pct:
            at_level = True
            step_price["status"] = "triggered"
            step_price["detail"] = f"Price AT {nearest_level.level_type.value.upper()} @ {nearest_level.price:.2f}"
        else:
            step_price["status"] = "waiting"
            step_price["detail"] = f"Price {current_price:.2f} is {dist_pct*100:.2f}% from nearest level ({nearest_level.level_type.value.upper()} @ {nearest_level.price:.2f})"
        step_price["sub"].append(f"Current: {current_price:.2f}")
        step_price["sub"].append(f"Nearest: {nearest_level.level_type.value.upper()} @ {nearest_level.price:.2f}")
        step_price["sub"].append(f"Proximity threshold: {proximity_pct*100:.1f}%")

    # ── Step 4: Absorption Check ──
    step_absorption = {
        "step": 4,
        "name": "Absorption",
        "status": "inactive",
        "icon": "🛡️",
        "detail": "Waiting for absorption at level",
        "sub": [],
    }
    has_absorption = False
    if trade and absorption_signals:
        has_absorption = True
        step_absorption["status"] = "triggered"
        step_absorption["detail"] = f"{len(absorption_signals)} absorption signal(s) detected"
        for sig in absorption_signals:
            s = _serialize(sig)
            step_absorption["sub"].append(
                f"Str: {s.get('strength', 0):.0f} | Price: {s.get('price_level', 0):.2f} | {s.get('direction', '?')}"
            )
    elif at_level:
        step_absorption["status"] = "watching"
        step_absorption["detail"] = "Price at level — scanning for absorption patterns"

    # ── Step 5: Entry Decision ──
    step_entry = {
        "step": 5,
        "name": "Entry Decision",
        "status": "inactive",
        "icon": "🚪",
        "detail": "No entry conditions met",
        "sub": [],
    }
    if trade:
        phase_val = trade.phase.value if hasattr(trade.phase, 'value') else trade.phase
        if phase_val in ("absorption", "position_open", "break_even", "trailing"):
            step_entry["status"] = "triggered"
            if trade.entry_price > 0:
                step_entry["detail"] = f"Entered {'LONG' if trade.direction == Side.BUY else 'SHORT'} @ {trade.entry_price:.2f}"
            else:
                step_entry["detail"] = "Absorption confirmed — awaiting position fill"
            step_entry["sub"].append(f"Phase: {phase_val}")
            if trade.entry_price:
                step_entry["sub"].append(f"Entry: {trade.entry_price:.2f}")
            if trade.stop_loss:
                step_entry["sub"].append(f"SL: {trade.stop_loss:.2f}")
            if trade.take_profit:
                step_entry["sub"].append(f"TP: {trade.take_profit:.2f}")
            if rr_ratio:
                step_entry["sub"].append(f"R:R: {rr_ratio:.1f}x")
    elif has_absorption:
        step_entry["status"] = "watching"
        step_entry["detail"] = "Absorption seen — evaluating composite score"
        score = aggregator.min_composite_score
        step_entry["sub"].append(f"Min score needed: {score}")

    # ── Step 6: Trade Management ──
    step_mgmt = {
        "step": 6,
        "name": "Trade Management",
        "status": "inactive",
        "icon": "⚙️",
        "detail": "No active trade to manage",
        "sub": [],
    }
    if trade:
        phase_val = trade.phase.value if hasattr(trade.phase, 'value') else str(trade.phase)
        if phase_val == "position_open":
            step_mgmt["status"] = "active"
            step_mgmt["detail"] = "Position open — waiting for initiative to move to BE"
            step_mgmt["sub"].append("Next: Initiative print → Break-Even trigger")
        elif phase_val == "break_even":
            step_mgmt["status"] = "active"
            step_mgmt["detail"] = "Break-even set — waiting for more initiative to trail"
            step_mgmt["sub"].append(f"BE Price: {trade.entry_price:.2f}")
            step_mgmt["sub"].append("Next: Initiative print → Trail stop")
        elif phase_val == "trailing":
            step_mgmt["status"] = "active"
            step_mgmt["detail"] = f"Trailing stop @ {trade.stop_loss:.2f}"
            step_mgmt["sub"].append(f"Trail: {trade.stop_loss:.2f}")
            step_mgmt["sub"].append(f"Initiative signals: {len(initiative_signals)}")
            step_mgmt["sub"].append("Next: More initiative → tighter trail | Exhaustion/Divergence → exit")
        elif phase_val == "closed":
            step_mgmt["status"] = "completed"
            step_mgmt["detail"] = f"Trade closed — PnL: {trade.pnl_ticks:.1f} ticks"
            if trade.notes:
                step_mgmt["sub"].append(f"Reason: {trade.notes}")

    # ── Build overall status ──
    if not bias:
        overall = "NO_DATA"
        reason = "No volume profile data — waiting for market data and profile computation"
    elif not bias.qualified_levels:
        overall = "NO_LEVELS"
        reason = "Profile computed but no levels qualified — flat/unclear market structure"
    elif not at_level and (not trade or trade.phase == TradePhase.CLOSED):
        overall = "WAITING_FOR_PRICE"
        reason = f"Levels ready but price ({current_price:.2f}) hasn't reached them — patience"
    elif at_level and not has_absorption and (not trade or trade.phase == TradePhase.CLOSED):
        overall = "AT_LEVEL_SCANNING"
        reason = f"Price at {nearest_level.level_type.value.upper()} — scanning for absorption patterns"
    elif trade and trade.phase == TradePhase.WATCHING:
        overall = "WATCHING"
        reason = f"Watching level @ {trade.qualified_level:.2f} for absorption"
    elif trade and trade.phase == TradePhase.ABSORPTION_DETECTED:
        overall = "ENTRY_READY"
        reason = "Absorption confirmed — entry signal active"
    elif trade and trade.phase == TradePhase.POSITION_OPEN:
        overall = "IN_TRADE"
        reason = "Position open — managing trade"
    elif trade and trade.phase == TradePhase.BREAK_EVEN:
        overall = "BREAK_EVEN"
        reason = "Break-even set — trailing mode pending"
    elif trade and trade.phase == TradePhase.TRAILING:
        overall = "TRAILING"
        reason = f"Trailing stop @ {trade.stop_loss:.2f}"
    else:
        overall = "IDLE"
        reason = "System active — monitoring"

    return {
        "symbol": symbol,
        "overall": overall,
        "reason": reason,
        "current_price": current_price,
        "steps": [step_profile, step_levels, step_price, step_absorption, step_entry, step_mgmt],
        "trade": serialize_trade(trade, instrument=symbol),
        "bias_direction": bias.direction.value if bias and hasattr(bias.direction, 'value') else (bias.direction if bias else "neutral"),
        "bias_confidence": bias.confidence if bias else 0,
    }


def _empty_strategy_status(symbol: str, reason: str):
    """Return empty strategy status."""
    return {
        "symbol": symbol,
        "overall": "OFFLINE",
        "reason": reason,
        "current_price": 0,
        "steps": [],
        "trade": None,
        "bias_direction": "neutral",
        "bias_confidence": 0,
    }


@app.get("/api/delta/{symbol}")
async def get_delta(
    symbol: str,
    count: int = Query(default=500, le=5000),
    bars: Optional[int] = Query(default=None, ge=1, le=5000),
):
    """Cumulative delta over the most recent info bars (running sum from the window start)."""
    system = get_system()
    pipeline = system.pipelines.get(symbol) if system else None
    if not pipeline:
        return JSONResponse({"error": f"Unknown symbol: {symbol}"}, status_code=404)

    window = _recent_bars(pipeline, count, bars)
    cum = 0.0
    result = []
    for b, t in zip(window, pipeline.get_recent_display_times(len(window))):
        cum += b.delta
        result.append({
            "time": t, "time_ms": b.timestamp_ms, "seq": b.seq,
            "value": round(cum, 4), "bar_delta": round(b.delta, 4),
        })
    return result


@app.get("/api/footprint/{symbol}")
async def get_footprint(
    symbol: str,
    count: int = Query(default=100, le=1000),
):
    """
    Bar-level aggressive buy/sell volume. The info-bar data has NO price-level
    volume, so `levels` is always empty and `has_price_levels` is false: a real
    footprint ladder needs trade-level data this dataset does not contain.
    """
    system = get_system()
    pipeline = system.pipelines.get(symbol) if system else None
    if not pipeline:
        return JSONResponse({"error": f"Unknown symbol: {symbol}"}, status_code=404)

    window = pipeline.get_recent_bars(count)
    return {
        "has_price_levels": False,
        "reason": "Information-driven bars carry bar totals only; no price-level volume exists.",
        "bars": [
            {
                "time": t, "time_ms": b.timestamp_ms, "seq": b.seq,
                "buy_volume": round(b.buy_volume, 4), "sell_volume": round(b.sell_volume, 4),
                "delta": round(b.delta, 4), "delta_pct": round(b.delta_pct, 4),
                "levels": [],
            }
            for b, t in zip(window, pipeline.get_recent_display_times(len(window)))
        ],
    }


@app.get("/api/stats")
async def get_stats():
    """Get system-wide statistics."""
    system = get_system()
    if not system:
        return {"data_source": "none", "ws_clients": ws_manager.client_count,
                "instruments": [], "running": False}

    return {
        "data_source": f"info-bars:{system.info_bars.bar_type.value}",
        "replay_finished": system.replay_finished,
        "ws_clients": ws_manager.client_count,
        "instruments": [p.stats for p in system.pipelines.values()],
        "running": system._running,
    }


# ──────────────────────────────────────────────
# WebSocket Endpoint
# ──────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Real-time data stream.
    Clients receive all broadcasts on all channels.
    """
    await ws_manager.connect(ws)
    try:
        # Send initial state snapshot
        system = get_system()
        if system:
            await _send_initial_state(ws, system)

        # Keep connection alive — listen for client messages (pings, etc.)
        while True:
            data = await ws.receive_text()
            # Could handle subscriptions/commands here
            if data == "ping":
                await ws.send_text('{"channel":"pong","data":{}}')
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WebSocket error: {e}")
    finally:
        await ws_manager.disconnect(ws)


async def _send_initial_state(ws: WebSocket, system):
    """Send current state snapshot to a newly connected client."""
    import json, time

    for sym, pipeline in system.pipelines.items():
        # Send current stats
        stats = pipeline.stats
        trade = pipeline.aggregator.get_active_trade(sym)
        stats["trade_phase"] = trade.phase.value if trade else "none"

        await ws.send_text(json.dumps({
            "channel": "stats",
            "symbol": sym,
            "data": _serialize(stats),
            "ts": int(time.time() * 1000),
        }))

        # Send current bias
        bias = pipeline.profile_framing.current_bias
        if bias:
            await ws.send_text(json.dumps({
                "channel": "bias",
                "symbol": sym,
                "data": _serialize(bias),
                "ts": int(time.time() * 1000),
            }))

        # Send current state, including WATCHING or a reset with no active trade.
        await ws.send_text(json.dumps({
            "channel": "trade_state",
            "symbol": sym,
            "data": serialize_trade(trade, instrument=sym),
            "ts": int(time.time() * 1000),
        }))

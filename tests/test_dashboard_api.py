"""Dashboard endpoints serve info bars faithfully (no time buckets, no fabricated price levels)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from orderflow_system.dashboard import app as dash
from orderflow_system.config.settings import get_btcusd_config
from orderflow_system.data.models import BarType
from orderflow_system.pipeline import InfoBarPipeline

from test_pipeline import synth_bars


@pytest.fixture
def client():
    bars = synth_bars(3000)
    # several bars closing in the same second / same ms: the chart needs unique increasing x
    for i in (100, 101, 102, 500):
        bars[i].timestamp_ms = bars[i - 1].timestamp_ms
    pipe = InfoBarPipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE)
    for b in bars:
        pipe.process_bar(b)
    system = SimpleNamespace(
        pipelines={"BTCUSDT": pipe}, _running=True, replay_finished=True,
        info_bars=SimpleNamespace(bar_type=BarType.DOLLAR_IMBALANCE),
    )
    dash.set_system(system)
    yield TestClient(dash.app), pipe
    dash.set_system(None)


def test_bars_are_info_bars_with_unique_increasing_display_time(client):
    c, pipe = client
    rows = c.get("/api/bars/BTCUSDT?bars=1000").json()
    assert len(rows) == 1000
    times = [r["time"] for r in rows]
    assert all(b > a for a, b in zip(times, times[1:]))
    assert [r["seq"] for r in rows] == sorted(r["seq"] for r in rows)
    # exact close stays available and untouched
    assert rows[-1]["time_ms"] == pipe.bars[-1].timestamp_ms
    assert rows[-1]["delta"] == pytest.approx(pipe.bars[-1].delta, abs=1e-3)


def test_window_count_is_stable(client):
    c, _ = client
    a = c.get("/api/bars/BTCUSDT?bars=50").json()
    b = c.get("/api/bars/BTCUSDT?bars=50&unused=3600").json()
    assert a == b


def test_footprint_endpoint_never_returns_price_levels(client):
    c, _ = client
    out = c.get("/api/footprint/BTCUSDT?count=20").json()
    assert out["has_price_levels"] is False
    assert len(out["bars"]) == 20 and all(b["levels"] == [] for b in out["bars"])


def test_volume_profile_is_labelled_bar_vwap(client):
    c, _ = client
    vp = c.get("/api/volume-profile/BTCUSDT?bars=500").json()
    assert vp[0]["source"] == "bar_vwap" and vp[0]["total_volume"] > 0
    sessions = c.get("/api/volume-profile/BTCUSDT").json()
    assert sessions and all(s["source"] == "bar_vwap" for s in sessions)


def test_markers_align_with_displayed_bars(client):
    c, _ = client
    window = c.get("/api/bars/BTCUSDT?bars=1500").json()
    shown = {r["time"] for r in window}
    markers = c.get("/api/markers/BTCUSDT?bars=1500&limit=500").json()
    assert markers and all(m["time"] in shown for m in markers)


def test_stats_instruments_strategy_and_removed_endpoints(client):
    c, _ = client
    assert c.get("/api/stats").json()["data_source"] == "info-bars:fixed_dollar_imbalance"
    inst = c.get("/api/instruments").json()[0]
    assert inst["symbol"] == "BTCUSDT" and inst["bar_type"] == "fixed_dollar_imbalance"
    assert c.get("/api/strategy-status/BTCUSDT").json()["symbol"] == "BTCUSDT"
    assert c.get("/api/scanner").json()[0]["symbol"] == "BTCUSDT"
    for gone in ("orderbook", "tape", "microstructure"):
        assert c.get(f"/api/{gone}/BTCUSDT").status_code == 404
    assert c.get("/api/bars/NOPE").status_code == 404


def test_endpoints_empty_without_system():
    dash.set_system(None)
    c = TestClient(dash.app)
    assert c.get("/api/bars/BTCUSDT").json() == []
    assert c.get("/api/instruments").json() == []
    assert c.get("/api/stats").json()["running"] is False


def test_websocket_initial_state_and_ping(monkeypatch):
    pipe = marker_pipeline(monkeypatch)
    c = TestClient(dash.app)
    assert pipe.aggregator.signal_history and pipe.profile_framing.current_bias
    with c.websocket_connect("/ws") as ws:
        messages = [ws.receive_json() for _ in range(3)]
        assert {m["channel"] for m in messages} == {"stats", "bias", "trade_state"}
        state = next(m["data"] for m in messages if m["channel"] == "trade_state")
        trade = pipe.aggregator.get_active_trade(pipe.symbol)
        assert state["trade_id"] == trade.trade_id
        assert state["phase"] == trade.phase.value
        ws.send_text("ping")
        assert ws.receive_json()["channel"] == "pong"
    dash.set_system(None)


def marker_pipeline(monkeypatch, include_divergence=False):
    from orderflow_system.data.models import InfoBar, Side, Signal, SignalType
    from orderflow_system.signals.profile_framing import DailyBias, LevelType, QualifiedLevel

    pipe = InfoBarPipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE)
    pipe.aggregator.min_composite_score = 0
    pipe.aggregator.signal_cooldown_seconds = 60 if include_divergence else 0
    levels = [QualifiedLevel(100, LevelType.VAL, Side.BUY, 100),
              QualifiedLevel(100, LevelType.VAH, Side.SELL, 100)]
    pipe.profile_framing._bias_history.append(DailyBias("d", qualified_levels=levels))

    def detect(bar, *_):
        side = Side.BUY if bar.seq == 1 else Side.SELL
        absorption = Signal(bar.timestamp_ms, SignalType.ABSORPTION, side, 100, 100)
        pipe.absorption._signal_history.append(absorption)
        signals = [absorption]
        if include_divergence:
            divergence = Signal(bar.timestamp_ms, SignalType.DIVERGENCE, Side.SELL, 100.1, 10)
            pipe.divergence._signal_history.append(divergence)
            signals.append(divergence)
        return signals

    monkeypatch.setattr(pipe, "_detect", detect)
    pipe.process_bar(InfoBar(100_002, 100, 100.1, 99.9, 100, volume=1, buy_volume=.5,
                            sell_volume=.5, dollar_value=100, seq=1, duration_ms=1))
    system = SimpleNamespace(pipelines={"BTCUSDT": pipe})
    dash.set_system(system)
    return pipe


def test_markers_associate_only_consumed_signals(monkeypatch):
    marker_pipeline(monkeypatch, include_divergence=True)
    try:
        c = TestClient(dash.app)
        markers = c.get("/api/markers/BTCUSDT").json()
        assert len(markers) == 2
        entry, = [m for m in markers if m["text"].startswith("ENTRY")]
        assert entry["position"] == "belowBar"
        assert [m["text"] for m in markers if m["text"] == "DIV"] == ["DIV"]
    finally:
        dash.set_system(None)


def test_markers_keep_stop_and_replacement_entry_on_one_bar(monkeypatch):
    from orderflow_system.data.models import InfoBar

    pipe = marker_pipeline(monkeypatch)
    pipe.process_bar(InfoBar(100_002, 100, 100.1, 99.6, 100, volume=1, buy_volume=.5,
                            sell_volume=.5, dollar_value=100, seq=2, duration_ms=0))
    try:
        markers = TestClient(dash.app).get("/api/markers/BTCUSDT").json()
        last_time = pipe.get_recent_display_times(1)[0]
        last = [m for m in markers if m["time"] == last_time]
        assert len(last) == 2
        stop, = [m for m in last if m["text"] == "STOP"]
        entry, = [m for m in last if m["text"].startswith("ENTRY")]
        assert stop["position"] == "belowBar"  # closed BUY trade
        assert entry["position"] == "aboveBar"  # replacement SELL trade
    finally:
        dash.set_system(None)


def test_dashboard_javascript_parses():
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if not node:
        pytest.skip("node unavailable for JavaScript syntax verification")
    source = Path(__file__).resolve().parents[1] / "orderflow_system/dashboard/static/app.js"
    result = subprocess.run([node, "--check", str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

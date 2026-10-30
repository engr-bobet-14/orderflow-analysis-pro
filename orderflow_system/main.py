"""
Main Orchestrator — wires the info-bar pipeline to storage, alerts and the dashboard.

Architecture:
  Parquet info-bar dataset -> InfoBarReplayFeed -> InfoBarPipeline (pure, deterministic)
                                                      |  delta / footprint / patterns /
                                                      |  session profile / bias / signal state machine
                                                      v
                                   SQLite (optional) + Telegram (optional) + Dashboard WebSocket

The pipeline (orderflow_system/pipeline.py) does all analysis with no I/O and no
wall clock. This module only moves data in and results out.

Data source: information-driven bars (six bar types, selected via --bar-type or
settings.INFO_BARS). Live Binance feeds are not implemented yet.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import hashlib
import json
import signal
import sys
from dataclasses import asdict, replace
from typing import Optional

from orderflow_system.config.settings import (
    DASHBOARD,
    DB_PATH,
    INFO_BARS,
    LOG_LEVEL,
    TELEGRAM,
    InfoBarConfig,
    InstrumentConfig,
    get_all_configs,
)
from orderflow_system.alerts.telegram_bot import TelegramAlertBot
from orderflow_system.dashboard.app import app as dashboard_app, set_system, ws_manager
from orderflow_system.dashboard.websocket_manager import WebSocketManager
from orderflow_system.dashboard.trade_serializer import serialize_trade
from orderflow_system.data.database import Database
from orderflow_system.data.infobar_feed import InfoBarReader, InfoBarReplayFeed, open_bars
from orderflow_system.data.models import BarType, InfoBar
from orderflow_system.pipeline import BarResult, InfoBarPipeline

logger = logging.getLogger("orderflow_system")

_DB_FLUSH_BARS = 2000
_STATS_EVERY_BARS = 500


class OrderflowSystem:
    """
    Runs one info-bar replay through the pipeline and fans results out.

    send_alerts: Telegram alerts are OFF by default because a replay emits
    historical signals; enable only when the feed is the live edge.
    """

    def __init__(
        self,
        instruments: Optional[list[InstrumentConfig]] = None,
        info_bars: InfoBarConfig = INFO_BARS,
        send_alerts: bool = False,
    ):
        instruments = instruments or get_all_configs()
        self.info_bars = info_bars
        self.send_alerts = send_alerts

        cfg = next((c for c in instruments if c.instrument.value == info_bars.symbol), None)
        if cfg is None:
            raise ValueError(
                f"No instrument config for symbol {info_bars.symbol!r}; "
                f"configured: {[c.instrument.value for c in instruments]}"
            )
        self.symbol = cfg.instrument.value
        config_payload = asdict(cfg)
        self.config_digest = hashlib.sha256(json.dumps(
            config_payload, sort_keys=True, separators=(",", ":"), default=lambda value: value.value,
        ).encode()).hexdigest()
        self.run_id: Optional[str] = None
        self._first_session_date: Optional[str] = None
        self._first_session_complete = False
        self._tasks: list[asyncio.Task] = []
        self.pipelines: dict[str, InfoBarPipeline] = {
            self.symbol: InfoBarPipeline(
                cfg, bar_type=info_bars.bar_type, history_bars=info_bars.history_bars
            )
        }

        self.telegram = TelegramAlertBot(
            bot_token=TELEGRAM.bot_token, chat_id=TELEGRAM.chat_id,
        )
        self.db = Database(DB_PATH)
        self.ws_manager: WebSocketManager = ws_manager
        self.reader: Optional[InfoBarReader] = None
        self.feed: Optional[InfoBarReplayFeed] = None
        self.replay_finished = False
        self._running = False
        self._stopping = False
        self._stop_event = asyncio.Event()
        self._bar_buffer: list[InfoBar] = []

    # ── lifecycle ───────────────────────────────────────────

    async def start(self):
        logger.info("=" * 60)
        logger.info("  ORDERFLOW ANALYSIS — INFORMATION-DRIVEN BARS")
        logger.info("=" * 60)

        # Discover the dataset before allocating a SQLite worker thread.
        self.reader = open_bars(
            self.info_bars.root, self.info_bars.symbol, self.info_bars.bar_type,
            start=self.info_bars.start, end=self.info_bars.end, strict=self.info_bars.strict,
            version=self.info_bars.version,
        )
        info = self.reader.info
        logger.info(
            f"Dataset: {info.parquet_path} | {self.info_bars.bar_type.value} "
            f"| window [{self.info_bars.start or 'begin'} .. {self.info_bars.end or 'end'})"
        )
        if self.info_bars.persist:
            await self.db.connect()
        try:
            if self.info_bars.persist:
                run_payload = [info.dataset_id, self.info_bars.bar_type.value,
                               self.reader.start_ms, self.reader.end_ms, self.config_digest]
                self.run_id = hashlib.sha256(json.dumps(
                    run_payload, separators=(",", ":"),
                ).encode()).hexdigest()
                await self.db.create_run(
                    self.run_id, info.dataset_id, str(info.parquet_path.resolve()), info.version,
                    self.info_bars.bar_type, self.symbol, self.reader.start_ms, self.reader.end_ms,
                    self.config_digest,
                )
            await self.telegram.initialize()
            self.feed = InfoBarReplayFeed(
                self.reader, self._on_bar, bars_per_second=self.info_bars.bars_per_second
            )
            self._running = True
            self._tasks = [asyncio.create_task(self._replay(), name="orderflow-replay")]
            if DASHBOARD.enabled:
                import uvicorn
                set_system(self)
                server = uvicorn.Server(uvicorn.Config(
                    dashboard_app, host=DASHBOARD.host, port=DASHBOARD.port,
                    log_level=DASHBOARD.log_level, loop="none",
                ))
                self._uvicorn = server
                self._tasks.append(asyncio.create_task(server.serve(), name="orderflow-dashboard"))
                logger.info(f"Dashboard enabled at http://{DASHBOARD.host}:{DASHBOARD.port}")
            stop_task = asyncio.create_task(self._stop_event.wait(), name="orderflow-stop")
            self._tasks.append(stop_task)
            done, _ = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
            # Re-raise any child failure only after the finally block has joined siblings.
            for task in done:
                task.result()
        finally:
            try:
                await self.stop()
            finally:
                for task in self._tasks:
                    if not task.done():
                        task.cancel()
                if self._tasks:
                    await asyncio.gather(*self._tasks, return_exceptions=True)
                try:
                    if self.db._db is not None:
                        await self._flush_bars()
                finally:
                    await self.db.close()
                    self._running = False
                    if DASHBOARD.enabled:
                        set_system(None)
                    logger.info("System stopped.")

    async def _replay(self):
        sent = await self.feed.run()
        await self._flush_bars()
        self.replay_finished = True
        pipeline = self.pipelines[self.symbol]
        rep = self.reader.report
        logger.info(
            f"Replay finished: {sent} bars | rejected={rep.rows_rejected} "
            f"tied_ts={rep.tied_timestamps} gap_bars={rep.bars_spanning_missing_hours} "
            f"| {pipeline.stats}"
        )
        await self.ws_manager.broadcast_stats(self._stats_payload())
        if DASHBOARD.enabled:
            await self._stop_event.wait()    # keep serving the dashboard until stopped
        else:
            await self.stop()

    async def stop(self):
        if self._stopping:      # uvicorn re-raises the signal to our handler: stop() runs once
            return
        self._stopping = True
        logger.info("Shutting down...")
        self._running = False
        if self.feed:
            await self.feed.stop()
        if getattr(self, "_uvicorn", None):
            self._uvicorn.should_exit = True
        self._stop_event.set()

    # ── per-bar I/O ─────────────────────────────────────────

    async def _on_bar(self, bar: InfoBar):
        pipeline = self.pipelines[self.symbol]
        result: BarResult = pipeline.process_bar(bar)   # sync, no I/O
        if self._first_session_date is None:
            from orderflow_system.pipeline import utc_date
            self._first_session_date = utc_date(bar.timestamp_ms)
            session_midnight_ms = (bar.timestamp_ms // 86_400_000) * 86_400_000
            predecessor_ms = self.reader.first_predecessor_ms if self.reader is not None else None
            self._first_session_complete = predecessor_ms is None or predecessor_ms < session_midnight_ms
        paced = self.info_bars.bars_per_second > 0

        if self.info_bars.persist:
            self._bar_buffer.append(bar)
            for sig in result.signals:
                await self.db.insert_signal(
                    self.symbol, sig, bar_type=self.info_bars.bar_type, commit=False,
                    run_id=self.run_id, bar_seq=bar.seq,
                )
            if len(self._bar_buffer) >= _DB_FLUSH_BARS:
                await self._flush_bars()

        if result.closed_profile is not None:
            await self._on_session_closed(pipeline, result)

        for agg in result.aggregated:
            trade = agg.execution
            if self.send_alerts:
                await self.telegram.send_signal_alert(
                    self.symbol, agg, pipeline.profile_framing.current_bias, trade
                )
            if paced:
                await self.ws_manager.broadcast_signal(self.symbol, agg)
                if trade:
                    await self.ws_manager.broadcast_trade_state(self.symbol, serialize_trade(trade))

        # Dashboard polls bars over REST; in a fast replay only periodic stats are pushed.
        if pipeline.bars_processed % _STATS_EVERY_BARS == 0 or paced:
            await self.ws_manager.broadcast_stats(self._stats_payload())

    async def _on_session_closed(self, pipeline: InfoBarPipeline, result: BarResult):
        vp, bias = result.closed_profile, result.bias
        if self.info_bars.persist:
            complete = vp.session_date != self._first_session_date or self._first_session_complete
            await self.db.insert_volume_profile(
                self.symbol, vp, bar_type=self.info_bars.bar_type,
                run_id=self.run_id, complete=complete,
            )
        if self.send_alerts and bias is not None:
            await self.telegram.send_daily_bias_update(self.symbol, bias)
        if self.info_bars.bars_per_second > 0:
            await self.ws_manager.broadcast_volume_profile(self.symbol, {
                "poc": vp.poc, "vah": vp.vah, "val": vp.val, "shape": vp.shape,
                "total_volume": vp.total_volume, "lvn_levels": vp.lvn_levels,
                "source": vp.source,
            })
            if bias is not None:
                await self.ws_manager.broadcast_bias(self.symbol, bias)

    async def _flush_bars(self):
        if self._bar_buffer:
            await self.db.insert_bars_batch(
                self.symbol, self.info_bars.bar_type, self._bar_buffer,
                dataset_id=self.reader.info.dataset_id,
            )
            self._bar_buffer = []
        if self.db._db is not None:
            await self.db.commit()

    def _stats_payload(self) -> dict:
        return {
            "data_source": f"info-bars:{self.info_bars.bar_type.value}",
            "replay_finished": self.replay_finished,
            "ws_clients": self.ws_manager.client_count,
            "instruments": [p.stats for p in self.pipelines.values()],
        }


def setup_logging():
    """Configure logging for the system."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("orderflow_system.log", encoding="utf-8"),
        ],
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Orderflow analysis on information-driven bars")
    p.add_argument("--bar-type", choices=[b.value for b in BarType], default=INFO_BARS.bar_type.value)
    p.add_argument("--start", default=INFO_BARS.start, help="Inclusive UTC date/datetime")
    p.add_argument("--end", default=INFO_BARS.end, help="Exclusive UTC date/datetime")
    p.add_argument("--root", default=INFO_BARS.root)
    p.add_argument("--version", default=INFO_BARS.version, help="Dataset stamp, e.g. 20261026T135000Z")
    p.add_argument("--speed", type=float, default=INFO_BARS.bars_per_second,
                   help="Replay bars/second (0 = as fast as possible)")
    p.add_argument("--persist", action="store_true", default=INFO_BARS.persist,
                   help="Write bars, signals and profiles to SQLite")
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--alerts", action="store_true", help="Send Telegram alerts for replayed signals")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None):
    """Entry point."""
    args = parse_args(argv)
    setup_logging()
    if args.no_dashboard:
        DASHBOARD.enabled = False

    cfg = replace(
        INFO_BARS, root=args.root, bar_type=BarType(args.bar_type), version=args.version,
        start=args.start, end=args.end, bars_per_second=args.speed, persist=args.persist,
    )
    logger.info(f"Info bars: {cfg.bar_type.value} {cfg.symbol} from {cfg.root}")

    system = OrderflowSystem(info_bars=cfg, send_alerts=args.alerts)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def shutdown_handler():
        logger.info("Received shutdown signal...")
        loop.create_task(system.stop())

    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, shutdown_handler)

    try:
        loop.run_until_complete(system.start())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        loop.run_until_complete(system.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()

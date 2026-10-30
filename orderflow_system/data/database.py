"""
SQLite storage for info bars, ticks, volume profiles, and signals.
Lightweight, zero-cost, zero-config alternative to TimescaleDB.
"""

from __future__ import annotations

import aiosqlite
import json
import logging
import time
from typing import Optional

from orderflow_system.data.models import (
    BarType, InfoBar, Tick, Side, Signal, SignalType, VolumeProfileResult,
)

logger = logging.getLogger(__name__)


class Database:
    """Async SQLite database for orderflow data storage."""

    def __init__(self, db_path: str = "orderflow_data.db"):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self):
        self._db = await aiosqlite.connect(self.db_path)
        try:
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA synchronous=NORMAL")
            await self._create_tables()
        except BaseException:
            await self.close()
            raise
        logger.info(f"Database connected: {self.db_path}")

    async def commit(self):
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("Database closed")

    async def _create_tables(self):
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                price REAL NOT NULL,
                size REAL NOT NULL,
                side TEXT NOT NULL,
                trade_id TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_ticks_instrument_ts
                ON ticks(instrument, timestamp_ms);

            CREATE TABLE IF NOT EXISTS info_bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument TEXT NOT NULL,
                bar_type TEXT NOT NULL,
                seq INTEGER NOT NULL,            -- cumulative trade index at bar close
                timestamp_ms INTEGER NOT NULL,   -- bar CLOSE time
                open REAL, high REAL, low REAL, close REAL,
                volume REAL,
                buy_volume REAL,
                sell_volume REAL,
                delta REAL,
                tick_count INTEGER,
                dollar_value REAL,
                threshold REAL,
                duration_ms INTEGER,             -- close-to-close since previous bar (NULL = unknown)
                data_gap INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_info_bars_ts
                ON info_bars(instrument, bar_type, timestamp_ms);

            CREATE TABLE IF NOT EXISTS volume_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument TEXT NOT NULL,
                session_date TEXT NOT NULL,
                poc REAL, vah REAL, val REAL,
                total_volume REAL,
                shape TEXT,
                poc_position_pct REAL,
                lvn_json TEXT,
                volume_at_price_json TEXT,
                source TEXT NOT NULL DEFAULT 'ticks',
                bar_type TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_vp_instrument_date
                ON volume_profiles(instrument, session_date);

            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                signal_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                price_level REAL,
                strength REAL,
                details_json TEXT,
                bar_type TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_signals_instrument_ts
                ON signals(instrument, timestamp_ms);

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                dataset_path TEXT NOT NULL,
                version TEXT NOT NULL,
                bar_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                start_ms INTEGER,
                end_ms INTEGER,
                config_digest TEXT NOT NULL,
                created_utc_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_time_ms INTEGER,
                exit_time_ms INTEGER,
                entry_price REAL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                pnl_ticks REAL,
                rr_ratio REAL,
                signals_json TEXT,
                notes TEXT
            );
        """)
        await self._migrate()
        await self._db.commit()

    async def _migrate(self):
        """Preserve historical rows and partition new replay data by provenance."""
        for table, column, ddl in (
            ("volume_profiles", "source", "TEXT NOT NULL DEFAULT 'ticks'"),
            ("volume_profiles", "bar_type", "TEXT NOT NULL DEFAULT ''"),
            ("signals", "bar_type", "TEXT NOT NULL DEFAULT ''"),
            ("info_bars", "duration_ms", "INTEGER"),
            ("info_bars", "data_gap", "INTEGER NOT NULL DEFAULT 0"),
            ("info_bars", "dataset_id", "TEXT NOT NULL DEFAULT 'legacy'"),
            ("signals", "run_id", "TEXT NOT NULL DEFAULT ''"),
            ("signals", "bar_seq", "INTEGER"),
            ("volume_profiles", "run_id", "TEXT NOT NULL DEFAULT ''"),
            ("volume_profiles", "complete", "INTEGER NOT NULL DEFAULT 0"),
        ):
            cur = await self._db.execute(f"PRAGMA table_info({table})")
            if column not in {row[1] for row in await cur.fetchall()}:
                await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        for table in ("signals", "volume_profiles"):
            await self._db.execute(
                f"UPDATE {table} SET run_id = 'legacy:' || bar_type WHERE run_id = ''"
            )
        # These older keys can reject valid overlapping runs on a reconnect.
        for name in ("idx_signals_unique", "idx_vp_unique", "idx_info_bars_key"):
            await self._db.execute(f"DROP INDEX IF EXISTS {name}")
        await self._db.executescript("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_run_unique
                ON signals(run_id, bar_seq, signal_type, direction, price_level)
                WHERE run_id NOT LIKE 'legacy%';
            CREATE UNIQUE INDEX IF NOT EXISTS idx_vp_run_unique
                ON volume_profiles(run_id, session_date)
                WHERE run_id NOT LIKE 'legacy%';
            CREATE UNIQUE INDEX IF NOT EXISTS idx_info_bars_dataset_key
                ON info_bars(dataset_id, instrument, bar_type, seq)
                WHERE dataset_id <> 'legacy';
        """)

    async def create_run(
        self, run_id: str, dataset_id: str, dataset_path: str, version: str,
        bar_type: BarType, symbol: str, start_ms: Optional[int], end_ms: Optional[int],
        config_digest: str,
    ):
        await self._db.execute(
            "INSERT OR IGNORE INTO runs "
            "(run_id, dataset_id, dataset_path, version, bar_type, symbol, start_ms, "
            "end_ms, config_digest, created_utc_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, dataset_id, dataset_path, version, bar_type.value, symbol,
             start_ms, end_ms, config_digest, time.time_ns() // 1_000_000),
        )
        await self._db.commit()

    async def _read_run(self, instrument: str, bar_type: Optional[BarType], run_id: Optional[str]):
        if run_id is not None:
            return run_id
        cur = await self._db.execute(
            "SELECT run_id FROM runs WHERE symbol = ? AND bar_type = ? "
            "ORDER BY created_utc_ms DESC, rowid DESC LIMIT 1",
            (instrument, bar_type.value if bar_type else ""),
        )
        row = await cur.fetchone()
        if row:
            return row[0]
        default_run = self._default_run(bar_type)
        cur = await self._db.execute(
            "SELECT 1 FROM signals WHERE run_id = ? UNION ALL "
            "SELECT 1 FROM volume_profiles WHERE run_id = ? LIMIT 1",
            (default_run, default_run),
        )
        return default_run if await cur.fetchone() else "legacy:" + (bar_type.value if bar_type else "")

    @staticmethod
    def _default_run(bar_type: Optional[BarType]) -> str:
        return "default:" + (bar_type.value if bar_type else "")

    # ── Ticks ──

    async def insert_tick(self, instrument: str, tick: Tick):
        await self._db.execute(
            "INSERT INTO ticks (instrument, timestamp_ms, price, size, side, trade_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (instrument, tick.timestamp_ms, tick.price, tick.size,
             tick.side.value, tick.trade_id),
        )

    async def insert_ticks_batch(self, instrument: str, ticks: list[Tick]):
        data = [
            (instrument, t.timestamp_ms, t.price, t.size, t.side.value, t.trade_id)
            for t in ticks
        ]
        await self._db.executemany(
            "INSERT INTO ticks (instrument, timestamp_ms, price, size, side, trade_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            data,
        )
        await self._db.commit()

    async def get_ticks(
        self, instrument: str, start_ms: int, end_ms: int
    ) -> list[Tick]:
        cursor = await self._db.execute(
            "SELECT timestamp_ms, price, size, side, trade_id FROM ticks "
            "WHERE instrument = ? AND timestamp_ms >= ? AND timestamp_ms <= ? "
            "ORDER BY timestamp_ms",
            (instrument, start_ms, end_ms),
        )
        rows = await cursor.fetchall()
        return [
            Tick(
                timestamp_ms=r[0], price=r[1], size=r[2],
                side=Side(r[3]), trade_id=r[4] or ""
            )
            for r in rows
        ]

    # ── Info bars ──

    async def insert_bars_batch(
        self, instrument: str, bar_type: BarType, bars: list[InfoBar],
        dataset_id: str = "default",
    ):
        """Idempotent within one dataset identity."""
        await self._db.executemany(
            "INSERT OR IGNORE INTO info_bars "
            "(dataset_id, instrument, bar_type, seq, timestamp_ms, open, high, low, close, volume, "
            "buy_volume, sell_volume, delta, tick_count, dollar_value, threshold, "
            "duration_ms, data_gap) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (dataset_id, instrument, bar_type.value, b.seq, b.timestamp_ms, b.open, b.high, b.low,
                 b.close, b.volume, b.buy_volume, b.sell_volume, b.delta, b.tick_count,
                 b.dollar_value, b.threshold, b.duration_ms, int(b.data_gap))
                for b in bars
            ],
        )
        await self._db.commit()

    async def get_bars(
        self, instrument: str, bar_type: BarType, start_ms: int, end_ms: int,
        run_id: Optional[str] = None, dataset_id: Optional[str] = None,
    ) -> list[InfoBar]:
        """Bars with CLOSE time in [start_ms, end_ms], oldest first."""
        run_start, run_end = None, None
        if run_id is not None or dataset_id is None:
            selected_run = await self._read_run(instrument, bar_type, run_id)
            cur = await self._db.execute(
                "SELECT dataset_id, start_ms, end_ms FROM runs WHERE run_id = ? "
                "AND symbol = ? AND bar_type = ?", (selected_run, instrument, bar_type.value),
            )
            row = await cur.fetchone()
            if row:
                if dataset_id is not None and dataset_id != row[0]:
                    return []
                dataset_id, run_start, run_end = row
            elif run_id is not None:
                return []
            else:
                cur = await self._db.execute(
                    "SELECT 1 FROM info_bars WHERE dataset_id = 'default' "
                    "AND instrument = ? AND bar_type = ? LIMIT 1", (instrument, bar_type.value),
                )
                dataset_id = "default" if await cur.fetchone() else "legacy"
        cursor = await self._db.execute(
            "SELECT timestamp_ms, open, high, low, close, volume, buy_volume, "
            "sell_volume, tick_count, dollar_value, threshold, seq, duration_ms, data_gap "
            "FROM info_bars "
            "WHERE dataset_id = ? AND instrument = ? AND bar_type = ? "
            "AND timestamp_ms >= ? AND timestamp_ms <= ? "
            "AND (? IS NULL OR timestamp_ms >= ?) "
            "AND (? IS NULL OR timestamp_ms < ?) "
            "ORDER BY timestamp_ms, seq",
            (dataset_id, instrument, bar_type.value, start_ms, end_ms,
             run_start, run_start, run_end, run_end),
        )
        rows = await cursor.fetchall()
        return [
            InfoBar(
                timestamp_ms=r[0], open=r[1], high=r[2], low=r[3], close=r[4],
                volume=r[5], buy_volume=r[6], sell_volume=r[7], tick_count=r[8],
                dollar_value=r[9], threshold=r[10], seq=r[11], bar_type=bar_type,
                duration_ms=r[12], data_gap=bool(r[13]),
            )
            for r in rows
        ]

    # ── Volume Profiles ──

    async def insert_volume_profile(
        self, instrument: str, vp: VolumeProfileResult, bar_type: Optional[BarType] = None,
        run_id: Optional[str] = None, complete: bool = True,
    ):
        run_id = run_id or self._default_run(bar_type)
        # Never downgrade a complete profile when replaying a partial session.
        await self._db.execute(
            "INSERT INTO volume_profiles "
            "(instrument, session_date, poc, vah, val, total_volume, shape, "
            "poc_position_pct, lvn_json, volume_at_price_json, source, bar_type, run_id, complete) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id, session_date) WHERE run_id NOT LIKE 'legacy%' "
            "DO UPDATE SET poc=excluded.poc, vah=excluded.vah, val=excluded.val, "
            "total_volume=excluded.total_volume, shape=excluded.shape, "
            "poc_position_pct=excluded.poc_position_pct, lvn_json=excluded.lvn_json, "
            "volume_at_price_json=excluded.volume_at_price_json, source=excluded.source, "
            "complete=excluded.complete WHERE excluded.complete >= volume_profiles.complete",
            (instrument, vp.session_date, vp.poc, vp.vah, vp.val,
             vp.total_volume, vp.shape, vp.poc_position_pct,
             json.dumps(vp.lvn_levels),
             json.dumps({str(k): v for k, v in vp.volume_at_price.items()}),
             vp.source, bar_type.value if bar_type else "", run_id, int(complete)),
        )
        await self._db.commit()

    async def get_volume_profiles(
        self, instrument: str, days: int = 5, bar_type: Optional[BarType] = None,
        run_id: Optional[str] = None,
    ) -> list[VolumeProfileResult]:
        run_id = await self._read_run(instrument, bar_type, run_id)
        cursor = await self._db.execute(
            "SELECT session_date, poc, vah, val, total_volume, shape, "
            "poc_position_pct, lvn_json, volume_at_price_json, source "
            "FROM volume_profiles WHERE instrument = ? AND bar_type = ? AND run_id = ? "
            "ORDER BY session_date DESC LIMIT ?",
            (instrument, bar_type.value if bar_type else "", run_id, days),
        )
        rows = await cursor.fetchall()
        results = []
        for r in rows:
            vap_raw = json.loads(r[8]) if r[8] else {}
            results.append(VolumeProfileResult(
                session_date=r[0], poc=r[1], vah=r[2], val=r[3],
                total_volume=r[4], shape=r[5], poc_position_pct=r[6],
                lvn_levels=json.loads(r[7]) if r[7] else [],
                volume_at_price={float(k): v for k, v in vap_raw.items()},
                source=r[9],
            ))
        return list(reversed(results))  # Oldest first

    # ── Signals ──

    async def insert_signal(
        self, instrument: str, signal: Signal, bar_type: Optional[BarType] = None,
        commit: bool = True, run_id: Optional[str] = None, bar_seq: Optional[int] = None,
    ):
        run_id = run_id or self._default_run(bar_type)
        # Older callers have no bar sequence; preserve their timestamp identity.
        bar_seq = signal.details.get("bar_seq", signal.timestamp_ms) if bar_seq is None else bar_seq
        await self._db.execute(
            "INSERT OR IGNORE INTO signals "
            "(instrument, timestamp_ms, signal_type, direction, price_level, "
            "strength, details_json, bar_type, run_id, bar_seq) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (instrument, signal.timestamp_ms, signal.signal_type.value,
             signal.direction.value, signal.price_level, signal.strength,
             json.dumps(signal.details), bar_type.value if bar_type else "", run_id, bar_seq),
        )
        if commit:
            await self._db.commit()

    async def get_signals(
        self, instrument: str, bar_type: Optional[BarType] = None,
        run_id: Optional[str] = None,
    ) -> list[Signal]:
        run_id = await self._read_run(instrument, bar_type, run_id)
        cursor = await self._db.execute(
            "SELECT timestamp_ms, signal_type, direction, price_level, strength, details_json, bar_seq "
            "FROM signals WHERE instrument = ? AND bar_type = ? AND run_id = ? "
            "ORDER BY timestamp_ms, bar_seq, id",
            (instrument, bar_type.value if bar_type else "", run_id),
        )
        results = []
        for row in await cursor.fetchall():
            details = json.loads(row[5]) if row[5] else {}
            details["bar_seq"] = row[6]
            results.append(Signal(timestamp_ms=row[0], signal_type=SignalType(row[1]),
                                  direction=Side(row[2]), price_level=row[3], strength=row[4],
                                  details=details))
        return results

    # ── Trade Journal ──

    async def log_trade(
        self,
        instrument: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        stop_loss: float,
        take_profit: float,
        pnl_ticks: float,
        rr_ratio: float,
        signals: list[Signal],
        notes: str = "",
        entry_time_ms: int = 0,
        exit_time_ms: int = 0,
    ):
        signals_json = json.dumps([
            {"type": s.signal_type.value, "strength": s.strength,
             "price": s.price_level, "ts": s.timestamp_ms}
            for s in signals
        ])
        await self._db.execute(
            "INSERT INTO trade_journal "
            "(instrument, direction, entry_time_ms, exit_time_ms, entry_price, "
            "exit_price, stop_loss, take_profit, pnl_ticks, rr_ratio, "
            "signals_json, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (instrument, direction, entry_time_ms, exit_time_ms,
             entry_price, exit_price, stop_loss, take_profit,
             pnl_ticks, rr_ratio, signals_json, notes),
        )
        await self._db.commit()

"""Replay provenance, lossless schema migration and I/O lifecycle regressions."""

import asyncio
import sqlite3
import threading

import pytest

from orderflow_system.config.settings import DASHBOARD, InfoBarConfig
from orderflow_system.data.database import Database
from orderflow_system.data.models import BarType, Signal, SignalType, Side, VolumeProfileResult
from orderflow_system.main import OrderflowSystem
from test_pipeline import synth_bars


BT = BarType.DOLLAR_IMBALANCE


def old_database(path, current):
    con = sqlite3.connect(path)
    bar_column = ", bar_type TEXT NOT NULL DEFAULT ''" if current else ""
    con.executescript(f"""
        CREATE TABLE signals (
            id INTEGER PRIMARY KEY, instrument TEXT NOT NULL, timestamp_ms INTEGER NOT NULL,
            signal_type TEXT NOT NULL, direction TEXT NOT NULL, price_level REAL,
            strength REAL, details_json TEXT {bar_column});
        CREATE TABLE volume_profiles (
            id INTEGER PRIMARY KEY, instrument TEXT NOT NULL, session_date TEXT NOT NULL,
            poc REAL, vah REAL, val REAL, total_volume REAL, shape TEXT,
            poc_position_pct REAL, lvn_json TEXT, volume_at_price_json TEXT {bar_column});
    """)
    types = [BT.value, BarType.TICK_RUN.value] if current else [None, None]
    for index, bt in enumerate(types):
        suffix = ",bar_type" if current else ""
        placeholders = ",?" if current else ""
        params = [index + 1, "BTCUSDT", 1000, "absorption", "buy", 100, 80, "{}"]
        if current:
            params.append(bt)
        con.execute(f"INSERT INTO signals (id,instrument,timestamp_ms,signal_type,direction,"
                    f"price_level,strength,details_json{suffix}) VALUES (?,?,?,?,?,?,?,?{placeholders})", params)
        params = [index + 1, "BTCUSDT", "2023-01-01", 100, 101, 99, 10, "d_shape", 50, "[]", "{}"]
        if current:
            params.append(bt)
        con.execute(f"INSERT INTO volume_profiles (id,instrument,session_date,poc,vah,val,total_volume,"
                    f"shape,poc_position_pct,lvn_json,volume_at_price_json{suffix}) "
                    f"VALUES (?,?,?,?,?,?,?,?,?,?,?{placeholders})", params)
    if not current:
        con.executescript("""
            CREATE TABLE candles (id INTEGER PRIMARY KEY, instrument TEXT,
                timestamp_ms INTEGER, open REAL, high REAL, low REAL, close REAL);
            INSERT INTO candles VALUES (1, 'BTCUSDT', 1000, 100, 101, 99, 100);
            INSERT INTO candles VALUES (2, 'BTCUSDT', 1000, 100, 101, 99, 100);
        """)
    if current:
        con.executescript("""
            CREATE TABLE info_bars (
                id INTEGER PRIMARY KEY, instrument TEXT NOT NULL, bar_type TEXT NOT NULL,
                seq INTEGER NOT NULL, timestamp_ms INTEGER NOT NULL, open REAL, high REAL,
                low REAL, close REAL, volume REAL, buy_volume REAL, sell_volume REAL,
                delta REAL, tick_count INTEGER, dollar_value REAL, threshold REAL,
                duration_ms INTEGER, data_gap INTEGER NOT NULL DEFAULT 0);
            INSERT INTO info_bars (id, instrument, bar_type, seq, timestamp_ms,
                open, high, low, close, volume, buy_volume, sell_volume, delta,
                tick_count, dollar_value, threshold, duration_ms, data_gap)
                VALUES (1, 'BTCUSDT', 'fixed_dollar_imbalance', 1, 1000,
                        100, 101, 99, 100, 10, 6, 4, 2, 100, 1000, 1000, 1000, 0);
            CREATE UNIQUE INDEX idx_info_bars_key ON info_bars(instrument,bar_type,seq);
            CREATE UNIQUE INDEX idx_signals_unique ON signals(instrument,bar_type,timestamp_ms,
                signal_type,direction,price_level);
            CREATE UNIQUE INDEX idx_vp_unique ON volume_profiles(instrument,bar_type,session_date);
        """)
    con.commit()
    con.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("current", [False, True], ids=["historical", "previous-info-bar"])
async def test_migration_preserves_duplicates_overlapping_runs_and_restarts(tmp_path, current):
    path = tmp_path / "old.db"
    old_database(path, current)
    db = Database(str(path))
    await db.connect()
    for table in ("signals", "volume_profiles"):
        rows = await (await db._db.execute(f"SELECT run_id FROM {table}")).fetchall()
        assert len(rows) == 2
        assert all(row[0].startswith("legacy:") for row in rows)
    indexes = {r[0] for r in await (await db._db.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    )).fetchall()}
    assert not indexes & {"idx_info_bars_key", "idx_signals_unique", "idx_vp_unique"}
    assert {"idx_info_bars_dataset_key", "idx_signals_run_unique", "idx_vp_run_unique"} <= indexes
    for table, required in (("signals", {"run_id", "bar_seq"}),
                            ("volume_profiles", {"run_id", "complete"}),
                            ("info_bars", {"dataset_id"})):
        columns = {r[1] for r in await (await db._db.execute(f"PRAGMA table_info({table})")).fetchall()}
        assert required <= columns
    bars = synth_bars(2)
    bars[1].timestamp_ms = bars[0].timestamp_ms
    if current:
        historical_bars = await db.get_bars("BTCUSDT", BT, 0, 10000)
        assert len(historical_bars) == 1 and historical_bars[0].seq == 1
    historical_type = BT if current else None
    assert len(await db.get_signals("BTCUSDT", historical_type)) == (1 if current else 2)
    assert len(await db.get_volume_profiles("BTCUSDT", bar_type=historical_type)) == (1 if current else 2)
    for index in (1, 2):
        run, dataset = f"run-{index}", f"dataset-{index}"
        await db.create_run(run, dataset, f"/read-only/{index}.parquet", "stamp", BT,
                            "BTCUSDT", bars[0].timestamp_ms, bars[-1].timestamp_ms + 1, "config")
        for _ in range(2):
            await db.insert_bars_batch("BTCUSDT", BT, bars, dataset_id=dataset)
            for bar in bars:
                sig = Signal(bar.timestamp_ms, SignalType.ABSORPTION, Side.BUY, bar.close)
                await db.insert_signal("BTCUSDT", sig, BT, run_id=run, bar_seq=bar.seq)
            vp = VolumeProfileResult("2023-01-01", 100, 101, 99, total_volume=10)
            await db.insert_volume_profile("BTCUSDT", vp, BT, run_id=run)
        assert len(await db.get_signals("BTCUSDT", BT, run_id=run)) == 2
        assert len(await db.get_bars("BTCUSDT", BT, 0, 10**15, run_id=run)) == 2
    assert len(await db.get_signals("BTCUSDT", BT)) == 2
    for _ in range(2):
        await db.close()
        await db.connect()
        for table, count in (("signals", 6), ("volume_profiles", 4), ("info_bars", 5 if current else 4), ("runs", 2)):
            got = await (await db._db.execute(f"SELECT COUNT(*) FROM {table}")).fetchone()
            assert got[0] == count
        if not current:
            assert (await (await db._db.execute("SELECT COUNT(*) FROM candles")).fetchone())[0] == 2
    await db.close()


@pytest.mark.asyncio
async def test_partial_profile_cannot_replace_complete_profile(tmp_path):
    db = Database(str(tmp_path / "profiles.db"))
    await db.connect()
    full = VolumeProfileResult("2023-01-01", 100, 101, 99, total_volume=10)
    partial = VolumeProfileResult("2023-01-01", 200, 201, 199, total_volume=1)
    await db.insert_volume_profile("BTCUSDT", partial, BT, run_id="run", complete=False)
    await db.insert_volume_profile("BTCUSDT", full, BT, run_id="run", complete=True)
    await db.insert_volume_profile("BTCUSDT", partial, BT, run_id="run", complete=False)
    got = await db.get_volume_profiles("BTCUSDT", bar_type=BT, run_id="run")
    assert len(got) == 1 and got[0].poc == 100
    assert (await (await db._db.execute("SELECT complete FROM volume_profiles")).fetchone())[0] == 1
    await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["_create_tables", "_migrate"])
async def test_connect_failure_closes_allocated_worker(tmp_path, monkeypatch, method):
    db = Database(str(tmp_path / "failure.db"))
    connection = []

    async def fail():
        connection.append(db._db)
        raise RuntimeError("schema failure")

    monkeypatch.setattr(db, method, fail)
    with pytest.raises(RuntimeError, match="schema failure"):
        await db.connect()
    assert db._db is None
    worker = getattr(connection[0], "_thread", connection[0])
    worker.join(timeout=1)
    assert not worker.is_alive()


def sqlite_workers():
    return {thread for thread in threading.enumerate()
            if "_connection_worker_thread" in repr(thread)}


@pytest.mark.asyncio
async def test_missing_dataset_never_allocates_database_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(DASHBOARD, "enabled", False)
    baseline = sqlite_workers()
    system = OrderflowSystem(info_bars=InfoBarConfig(root=str(tmp_path / "absent"), persist=True))
    system.db = Database(str(tmp_path / "missing.db"))
    with pytest.raises(FileNotFoundError):
        await asyncio.wait_for(system.start(), 2)
    assert system.db._db is None
    assert sqlite_workers() == baseline


@pytest.mark.asyncio
async def test_failed_flush_still_closes_database(dataset_root, tmp_path, monkeypatch):
    monkeypatch.setattr(DASHBOARD, "enabled", False)
    system = OrderflowSystem(info_bars=InfoBarConfig(root=str(dataset_root), persist=True))
    system.db = Database(str(tmp_path / "flush.db"))

    async def fail():
        raise RuntimeError("flush failure")

    monkeypatch.setattr(system, "_flush_bars", fail)
    with pytest.raises(RuntimeError, match="flush failure"):
        await system.start()
    assert system.db._db is None
    assert all(task.done() for task in system._tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_child", ["dashboard", "replay", "stop"])
async def test_child_failure_joins_other_child_before_database_close(
    dataset_root, tmp_path, monkeypatch, failing_child,
):
    import uvicorn

    monkeypatch.setattr(DASHBOARD, "enabled", True)
    system = OrderflowSystem(info_bars=InfoBarConfig(
        root=str(dataset_root), persist=True, bars_per_second=1000,
    ))
    system.db = Database(str(tmp_path / "children.db"))
    replay_active = asyncio.Event()
    server_active = asyncio.Event()
    replay_finished = asyncio.Event()
    server_finished = asyncio.Event()
    original_on_bar = system._on_bar

    async def on_bar(bar):
        replay_active.set()
        await server_active.wait()
        try:
            if failing_child == "replay":
                raise RuntimeError("replay failure")
            await original_on_bar(bar)
            await asyncio.Event().wait()
        finally:
            replay_finished.set()

    class FakeServer:
        def __init__(self, config):
            self.should_exit = False

        async def serve(self):
            server_active.set()
            try:
                await replay_active.wait()
                if failing_child == "dashboard":
                    raise RuntimeError("dashboard failure")
                await asyncio.Event().wait()
            finally:
                server_finished.set()

    original_close = system.db.close

    async def close():
        assert replay_finished.is_set() and server_finished.is_set()
        assert all(task.done() for task in system._tasks)
        await original_close()

    monkeypatch.setattr(system, "_on_bar", on_bar)
    monkeypatch.setattr(system.db, "close", close)
    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    if failing_child == "stop":
        running = asyncio.create_task(system.start())
        await asyncio.wait_for(replay_active.wait(), 2)
        await asyncio.wait_for(server_active.wait(), 2)
        await system.stop()
        await asyncio.wait_for(running, 2)
    else:
        with pytest.raises(RuntimeError, match=f"{failing_child} failure"):
            await asyncio.wait_for(system.start(), 2)
    assert system.db._db is None
    assert all(task.done() for task in system._tasks)


@pytest.mark.asyncio
async def test_replay_run_identity_is_deterministic_and_first_profile_partial(tmp_path, monkeypatch):
    from conftest import make_rows, write_dataset

    monkeypatch.setattr(DASHBOARD, "enabled", False)
    write_dataset(tmp_path, make_rows(5, t0="2023-01-01T23:58:00", step_s=60))
    path = tmp_path / "runs.db"
    ids = []
    for _ in range(2):
        system = OrderflowSystem(info_bars=InfoBarConfig(root=str(tmp_path), start="2023-01-01T23:59:00", persist=True))
        system.db = Database(str(path))
        await system.start()
        ids.append(system.run_id)
    assert ids[0] == ids[1]
    con = sqlite3.connect(path)
    assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM info_bars").fetchone()[0] == 4
    assert con.execute("SELECT complete FROM volume_profiles").fetchall() == [(0,)]
    con.close()


@pytest.mark.asyncio
async def test_bar_reads_respect_explicit_run_identity_and_window(tmp_path):
    db = Database(str(tmp_path / "windows.db"))
    await db.connect()
    bars = synth_bars(4)
    await db.insert_bars_batch("BTCUSDT", BT, bars, dataset_id="shared-dataset")
    await db.create_run("first", "shared-dataset", "/shared.parquet", "stamp", BT,
                        "BTCUSDT", bars[0].timestamp_ms, bars[2].timestamp_ms, "config")
    await db.create_run("second", "shared-dataset", "/shared.parquet", "stamp", BT,
                        "BTCUSDT", bars[2].timestamp_ms, None, "config")
    assert [bar.seq for bar in await db.get_bars("BTCUSDT", BT, 0, 10**15, run_id="first")] == [b.seq for b in bars[:2]]
    assert [bar.seq for bar in await db.get_bars("BTCUSDT", BT, 0, 10**15)] == [b.seq for b in bars[2:]]
    assert await db.get_bars("BTCUSDT", BT, 0, 10**15, run_id="unknown") == []
    assert await db.get_bars("BTCUSDT", BT, 0, 10**15, run_id="first", dataset_id="other") == []
    assert len(await db.get_bars("BTCUSDT", BT, 0, 10**15, dataset_id="shared-dataset")) == 4
    await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario,expected", [("midnight", 1), ("noon", 0), ("first-file-bar", 1)])
async def test_first_profile_completeness_uses_accepted_predecessor(tmp_path, monkeypatch, scenario, expected):
    from conftest import make_rows, write_dataset
    from orderflow_system.data.infobar_feed import parse_ts_ms

    monkeypatch.setattr(DASHBOARD, "enabled", False)
    rows = make_rows(4)
    if scenario == "midnight":
        closes = ["2022-12-31T23:59:55", "2023-01-01T00:00:05", "2023-01-01T12:00:05", "2023-01-02T00:00:05"]
        start = "2023-01-01T00:00:00"
    elif scenario == "noon":
        closes = ["2023-01-01T00:00:05", "2023-01-01T11:59:55", "2023-01-01T12:00:05", "2023-01-02T00:00:05"]
        start = "2023-01-01T12:00:00"
    else:
        closes = ["2023-01-01T12:00:05", "2023-01-01T13:00:05", "2023-01-01T14:00:05", "2023-01-02T00:00:05"]
        start = None
    for row, close in zip(rows, closes):
        row["ts"] = parse_ts_ms(close)
    write_dataset(tmp_path, rows, row_group_size=1)
    path = tmp_path / "complete.db"
    system = OrderflowSystem(info_bars=InfoBarConfig(root=str(tmp_path), start=start, persist=True))
    system.db = Database(str(path))
    await system.start()
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT session_date,complete FROM volume_profiles").fetchall() == [("2023-01-01", expected)]
    expected_predecessor = None if scenario == "first-file-bar" else rows[0 if scenario == "midnight" else 1]["ts"]
    assert system.reader.first_predecessor_ms == expected_predecessor


@pytest.mark.asyncio
async def test_run_and_all_bar_batches_keep_discovered_dataset_identity(tmp_path, monkeypatch):
    import os
    from conftest import make_rows, write_dataset
    from orderflow_system.data.infobar_feed import find_dataset
    import orderflow_system.main as main_module

    monkeypatch.setattr(DASHBOARD, "enabled", False)
    monkeypatch.setattr(main_module, "_DB_FLUSH_BARS", 2)
    write_dataset(tmp_path, make_rows(6))
    info = find_dataset(tmp_path, "BTCUSDT", BT)
    identity = info.dataset_id
    system = OrderflowSystem(info_bars=InfoBarConfig(root=str(tmp_path), persist=True))
    path = tmp_path / "frozen-id.db"
    system.db = Database(str(path))
    original_on_bar = system._on_bar
    changed = False

    async def change_metadata_once(bar):
        nonlocal changed
        if not changed:
            stat = info.parquet_path.stat()
            os.utime(info.parquet_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            changed = True
        await original_on_bar(bar)

    monkeypatch.setattr(system, "_on_bar", change_metadata_once)
    await system.start()
    assert changed
    assert find_dataset(tmp_path, "BTCUSDT", BT).dataset_id != identity
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT dataset_id FROM runs").fetchall() == [(identity,)]
        assert con.execute("SELECT DISTINCT dataset_id FROM info_bars").fetchall() == [(identity,)]
        assert con.execute("SELECT COUNT(*) FROM info_bars").fetchone()[0] == 6

import pytest

from orderflow_system.data.infobar_feed import (
    DataIntegrityError, find_dataset, open_bars, InfoBarReader, InfoBarReplayFeed,
)
from orderflow_system.data.models import BarType

from conftest import make_rows, write_dataset

ROOT_ARGS = ("BTCUSDT", BarType.DOLLAR_IMBALANCE)


def test_reads_all_bars_in_order(dataset_root):
    r = open_bars(dataset_root, *ROOT_ARGS)
    bars = list(r)
    assert len(bars) == 100
    assert [b.seq for b in bars] == sorted(b.seq for b in bars)
    assert bars[0].duration_ms is None and bars[1].duration_ms == 30_000
    assert bars[0].bar_type == BarType.DOLLAR_IMBALANCE
    assert r.report.rows_yielded == 100 and r.report.rows_rejected == 0


def test_bar_fields_map_to_columns(dataset_root):
    rows = make_rows(100)
    b = next(iter(open_bars(dataset_root, *ROOT_ARGS)))
    assert b.timestamp_ms == rows[0]["ts"]
    assert (b.open, b.high, b.low, b.close) == (rows[0]["o"], rows[0]["h"], rows[0]["l"], rows[0]["c"])
    assert b.volume == rows[0]["vol"] and b.buy_volume == rows[0]["buy"]
    assert b.tick_count == rows[0]["nt"] and b.seq == rows[0]["seq"]
    assert b.delta == pytest.approx(rows[0]["buy"] - rows[0]["sell"])


def test_time_window_is_start_inclusive_end_exclusive(dataset_root):
    all_bars = list(open_bars(dataset_root, *ROOT_ARGS))
    start, end = all_bars[10].timestamp_ms, all_bars[20].timestamp_ms
    iso = lambda ms: __import__("datetime").datetime.fromtimestamp(ms / 1000, tz=__import__("datetime").timezone.utc).isoformat()
    got = list(open_bars(dataset_root, *ROOT_ARGS, start=iso(start), end=iso(end)))
    assert [b.seq for b in got] == [b.seq for b in all_bars[10:20]]
    assert got[0].duration_ms == all_bars[10].duration_ms  # preceding close is past context


def test_row_group_pruning_skips_outside_groups(dataset_root):
    all_bars = list(open_bars(dataset_root, *ROOT_ARGS))
    r = open_bars(dataset_root, *ROOT_ARGS, start="2023-01-01T00:40:00")   # late in the file
    list(r)
    assert r.report.row_groups_skipped >= 1


def test_naive_dates_are_utc(dataset_root):
    a = list(open_bars(dataset_root, *ROOT_ARGS, start="2023-01-01T00:10:00"))
    b = list(open_bars(dataset_root, *ROOT_ARGS, start="2023-01-01T00:10:00+00:00"))
    assert [x.seq for x in a] == [x.seq for x in b] and a


def test_newest_version_wins_and_can_be_pinned(tmp_path):
    write_dataset(tmp_path, make_rows(5), stamp="20260101T000000Z")
    write_dataset(tmp_path, make_rows(9), stamp="20260202T000000Z")
    assert find_dataset(tmp_path, "BTCUSDT", BarType.DOLLAR_IMBALANCE).version == "20260202T000000Z"
    assert len(list(open_bars(tmp_path, *ROOT_ARGS))) == 9
    assert len(list(open_bars(tmp_path, *ROOT_ARGS, version="20260101T000000Z"))) == 5


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_dataset(tmp_path, "BTCUSDT", BarType.TICK_RUN)


def test_sidecar_bar_type_mismatch_rejected(tmp_path):
    d = write_dataset(tmp_path, make_rows(5), bar_type=BarType.DOLLAR_IMBALANCE)
    other = tmp_path / "BTCUSDT" / BarType.TICK_RUN.value
    other.mkdir(parents=True)
    for f in d.iterdir():
        (other / f.name).write_bytes(f.read_bytes())    # sidecar still says dollar_imbalance
    with pytest.raises(DataIntegrityError):
        find_dataset(tmp_path, "BTCUSDT", BarType.TICK_RUN)


def _corrupt(rows, i, **changes):
    rows = [dict(r) for r in rows]
    rows[i].update(changes)
    return rows


@pytest.mark.parametrize("changes,reason", [
    (dict(h=1.0), "inconsistent_ohlc"),
    (dict(vol=-1.0), "bad_volume"),
    (dict(buy=0.0, sell=0.0), "bad_volume"),            # buy+sell != vol
    (dict(nt=0), "non_positive_ticks"),
    (dict(c=float("nan")), "non_finite"),
])
def test_bad_rows_strict_raises_and_lenient_drops(tmp_path, changes, reason):
    rows = _corrupt(make_rows(30), 7, **changes)
    write_dataset(tmp_path, rows)
    with pytest.raises(DataIntegrityError):
        list(open_bars(tmp_path, *ROOT_ARGS, strict=True))
    r = open_bars(tmp_path, *ROOT_ARGS, strict=False)
    assert len(list(r)) == 29
    assert r.report.reject_reasons[reason] == 1


def test_out_of_order_rows_rejected(tmp_path):
    rows = make_rows(30)
    rows[10]["ts"] = rows[5]["ts"]                       # time goes backwards
    write_dataset(tmp_path, rows)
    with pytest.raises(DataIntegrityError):
        list(open_bars(tmp_path, *ROOT_ARGS))
    r = open_bars(tmp_path, *ROOT_ARGS, strict=False)
    list(r)
    assert r.report.reject_reasons["out_of_order"] >= 1


def test_tied_timestamps_allowed_ordered_by_seq(tmp_path):
    rows = make_rows(30)
    rows[11]["ts"] = rows[10]["ts"]                      # two bars closing in the same ms
    write_dataset(tmp_path, rows)
    r = open_bars(tmp_path, *ROOT_ARGS)
    bars = list(r)
    assert len(bars) == 30 and r.report.tied_timestamps == 1
    assert bars[11].duration_ms == 0 and bars[11].seq > bars[10].seq


def test_duplicate_or_decreasing_seq_rejected(tmp_path):
    rows = make_rows(30)
    rows[12]["seq"] = rows[11]["seq"]
    write_dataset(tmp_path, rows)
    with pytest.raises(DataIntegrityError):
        list(open_bars(tmp_path, *ROOT_ARGS))


def test_bar_spanning_missing_hour_is_flagged(tmp_path):
    rows = make_rows(30, step_s=1800)                    # 30 min bars
    write_dataset(tmp_path, rows, missing_hours=["2023-01-01T03:00:00+00:00"])
    r = open_bars(tmp_path, *ROOT_ARGS)
    bars = list(r)
    flagged = [b for b in bars if b.data_gap]
    # closes 03:00 (prev 02:30) -> overlaps [03:00,04:00)? only the bar closing after 03:00 does
    assert flagged and r.report.bars_spanning_missing_hours == len(flagged)
    assert all(bars[i].timestamp_ms > 1672542000000 for i, b in enumerate(bars) if b.data_gap)


@pytest.mark.asyncio
async def test_replay_feed_delivers_in_order_and_stops(dataset_root):
    seen = []

    async def on_bar(bar):
        seen.append(bar.seq)

    feed = InfoBarReplayFeed(open_bars(dataset_root, *ROOT_ARGS), on_bar, yield_every=10)
    assert await feed.run() == 100
    assert seen == sorted(seen) and len(seen) == 100

    seen.clear()

    async def stopper(bar):
        seen.append(bar.seq)
        if len(seen) == 5:
            await feed.stop()

    feed2 = InfoBarReplayFeed(open_bars(dataset_root, *ROOT_ARGS), stopper)
    feed = feed2
    await feed2.run()
    assert len(seen) == 5


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_timestamp_units_are_scaled_to_epoch_ms(tmp_path, unit):
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = write_dataset(tmp_path, make_rows(5))
    path = next(d.glob("*.parquet"))
    table = pq.read_table(path)
    idx = table.schema.get_field_index("Date Time")
    table = table.set_column(idx, "Date Time", table["Date Time"].cast(pa.timestamp(unit, tz="UTC")))
    pq.write_table(table, path)
    assert [b.timestamp_ms for b in open_bars(tmp_path, *ROOT_ARGS)] == [r["ts"] for r in make_rows(5)]


@pytest.mark.parametrize("name,reason", [("Date Time", "null_timestamp"),
                                         ("Tick Number", "null_seq"),
                                         ("Cumulative Ticks", "null_tick_count")])
def test_null_required_values_are_counted_in_lenient_mode(tmp_path, name, reason):
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = write_dataset(tmp_path, make_rows(5))
    path = next(d.glob("*.parquet"))
    table = pq.read_table(path)
    values = table[name].to_pylist()
    values[2] = None
    table = table.set_column(table.schema.get_field_index(name), name,
                             pa.array(values, type=table[name].type))
    pq.write_table(table, path)
    with pytest.raises(DataIntegrityError, match=reason):
        list(open_bars(tmp_path, *ROOT_ARGS))
    reader = open_bars(tmp_path, *ROOT_ARGS, strict=False)
    assert len(list(reader)) == 4
    assert reader.report.reject_reasons[reason] == 1


def test_non_timestamp_schema_is_rejected(tmp_path):
    import pyarrow.parquet as pq
    d = write_dataset(tmp_path, make_rows(5))
    path = next(d.glob("*.parquet"))
    table = pq.read_table(path)
    table = table.set_column(0, "Date Time", table["Date Time"].cast("int64"))
    pq.write_table(table, path)
    with pytest.raises(DataIntegrityError, match="Arrow timestamp"):
        list(open_bars(tmp_path, *ROOT_ARGS))


@pytest.mark.parametrize("dollar", [0.0, -1.0, 1.0, 1e12])
def test_dollar_value_and_vwap_must_be_real_and_within_bar(tmp_path, dollar):
    rows = make_rows(5)
    rows[2]["dv"] = dollar
    write_dataset(tmp_path, rows)
    with pytest.raises(DataIntegrityError, match="bad_dollar_value"):
        list(open_bars(tmp_path, *ROOT_ARGS))
    reader = open_bars(tmp_path, *ROOT_ARGS, strict=False)
    assert len(list(reader)) == 4
    assert reader.report.reject_reasons["bad_dollar_value"] == 1


def test_window_start_keeps_previous_close_across_pruned_groups(tmp_path):
    rows = make_rows(30, step_s=1800)
    # No closes during the reported outage, then one close well after it.
    for row in rows[8:]:
        row["ts"] += 4 * 3600_000
    write_dataset(tmp_path, rows, missing_hours=["2023-01-01T04:00:00+00:00"], row_group_size=4)
    all_bars = list(open_bars(tmp_path, *ROOT_ARGS))
    target = all_bars[8]
    assert target.data_gap
    for start in (rows[5]["ts"], rows[6]["ts"], rows[7]["ts"] + 1, target.timestamp_ms):
        reader = InfoBarReader(find_dataset(tmp_path, *ROOT_ARGS), start_ms=start)
        bar = next(b for b in reader if b.seq == target.seq)
        assert (bar.duration_ms, bar.data_gap) == (target.duration_ms, target.data_gap)


def test_dataset_identity_includes_consumed_sidecar_bytes(tmp_path):
    import json
    d = write_dataset(tmp_path, make_rows(5))
    first = find_dataset(tmp_path, *ROOT_ARGS)
    sidecar = next(d.glob("*.json"))
    metadata = json.loads(sidecar.read_bytes())
    metadata["missing_tick_hours"] = ["2023-01-01T03:00:00Z"]
    sidecar.write_text(json.dumps(metadata))
    second = find_dataset(tmp_path, *ROOT_ARGS)
    assert first.dataset_id != second.dataset_id
    # Existing readers retain identity of metadata they actually consumed.
    identity = second.dataset_id
    sidecar.unlink()
    assert second.dataset_id == identity
    third = find_dataset(tmp_path, *ROOT_ARGS)
    assert third.dataset_id not in (first.dataset_id, second.dataset_id)


def test_lenient_window_uses_globally_accepted_predecessor(tmp_path):
    rows = make_rows(6)
    for row, seq in zip(rows, [100, 200, 100, 150, 300, 400]):
        row["seq"] = seq
    write_dataset(tmp_path, rows, row_group_size=2)
    full = list(open_bars(tmp_path, *ROOT_ARGS, strict=False))
    reader = InfoBarReader(find_dataset(tmp_path, *ROOT_ARGS), start_ms=rows[4]["ts"], strict=False)
    window = list(reader)
    assert [bar.seq for bar in full] == [100, 200, 300, 400]
    assert window[0].duration_ms == full[2].duration_ms == 90_000
    assert reader.report.reject_reasons["out_of_order"] == 2
    assert reader.first_predecessor_ms == rows[1]["ts"]


def test_dataset_identity_is_frozen_at_discovery_when_file_metadata_changes(tmp_path):
    import os
    from dataclasses import FrozenInstanceError

    directory = write_dataset(tmp_path, make_rows(5))
    info = find_dataset(tmp_path, *ROOT_ARGS)
    identity = info.dataset_id
    path = next(directory.glob("*.parquet"))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert info.dataset_id == identity
    assert find_dataset(tmp_path, *ROOT_ARGS).dataset_id != identity
    with pytest.raises(FrozenInstanceError):
        info.dataset_id = "changed"


@pytest.mark.parametrize("first_index", [0, 1, 4, 7])
def test_first_predecessor_is_exposed_before_yield_including_pruned_groups(tmp_path, first_index):
    rows = make_rows(10)
    write_dataset(tmp_path, rows, row_group_size=2)
    reader = InfoBarReader(find_dataset(tmp_path, *ROOT_ARGS), start_ms=rows[first_index]["ts"])
    first = next(iter(reader))
    assert first.seq == rows[first_index]["seq"]
    assert reader.first_predecessor_ms == (rows[first_index - 1]["ts"] if first_index else None)

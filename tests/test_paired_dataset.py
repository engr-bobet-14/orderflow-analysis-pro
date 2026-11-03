"""Paired dataset integrity (plan §3.3, R8) and full-history trade alignment (plan §4/§7.1, R4/R10 subset)."""
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from orderflow_system.data import paired_dataset as pdx
from orderflow_system.data.infobar_feed import DataIntegrityError, open_bars
from orderflow_system.data.trade_alignment import verify_alignment
from tests import paired_fixture as fx

TYPES = {"fixed_tick_imbalance": None, "fixed_volume_run": None}


def ends_for(n_total: int) -> dict[str, list[int]]:
    """Two bar types over the same stream: fine bars, and coarse bars incl. one spanning two UTC days; both leave a tail."""
    fine = list(range(37, n_total - 5, 37))
    coarse = [250, 1100] + list(range(1300, n_total - 20, 211))   # bar 2 spans day 1 -> day 2
    return {"fixed_tick_imbalance": fine, "fixed_volume_run": coarse}


@pytest.fixture
def dataset(tmp_path):
    trades = fx.make_trades(days=3, per_day=600, gap_hour=5)
    ds_dir, root_hash = fx.write_paired(tmp_path, trades, ends_for(len(trades)),
                                        missing_hours=["2023-01-02T05:00:00+00:00"])
    return ds_dir, root_hash, trades


def registry(tmp_path, ds_dir, root_hash):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({ds_dir.name: root_hash}))
    return path


# ── integrity (R8) ──────────────────────────────────────────

def test_open_verifies_and_exposes_reader_layout(dataset, tmp_path):
    ds_dir, root_hash, trades = dataset
    ds = pdx.open_registered(ds_dir.parent, ds_dir.name, registry(tmp_path, ds_dir, root_hash))
    assert ds.root_hash == root_hash and ds.dataset_id == ds_dir.name
    assert ds.bar_types == sorted(TYPES)
    bars = list(open_bars(ds.bars_root, fx.SYMBOL, "fixed_tick_imbalance"))
    assert bars[0].seq == 37 and int(ds.hours.n_trades.sum()) == len(trades)


def test_unregistered_dataset_refused(dataset, tmp_path):
    ds_dir, _, _ = dataset
    path = tmp_path / "registry.json"
    path.write_text("{}")
    with pytest.raises(DataIntegrityError, match="not registered"):
        pdx.open_registered(ds_dir.parent, ds_dir.name, path)


def _bars_rel(bt="fixed_tick_imbalance"):
    return f"{fx.SYMBOL}/{bt}/{fx.STAMP}.parquet"


def _edited_bars(ds_dir, column, fn):
    table = pq.read_table(ds_dir / _bars_rel())
    values = table.column(column).to_numpy().copy()
    fn(values)
    table = table.set_column(table.schema.get_field_index(column), column, pa.array(values))
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, row_group_size=7)
    return sink.getvalue().to_pybytes()


def _balanced_buy_sell(ds_dir):
    buy = _edited_bars(ds_dir, "Cumulative Buy Volume", lambda v: v.__setitem__(3, v[3] + 0.5))
    table = pq.read_table(pa.BufferReader(buy))
    sell = table.column("Cumulative Sell Volume").to_numpy().copy()
    sell[3] -= 0.5                                       # total volume unchanged
    table = table.set_column(table.schema.get_field_index("Cumulative Sell Volume"), "Cumulative Sell Volume",
                             pa.array(sell))
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, row_group_size=7)
    return sink.getvalue().to_pybytes()


def _sidecar(ds_dir, **edit):
    path = ds_dir / _bars_rel().replace(".parquet", ".json")
    side = json.loads(path.read_text())
    side.update(edit)
    return json.dumps(side).encode()


def _hours(ds_dir, column, fn):
    df = pq.read_table(ds_dir / pdx.HOURS_FILE).to_pandas()
    fn(df, column)
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), sink)
    return sink.getvalue().to_pybytes()


MUTATIONS = {
    "balanced_buy_sell": (_bars_rel(), lambda d: _balanced_buy_sell(d)),
    "threshold": (_bars_rel(), lambda d: _edited_bars(d, "Threshold", lambda v: v.__setitem__(0, 2.0))),
    "sidecar_missing_hours": (_bars_rel().replace(".parquet", ".json"), lambda d: _sidecar(d, missing_tick_hours=[])),
    "sidecar_config": (_bars_rel().replace(".parquet", ".json"),
                       lambda d: _sidecar(d, config={"bar_type": "fixed_tick_run"})),
    "hours_digest": (pdx.HOURS_FILE, lambda d: _hours(d, "digest", lambda df, c: df.__setitem__(c, df[c].str[::-1]))),
    "hours_lots": (pdx.HOURS_FILE, lambda d: _hours(d, "sum_qty_lots", lambda df, c: df.__setitem__(c, df[c] + 1))),
    "trade_manifest": (pdx.TRADE_MANIFEST, lambda d: (d / pdx.TRADE_MANIFEST).read_bytes().replace(b'"n_trades"', b'"n_trades" ')),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
@pytest.mark.parametrize("consistent", [False, True])
def test_any_artifact_mutation_refused_before_first_bar(dataset, tmp_path, name, consistent):
    ds_dir, root_hash, _ = dataset
    rel, make = MUTATIONS[name]
    fx.rewrite_artifact(ds_dir, rel, make(ds_dir), consistent=consistent)
    # consistent=True: the manifest was rewritten to match -> only the pinned root catches it
    with pytest.raises(DataIntegrityError, match="root hash" if consistent else "does not match"):
        pdx.open_registered(ds_dir.parent, ds_dir.name, registry(tmp_path, ds_dir, root_hash))


def test_extra_and_missing_files_refused(dataset, tmp_path):
    ds_dir, root_hash, _ = dataset
    reg = registry(tmp_path, ds_dir, root_hash)
    (ds_dir / "notes.txt").write_text("x")
    with pytest.raises(DataIntegrityError, match="unlisted"):
        pdx.open_registered(ds_dir.parent, ds_dir.name, reg)
    (ds_dir / "notes.txt").unlink()
    (ds_dir / pdx.HOURS_FILE).unlink()
    with pytest.raises(DataIntegrityError, match="missing"):
        pdx.open_registered(ds_dir.parent, ds_dir.name, reg)


def test_replacement_mid_run_caught_by_recheck(dataset, tmp_path):
    ds_dir, root_hash, _ = dataset
    ds = pdx.open_registered(ds_dir.parent, ds_dir.name, registry(tmp_path, ds_dir, root_hash))
    ds.recheck()
    path = ds_dir / _bars_rel()
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(DataIntegrityError, match="changed"):
        ds.recheck()


def test_unknown_row_format_refused(dataset, tmp_path):
    ds_dir, _, _ = dataset
    manifest = json.loads((ds_dir / pdx.TRADE_MANIFEST).read_bytes())
    manifest["row_format"]["price_scale"] = 10
    new_root = fx.rewrite_artifact(ds_dir, pdx.TRADE_MANIFEST, fx.canonical_json(manifest), consistent=True)
    with pytest.raises(DataIntegrityError, match="row format"):
        pdx.PairedDataset.open(ds_dir, new_root)


# ── full-history alignment ──────────────────────────────────

def test_alignment_passes_with_multi_day_bar_gap_and_tails(dataset, tmp_path):
    ds_dir, root_hash, trades = dataset
    report = verify_alignment(pdx.PairedDataset.open(ds_dir, root_hash))
    assert report["status"] == "passed" and report["full_history"] is True
    assert report["n_trades"] == len(trades) and report["hours_verified"] == 72
    assert report["files_verified"] == len(json.loads((ds_dir / pdx.TRADE_MANIFEST).read_bytes())["files"])
    for bt, ends in ends_for(len(trades)).items():
        t = report["types"][bt]
        assert t["bars_checked"] == len(ends)
        assert t["tail_trades"] == len(trades) - ends[-1]           # I4: n(B) + n(T) == n_trades
    assert len(report["rss"]["per_day_bytes"]) == 3 and report["rss"]["peak_bytes"] > 0


def _rewrite_source(ds_dir, day_index, fn):
    manifest = json.loads((ds_dir / pdx.TRADE_MANIFEST).read_bytes())
    rec = manifest["files"][day_index]
    path = Path(manifest["source_root"]) / rec["path"]
    df = pq.read_table(path).to_pandas()
    df = fn(df)
    # gzip: even unchanged rows give different file bytes than the original snappy file
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="gzip")
    return path


def _swap_first_unequal(price, start):
    """Swap two adjacent trades with different prices (a swap of equal prices would be a no-op)."""
    i = start + int(np.argmax(price[start:-1] != price[start + 1:]))
    out = price.copy()
    out[i], out[i + 1] = price[i + 1], price[i]
    return out


SOURCE_MUTATIONS = {
    "maker_flip": lambda df: df.assign(is_buyer_maker=np.where(df.index == 10, ~df.is_buyer_maker, df.is_buyer_maker)),
    "missing_plus_duplicate": lambda df: df.drop(index=10).pipe(
        lambda d: __import__("pandas").concat([d, d.loc[[20]]]).sort_values(["trade_time_ms", "trade_id"])),
    "price_swap_inside_bar": lambda df: df.assign(price=_swap_first_unequal(df.price.to_numpy(), start=10)),
    "byte_identical_rows_recompressed": lambda df: df,
}


@pytest.mark.parametrize("name", sorted(SOURCE_MUTATIONS))
def test_source_mutation_fails_fast(dataset, name):
    ds_dir, root_hash, _ = dataset
    _rewrite_source(ds_dir, 0, SOURCE_MUTATIONS[name])
    ds = pdx.PairedDataset.open(ds_dir, root_hash)          # dataset artifacts themselves are intact
    with pytest.raises(DataIntegrityError, match="source file"):
        verify_alignment(ds)


@pytest.mark.parametrize("name", sorted(set(SOURCE_MUTATIONS) - {"byte_identical_rows_recompressed"}))
def test_row_mutation_caught_by_hour_digest_even_if_file_hash_matches(dataset, name):
    """File SHA refreshed in a consistent manifest: only the per-hour stream digest can catch the edit."""
    ds_dir, _, _ = dataset
    path = _rewrite_source(ds_dir, 0, SOURCE_MUTATIONS[name])
    manifest = json.loads((ds_dir / pdx.TRADE_MANIFEST).read_bytes())
    manifest["files"][0].update(size=path.stat().st_size, sha256=pdx.sha256_file(path))
    new_root = fx.rewrite_artifact(ds_dir, pdx.TRADE_MANIFEST, fx.canonical_json(manifest), consistent=True)
    with pytest.raises(DataIntegrityError, match="hours.parquet|strictly increasing"):
        verify_alignment(pdx.PairedDataset.open(ds_dir, new_root))


def test_recompressed_identical_rows_pass_once_file_hash_is_refreshed(dataset):
    ds_dir, _, _ = dataset
    path = _rewrite_source(ds_dir, 0, SOURCE_MUTATIONS["byte_identical_rows_recompressed"])
    manifest = json.loads((ds_dir / pdx.TRADE_MANIFEST).read_bytes())
    manifest["files"][0].update(size=path.stat().st_size, sha256=pdx.sha256_file(path))
    new_root = fx.rewrite_artifact(ds_dir, pdx.TRADE_MANIFEST, fx.canonical_json(manifest), consistent=True)
    assert verify_alignment(pdx.PairedDataset.open(ds_dir, new_root))["status"] == "passed"


def test_internally_consistent_dataset_with_wrong_bar_fails_alignment(dataset):
    """A builder bug (bar field wrong, manifests consistent) is caught by reconciliation, not hashing."""
    ds_dir, _, _ = dataset
    new_root = fx.rewrite_artifact(ds_dir, _bars_rel(), _balanced_buy_sell(ds_dir), consistent=True)
    ds = pdx.PairedDataset.open(ds_dir, new_root)
    with pytest.raises(DataIntegrityError, match="Cumulative Buy Volume"):
        verify_alignment(ds)


def _bars_with(ds_dir, column, array):
    """Bars parquet bytes with `column` replaced by a pyarrow array (nulls allowed)."""
    table = pq.read_table(ds_dir / _bars_rel())
    table = table.set_column(table.schema.get_field_index(column), column, array)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, row_group_size=7)
    return sink.getvalue().to_pybytes()


def _aligned_after(ds_dir, data):
    """Swap in corrupted bars with a CONSISTENT manifest (new pinned root), then run alignment."""
    new_root = fx.rewrite_artifact(ds_dir, _bars_rel(), data, consistent=True)
    return verify_alignment(pdx.PairedDataset.open(ds_dir, new_root))


@pytest.mark.parametrize("column, value", [
    ("Cumulative Volume", np.nan), ("Cumulative Dollar Value", np.inf), ("Cumulative Buy Volume", -np.inf),
    ("Cumulative Sell Volume", np.nan), ("Open", np.nan), ("High", np.inf),
])
def test_non_finite_bar_value_fails_alignment(dataset, column, value):
    """NaN/Infinity compare False against any tolerance; they must be rejected, never pass."""
    ds_dir, _, _ = dataset
    with pytest.raises(DataIntegrityError, match="non-finite"):
        _aligned_after(ds_dir, _edited_bars(ds_dir, column, lambda v: v.__setitem__(4, value)))


@pytest.mark.parametrize("column", ["Cumulative Volume", "Cumulative Buy Volume", "Close", "Tick Number",
                                    "Cumulative Ticks", "Date Time"])
def test_null_bar_value_fails_alignment(dataset, column):
    ds_dir, _, _ = dataset
    col = pq.read_table(ds_dir / _bars_rel()).column(column).combine_chunks()
    mask = np.zeros(len(col), bool)
    mask[4] = True
    nulled = pa.array(col.to_numpy(zero_copy_only=False), type=col.type, mask=mask)
    with pytest.raises(DataIntegrityError, match="null"):
        _aligned_after(ds_dir, _bars_with(ds_dir, column, nulled))


def _shifted_times(ds_dir, fn):
    col = pq.read_table(ds_dir / _bars_rel()).column("Date Time").combine_chunks()
    us = col.cast(pa.int64()).to_numpy().copy()
    fn(us)
    return _bars_with(ds_dir, "Date Time", pa.array(us).cast(col.type))


@pytest.mark.parametrize("name, fn", [
    ("all_shifted_one_day", lambda us: us.__iadd__(86_400_000_000)),
    ("one_shifted_one_ms", lambda us: us.__setitem__(5, us[5] - 1000)),
    ("two_swapped", lambda us: us.__setitem__(slice(5, 7), us[5:7][::-1].copy())),
])
def test_bar_close_time_must_equal_closing_trade_time(dataset, name, fn):
    ds_dir, _, _ = dataset
    with pytest.raises(DataIntegrityError, match="Date Time"):
        _aligned_after(ds_dir, _shifted_times(ds_dir, fn))


@pytest.mark.parametrize("name, retype", [
    # Codex: the same microsecond integers declared as nanoseconds are 1970 instants, yet cast to int64 unchanged
    ("us_ints_as_ns", lambda col: pa.array(col.cast(pa.int64()).to_numpy(), type=pa.timestamp("ns", tz="UTC"))),
    ("correct_instants_in_ms", lambda col: col.cast(pa.timestamp("ms", tz="UTC"))),
    ("tz_naive_us", lambda col: col.cast(pa.timestamp("us"))),
])
def test_non_canonical_close_time_type_fails_alignment(dataset, name, retype):
    ds_dir, _, _ = dataset
    col = pq.read_table(ds_dir / _bars_rel()).column("Date Time").combine_chunks()
    with pytest.raises(DataIntegrityError, match="Date Time.*timestamp\\[us, tz=UTC\\]"):
        _aligned_after(ds_dir, _bars_with(ds_dir, "Date Time", retype(col)))


def test_non_canonical_tick_number_type_fails_alignment(dataset):
    ds_dir, _, _ = dataset
    col = pq.read_table(ds_dir / _bars_rel()).column("Tick Number").combine_chunks()
    with pytest.raises(DataIntegrityError, match="Tick Number.*int64"):
        _aligned_after(ds_dir, _bars_with(ds_dir, "Tick Number", col.cast(pa.float64())))


def test_null_aggressor_flag_in_source_fails_before_canonicalisation(dataset):
    """A null is_buyer_maker must not become False (taker buy) in the canonical rows."""
    ds_dir, _, _ = dataset

    def null_maker(df):
        flags = df.is_buyer_maker.astype(object)
        flags.iloc[10] = None
        return df.assign(is_buyer_maker=flags)
    path = _rewrite_source(ds_dir, 0, null_maker)
    assert pq.read_table(path).column("is_buyer_maker").null_count == 1
    manifest = json.loads((ds_dir / pdx.TRADE_MANIFEST).read_bytes())
    manifest["files"][0].update(size=path.stat().st_size, sha256=pdx.sha256_file(path))
    new_root = fx.rewrite_artifact(ds_dir, pdx.TRADE_MANIFEST, fx.canonical_json(manifest), consistent=True)
    with pytest.raises(DataIntegrityError, match="null is_buyer_maker"):
        verify_alignment(pdx.PairedDataset.open(ds_dir, new_root))


def test_bar_count_beyond_stream_fails(tmp_path):
    trades = fx.make_trades(days=3, per_day=600)
    ends = ends_for(len(trades))
    ends["fixed_tick_imbalance"].append(len(trades) + 1)       # a Tick Number past the stream
    ds_dir, root_hash = fx.write_paired(tmp_path, trades, ends)
    with pytest.raises(DataIntegrityError, match="Tick Number"):
        verify_alignment(pdx.PairedDataset.open(ds_dir, root_hash))

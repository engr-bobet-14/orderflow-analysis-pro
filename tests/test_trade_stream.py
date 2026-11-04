"""Authentication precedes delivery, including when the query is streamed in tiny batches."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from orderflow_system.data.infobar_feed import DataIntegrityError
from orderflow_system.data.paired_dataset import PairedDataset
from orderflow_system.data.trade_stream import TradeStream
from tests.paired_fixture import make_trades, write_paired, canonical, rewrite_artifact

BT = 'fixed_dollar_imbalance'


def dataset(tmp_path, trades=None):
    if trades is None:
        trades = make_trades(days=3, per_day=80)
        trades.loc[:7, 'trade_time_ms'] = int(trades.trade_time_ms.iloc[0])//3600000*3600000+np.arange(8)
        trades.loc[:7, 'quantity'] = 1.
        trades.loc[:3, 'price'] = [20000., 20000.5, 19999.5, 20000.]
    directory, root = write_paired(tmp_path, trades, {BT: [4, 41, 90, 160, 200]})
    return PairedDataset.open(directory, root), trades


@pytest.mark.parametrize('batch', [1, 7, 997, 1000000])
def test_full_stream_and_empty_hours(tmp_path, batch):
    ds, trades = dataset(tmp_path)
    stream = TradeStream(ds, temp_dir=tmp_path/'spill', batch_rows=batch)
    hours = list(stream)
    assert len(hours) == 72
    assert any(len(h.rows) == 0 for h in hours)
    assert np.array_equal(np.concatenate([h.rows for h in hours]), canonical(trades))
    for h in hours:
        ref = ds.hours[ds.hours.hour_ms == h.hour_ms].iloc[0]
        assert h.first_ordinal == ref.first_ordinal
    assert stream.report['files_verified'] == 3
    assert stream.report['n_verified'] == 240
    assert stream.settings['threads'] == 4
    assert stream.settings['memory_limit'] == '953.6 MiB'
    assert str(tmp_path/'spill') in stream.settings['temp_directory']


@pytest.mark.parametrize('start,end', [(1, 1), (3, 15), (45, 130), (85, 170)])
def test_seek_verifies_whole_hours(tmp_path, start, end):
    ds, trades = dataset(tmp_path)
    s = TradeStream(ds, start, end, temp_dir=tmp_path/'spill', batch_rows=7)
    hours = list(s)
    combined = np.concatenate([h.rows for h in hours])
    first = hours[0].first_ordinal
    assert np.array_equal(combined, canonical(trades)[first-1:first-1+len(combined)])
    assert first <= start
    assert s.report['overscan_n'] == first+len(combined)-1-end
    assert hours[-1].first_ordinal+len(hours[-1].rows)-1 >= end


def mutate(ds, kind, refresh=True):
    f = ds.manifest['files'][0]
    path = Path(ds.manifest['source_root'])/f['path']
    df = pq.read_table(path).to_pandas().drop(columns=['date', 'symbol'], errors='ignore')
    if kind == 'maker':
        df.loc[0, 'is_buyer_maker'] = not df.loc[0, 'is_buyer_maker']
    elif kind == 'duplicate':
        df.iloc[1] = df.iloc[0]
    elif kind == 'swap':
        df.loc[[1, 2], 'price'] = df.loc[[2, 1], 'price'].to_numpy()
    elif kind == 'time':
        df.loc[0, 'trade_time_ms'] = df.loc[4, 'trade_time_ms']+1
    elif kind == 'reorder':
        df = df.iloc[::-1]
    elif kind == 'missing':
        path.unlink()
        return ds
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    if not refresh:
        return ds
    manifest = json.loads((ds.directory/'trade_manifest.json').read_text())
    manifest['files'][0].update(size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    root = rewrite_artifact(ds.directory, 'trade_manifest.json', json.dumps(manifest).encode(), consistent=True)
    return PairedDataset.open(ds.directory, root)


@pytest.mark.parametrize('kind', ['maker', 'duplicate', 'swap', 'time', 'missing'])
def test_mutation_refreshed_hash_refused_before_hour_delivery(tmp_path, kind):
    ds, _ = dataset(tmp_path)
    ds = mutate(ds, kind)
    s = TradeStream(ds, temp_dir=tmp_path/'spill', batch_rows=7)
    delivered = []
    with pytest.raises(DataIntegrityError):
        for h in s:
            delivered.extend(h.rows)
    assert not delivered


def test_physical_reorder_passes(tmp_path):
    ds, trades = dataset(tmp_path)
    ds = mutate(ds, 'reorder')
    assert np.array_equal(np.concatenate([h.rows for h in TradeStream(ds, temp_dir=tmp_path/'spill')]), canonical(trades))


def test_source_sha_and_post_query_order_corruption(tmp_path, monkeypatch):
    ds, _ = dataset(tmp_path)
    s = TradeStream(ds, temp_dir=tmp_path/'spill')
    original = s._canonical
    def corrupt(batch, day):
        a = original(batch, day)
        if len(a) > 1:
            a[[0, 1]] = a[[1, 0]]
        return a
    monkeypatch.setattr(s, '_canonical', corrupt)
    with pytest.raises(DataIntegrityError, match='increasing'):
        list(s)
    ds = mutate(ds, 'maker', refresh=False)
    with pytest.raises(DataIntegrityError, match='source file'):
        list(TradeStream(ds, temp_dir=tmp_path/'spill'))


def test_spill_must_be_local(tmp_path):
    ds, _ = dataset(tmp_path)
    with pytest.raises(ValueError, match='local'):
        TradeStream(ds, temp_dir=Path('/Volumes/not-allowed'))


def test_spill_cleanup_after_abort_and_error(tmp_path):
    ds,_=dataset(tmp_path)
    it=iter(TradeStream(ds,temp_dir=tmp_path/'spill',batch_rows=7))
    next(it)
    it.close()
    assert not list((tmp_path/'spill').iterdir())
    ds=mutate(ds,'maker')
    with pytest.raises(DataIntegrityError):
        list(TradeStream(ds,temp_dir=tmp_path/'spill'))
    assert not list((tmp_path/'spill').iterdir())


def test_full_stream_authenticates_leading_empty_hours(tmp_path):
    trades=make_trades(days=3,per_day=80)
    first_day=int(trades.trade_time_ms.iloc[0])//86400000*86400000
    times=trades.trade_time_ms.to_numpy().copy()
    times[:80]=first_day+6*3600000+(times[:80]-first_day)//2
    trades['trade_time_ms']=times
    ds,_=dataset(tmp_path,trades)
    stream=TradeStream(ds,temp_dir=tmp_path/'spill')
    hours=list(stream)
    assert len(hours)==72
    assert all(len(h.rows)==0 for h in hours[:6])
    assert hours[0].first_ordinal==1


def test_stream_without_psutil_reports_peak_rss(tmp_path, monkeypatch):
    import sys
    ds, _ = dataset(tmp_path)
    monkeypatch.setitem(sys.modules, 'psutil', None)
    stream = TradeStream(ds, temp_dir=tmp_path/'spill')
    assert sum(len(h.rows) for h in stream) == ds.manifest['n_trades']
    assert stream.report['rss_kind'] == 'peak'
    assert all(s['rss_kind'] == 'peak' and s['bytes'] > 0 for s in stream.report['per_day_rss'])


def test_missing_duckdb_names_tuning_extra(tmp_path, monkeypatch):
    import sys
    ds, _ = dataset(tmp_path)
    monkeypatch.setitem(sys.modules, 'duckdb', None)
    with pytest.raises(ImportError, match='tuning'):
        list(TradeStream(ds, temp_dir=tmp_path/'spill'))


def test_stream_disables_external_file_cache(tmp_path, monkeypatch):
    from contextlib import contextmanager
    import duckdb
    ds, _ = dataset(tmp_path)
    connect = duckdb.connect
    cache_bytes = []
    @contextmanager
    def inspected_connect(*args, **kwargs):
        with connect(*args, **kwargs) as con:
            yield con
            cache_bytes.append(con.execute("""
                SELECT coalesce(sum(memory_usage_bytes), 0)
                FROM duckdb_memory() WHERE tag = 'EXTERNAL_FILE_CACHE'
            """).fetchone()[0])
    monkeypatch.setattr(duckdb, 'connect', inspected_connect)
    stream = TradeStream(ds, temp_dir=tmp_path/'spill')
    assert sum(len(hour.rows) for hour in stream) == ds.manifest['n_trades']
    assert len(stream.report['per_day_rss']) == 3
    assert cache_bytes == [0]
    assert stream.settings['enable_external_file_cache'] is False


def test_suitewide_report_guard_rejects_other_module_output(tmp_path):
    import os
    import subprocess
    import sys
    child = tmp_path/'child'
    child.mkdir()
    watched = tmp_path/'watched'
    watched.mkdir()
    (child/'conftest.py').write_text('from tests.conftest import no_repo_trade_report_pollution\n')
    (child/'test_other_module.py').write_text('''
import os
from pathlib import Path

def test_add_report_from_other_module():
    (Path(os.environ['ORDERFLOW_TEST_REPORT_DIR'])/'pollution.json').write_text('{}')
''')
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo), ORDERFLOW_TEST_REPORT_DIR=str(watched))
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                             '--confcutdir='+str(child), str(child/'test_other_module.py')],
                            cwd=child, env=env, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout+result.stderr
    assert 'test polluted repo output/trade_replay' in result.stdout
    assert 'pollution.json' in result.stdout


def test_stream_records_quiescent_memory_on_own_connection(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import duckdb
    from orderflow_system.data import trade_stream
    ds, trades = dataset(tmp_path)
    connect = duckdb.connect
    connections, cursors = [], []
    @contextmanager
    def inspected_connect(*args, **kwargs):
        with connect(*args, **kwargs) as con:
            connections.append(con)
            def cursor():
                cursors.append(con)
                return con.cursor()
            yield SimpleNamespace(execute=con.execute, cursor=cursor)
    monkeypatch.setattr(duckdb, 'connect', inspected_connect)
    monkeypatch.setattr(trade_stream, 'malloc_in_use_bytes', lambda: None, raising=False)
    stream = TradeStream(ds, temp_dir=tmp_path/'spill')
    assert np.array_equal(np.concatenate([h.rows for h in stream]), canonical(trades))
    samples = stream.report['per_day_rss']
    assert len(samples) == 3
    for sample in samples:
        assert sample['duckdb_memory_bytes'] == 0
        assert sample['duckdb_memory_tags'] == {}
        assert sample['malloc_in_use_bytes'] is None
        assert sample['unavailable'] == ['malloc_in_use_bytes']
    assert len(connections) == 1 and cursors == connections*3


def test_malloc_in_use_is_unavailable_off_darwin(monkeypatch):
    import ctypes
    from types import SimpleNamespace
    from orderflow_system.data import trade_stream
    monkeypatch.setattr(trade_stream, 'sys', SimpleNamespace(platform='linux'), raising=False)
    monkeypatch.setattr(ctypes, 'CDLL', lambda *a: pytest.fail('malloc library loaded off darwin'))
    assert trade_stream.malloc_in_use_bytes() is None


def test_malloc_in_use_reads_darwin_zone_statistics(monkeypatch):
    import ctypes
    from types import SimpleNamespace
    from orderflow_system.data import trade_stream
    calls = []
    class Statistics:
        def __call__(self, zone, result):
            calls.append(zone)
            result._obj.size_in_use = 123456
    statistics = Statistics()
    monkeypatch.setattr(trade_stream, 'sys', SimpleNamespace(platform='darwin'), raising=False)
    def library(name):
        assert name is None
        return SimpleNamespace(malloc_zone_statistics=statistics)
    monkeypatch.setattr(ctypes, 'CDLL', library)
    assert trade_stream.malloc_in_use_bytes() == 123456
    assert calls == [None]

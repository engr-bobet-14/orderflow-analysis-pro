"""Cache serialization and the full-stream development boundary (red before T1)."""
import builtins
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.paired_fixture import make_trades, write_paired, rewrite_artifact
from orderflow_system.data.paired_dataset import PairedDataset
from orderflow_system.data.trade_stream import TradeStream
from scripts import trade_feature_cache as tfc

BT = 'fixed_dollar_imbalance'


def paired(tmp_path):
    trades = make_trades(days=3, per_day=80)
    directory, root = write_paired(tmp_path, trades, {BT: [4, 41, 90, 160, 200]})
    return PairedDataset.open(directory, root)


def test_policy_full_stream_never_opens_margin(tmp_path, monkeypatch):
    ds = paired(tmp_path)
    manifest = dict(ds.manifest)
    forbidden = Path(manifest['source_root'])/'date=2024-01-01/symbol=BTCUSDT/margin.parquet'
    manifest['files'] = [*manifest['files'], {'path': str(forbidden.relative_to(manifest['source_root'])),
        'min_time': 1704067200006, 'max_time': 1704067200010, 'size': 123, 'sha256': '0'*64}]
    root = rewrite_artifact(ds.directory, 'trade_manifest.json', json.dumps(manifest).encode(), consistent=True)
    ds = PairedDataset.open(ds.directory, root)
    opened, hashed, queries = [], [], []
    def intercept(original):
        def checked(path, *args, **kwargs):
            if not isinstance(path, int):
                opened.append(str(path))
                assert str(path) != str(forbidden)
            return original(path, *args, **kwargs)
        return checked
    for mod, name in [(builtins, 'open'), (io, 'open'), (os, 'open')]:
        monkeypatch.setattr(mod, name, intercept(getattr(mod, name)))
    import orderflow_system.data.trade_stream as stream
    original = stream._check_file
    def check(f, seen):
        hashed.append(str(f['abs']))
        assert f['abs'] != forbidden
        return original(f, seen)
    monkeypatch.setattr(stream, '_check_file', check)
    s = TradeStream(ds, temp_dir=tmp_path/'spill', development_only=True, query_observer=queries.extend)
    assert sum(len(h.rows) for h in s) == 240
    assert len(set(hashed)) == 3
    assert forbidden.as_posix() not in queries + opened
    assert s.report['excluded_by_policy'][0]['path'] == manifest['files'][-1]['path']
    assert s.report['excluded_by_policy'][0]['authenticated'] is False
    assert s.report['files_verified'] == 3


def test_cache_live_roundtrip_and_tampering(tmp_path, monkeypatch):
    ds = paired(tmp_path)
    monkeypatch.setattr(tfc, 'pinned_root', lambda _: ds.root_hash)
    monkeypatch.setattr(tfc, 'checked_write', lambda p: Path(p))
    paths, report = tfc.build(ds, tmp_path/'cache', start='2023-01-01', end='2023-01-04')
    path = paths[BT]
    cache = tfc.CachedTradeUpdates(path)
    bars = cache.bars()
    assert len(cache) == len(bars) == 5
    assert cache.trade_digest() == report['lanes'][BT]['trade_digest']
    assert cache[0].footprint_levels[0].dtype == np.dtype('<i8')
    assert sum(len(cache[i].finalised_days) for i in range(len(cache))) >= 2
    cache.validate_bars(bars)
    with pytest.raises(ValueError, match='production'):
        tfc.CachedTradeUpdates(path, production=True)
    bars.array = bars.array.copy()
    bars.array['seq'][0] += 1
    with pytest.raises(ValueError, match='alignment'):
        cache.validate_bars(bars)
    with (path/'stats.npy').open('r+b') as f:
        f.seek(-1, 2)
        f.write(b'x')
    with pytest.raises(ValueError, match='hash'):
        tfc.CachedTradeUpdates(path)


def test_range_and_provenance_guards(monkeypatch):
    with pytest.raises(ValueError, match='4 days'):
        tfc.validate_build_range('2023-01-01', '2023-01-06')
    with pytest.raises(ValueError, match='development'):
        tfc.validate_build_range('2023-12-31', '2024-01-02')
    monkeypatch.setattr(tfc, 'source_state', lambda: {'dirty': True})
    with pytest.raises(ValueError, match='clean'):
        tfc.require_source_mode('production')


def test_reference_fixture():
    refs = json.loads(Path('tests/fixtures/phase4_reference_digests.json').read_text())
    for run in refs.values():
        assert len(run['source_sha256']) == 64
        assert len(run['lanes']) == 6
        assert all(len(l['trade_digest']) == 64 for l in run['lanes'].values())


def test_manifest_tamper_refused(tmp_path,monkeypatch):
    ds = paired(tmp_path)
    monkeypatch.setattr(tfc,'pinned_root',lambda _:ds.root_hash)
    monkeypatch.setattr(tfc,'checked_write',lambda p:Path(p))
    paths,_ = tfc.build(ds,tmp_path/'cache',start='2023-01-01',end='2023-01-04')
    path = paths[BT]
    m = json.loads((path/'manifest.json').read_text())
    m['created_utc'] = 'tampered'
    (path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='manifest hash'):
        tfc.CachedTradeUpdates(path)


def test_source_change_and_crash_leave_no_visible_cache(tmp_path,monkeypatch):
    ds = paired(tmp_path)
    monkeypatch.setattr(tfc,'pinned_root',lambda _:ds.root_hash)
    monkeypatch.setattr(tfc,'checked_write',lambda p:Path(p))
    monkeypatch.setattr(tfc,'verify_source',lambda _:(_ for _ in ()).throw(ValueError('source_state changed')))
    with pytest.raises(ValueError,match='source_state changed'):
        tfc.build(ds,tmp_path/'cache',start='2023-01-01',end='2023-01-04')
    assert not list((tmp_path/'cache').iterdir())


def test_full_history_requires_explicit_owner_token():
    with pytest.raises(ValueError,match='owner-approved'):
        tfc.validate_build_range(None,None,full_history=True)
    with pytest.raises(ValueError,match='approval'):
        tfc.validate_build_range(None,None,full_history=True,owner_approved='unit-only')


def test_cache_dataset_type_bucket_time_mismatches(tmp_path,monkeypatch):
    ds = paired(tmp_path)
    monkeypatch.setattr(tfc,'pinned_root',lambda _:ds.root_hash)
    monkeypatch.setattr(tfc,'checked_write',lambda p:Path(p))
    paths,_ = tfc.build(ds,tmp_path/'cache',start='2023-01-01',end='2023-01-04')
    path = paths[BT]
    for kwargs in [{'dataset_id':'another'},{'bar_type':'fixed_tick_run'},{'fp_bucket_ticks':50},{'pf_bucket_ticks':500}]:
        with pytest.raises(ValueError,match='mismatch'):
            tfc.CachedTradeUpdates(path,**kwargs)
    cache = tfc.CachedTradeUpdates(path)
    bars = cache.bars()
    bars.array=bars.array.copy()
    bars.array['timestamp_ms'][0]+=1
    with pytest.raises(ValueError,match='alignment'):
        cache.validate_bars(bars)


@pytest.mark.parametrize('destination',['/Volumes/Market-Data/cache','/tmp/phase4-cache-forbidden'])
def test_build_cli_refuses_destination_before_dataset_open(destination,monkeypatch,capsys):
    monkeypatch.setattr(tfc,'open_registered',lambda *a:pytest.fail('opened dataset before path validation'))
    with pytest.raises(SystemExit) as error:
        tfc.main(['build','--start','2023-01-01','--end','2023-01-02','--out',destination])
    assert error.value.code==2 and 'allowlist' in capsys.readouterr().err


def test_empty_bar_zero_levels_and_multiday_serialization(tmp_path):
    from dataclasses import replace
    from orderflow_system.analytics.trade_accumulator import TradeBarStats,TradeBarUpdate
    from test_pipeline import synth_bars
    w=tfc._Writer(tmp_path/'staging')
    bar=synth_bars(1)[0]
    stats=TradeBarStats(*([0]*13))
    empty=(np.array([],dtype='<i4'),np.array([],dtype='<i8'),np.array([],dtype='<i8'))
    w.capture(replace(bar,seq=0,volume=0.,buy_volume=0.,sell_volume=0.,tick_count=0),TradeBarUpdate(stats,empty,[],100))
    days=[(1,np.array([2000000],dtype='<i8'),np.array([1000],dtype='<i8')),
          (2,np.array([2000100],dtype='<i8'),np.array([2000],dtype='<i8'))]
    w.capture(bar,TradeBarUpdate(stats,empty,days,100))
    w.finish()
    assert np.load(w.path/'fp_0.npy').dtype==np.dtype('<i4')
    assert np.load(w.path/'fp_offsets.npy').tolist()==[0,0,0]
    assert np.load(w.path/'day_bar_index.npy').tolist()==[1,1]
    assert np.load(w.path/'day_id.npy').tolist()==[1,2]
    assert np.load(w.path/'day_offsets.npy').tolist()==[0,1,2]


def test_rss_budget_aborts_before_publish(tmp_path,monkeypatch):
    ds=paired(tmp_path)
    monkeypatch.setattr(tfc,'pinned_root',lambda _:ds.root_hash)
    monkeypatch.setattr(tfc,'checked_write',lambda p:Path(p))
    monkeypatch.setattr(tfc,'peak_rss_bytes',lambda:5*1024**3)
    with pytest.raises(ValueError,match='RSS'):
        tfc.build(ds,tmp_path/'cache',start='2023-01-01',end='2023-01-04')
    assert not list((tmp_path/'cache').iterdir())

"""Shared-pass lanes, warm-up domains, reconciliation and conservation."""
import asyncio
from dataclasses import asdict, replace
from datetime import datetime, timezone

import pytest

from orderflow_system.config.settings import get_btcusd_config
from orderflow_system.data.infobar_feed import open_bars, DataIntegrityError
from orderflow_system.data.paired_dataset import PairedDataset
from orderflow_system.data.trade_stream import TradeReplay
from orderflow_system.pipeline import InfoBarPipeline
from tests.paired_fixture import make_trades, write_paired, STAMP, canonical, day_ms, DAY0
from scripts import replay_trades as replay_script

TYPES = ['fixed_dollar_imbalance', 'fixed_tick_run']


def setup(tmp_path, trades=None, ends=None):
    trades = make_trades(days=3, per_day=100) if trades is None else trades
    ends = ends or {TYPES[0]: [4, 33, 99, 101, 190, 210, 270], TYPES[1]: [3, 66, 130, 201, 250]}
    directory, root = write_paired(tmp_path, trades, ends)
    return PairedDataset.open(directory, root), trades


def replay(ds, tmp_path, start=None, end=None, batch=7, types=TYPES):
    lanes = [(open_bars(ds.bars_root, 'BTCUSDT', bt, start, end, version=STAMP),
              InfoBarPipeline(get_btcusd_config())) for bt in types]
    return TradeReplay(ds, lanes, temp_dir=tmp_path/'spill', batch_rows=batch)


@pytest.mark.parametrize('batch', [1, 7, 997, 1000000])
def test_shared_pass_conservation_and_multiday_tail(tmp_path, batch):
    ds, trades = setup(tmp_path)
    r = replay(ds, tmp_path, batch=batch)
    results = list(r)
    assert len(results) == 12
    assert r.report['stream']['n_verified'] == len(trades)
    assert 'trade_signals' not in r.report
    for bt in TYPES:
        lane = r.report['lanes'][bt]
        last = 270 if bt == TYPES[0] else 250
        assert lane['n_tail'] == len(trades)-last
        assert lane['lots_tail'] == int(canonical(trades.iloc[last:])['qty_lots'].sum())
        assert lane['identities'] == {'I1': True, 'I2': True, 'I3': True, 'I4': True}
        snapshot = r.lanes[bt].accumulator.developing_profiles()
        assert sum(int(x[2].sum()) for x in snapshot) == lane['developing_lots']


def test_tail_across_days_never_enters_profile(tmp_path):
    ds, trades = setup(tmp_path, ends={TYPES[0]: [4, 33]})
    r = replay(ds, tmp_path, types=[TYPES[0]])
    list(r)
    lane = r.report['lanes'][TYPES[0]]
    assert lane['n_tail'] == 267
    assert len(lane['developing_profiles']) == 1
    assert lane['developing_lots'] == int(canonical(trades.iloc[:33])['qty_lots'].sum())


@pytest.mark.parametrize('start,end', [('2023-01-01', '2023-01-02'), ('2023-01-02', '2023-01-03'),
                                      ('2023-01-03', '2023-01-04')])
def test_window_equals_full_trade_outputs(tmp_path, start, end):
    ds, _ = setup(tmp_path)
    full = replay(ds, tmp_path)
    outputs = list(full)
    window = replay(ds, tmp_path, start, end)
    limited = list(window)
    for bt in TYPES:
        wanted = [r for name,r in outputs if name == bt and start <= datetime.fromtimestamp(
            r.bar.timestamp_ms/1000, timezone.utc).strftime('%Y-%m-%d') < end]
        got = [r for name,r in limited if name == bt]
        for before, after in zip(wanted, got, strict=True):
            assert before.trade_stats == after.trade_stats
            assert before.trade_footprint == after.trade_footprint
            assert before.trade_profiles == after.trade_profiles
        lane = window.report['lanes'][bt]
        assert lane['n_tail'] == lane['lots_tail'] == 0
        assert lane['identities']['I1'] and lane['identities']['I2']
        # Final developing histograms equal the state committed at the last included close.
        if got:
            prefix = replay(ds, tmp_path, end=datetime.fromtimestamp(
                (got[-1].bar.timestamp_ms+1)/1000, timezone.utc).isoformat(), types=[bt])
            list(prefix)
            assert prefix.report['lanes'][bt]['developing_profiles'] == lane['developing_profiles']


def test_midday_result_release_preserves_frozen_trade_outputs_and_digests(tmp_path):
    import hashlib
    import json
    from dataclasses import asdict

    trades = make_trades(days=2, per_day=10000)
    bt = TYPES[0]
    ends = [4, 1000, 4000, 7500, 9990, 11000, 14000, 17500, 19990]
    directory, root = write_paired(tmp_path, trades, {bt: ends}, days=2)
    ds = PairedDataset.open(directory, root)
    run = replay(ds, tmp_path, '2023-01-01', '2023-01-02T12:00:00+00:00',
                 batch=997, types=[bt])
    payload = [{'seq': result.bar.seq, 'stats': asdict(result.trade_stats),
                'footprint': asdict(result.trade_footprint),
                'profiles': [asdict(p) for p in result.trade_profiles]}
               for _, result in run]
    # Frozen from 948f1d0, before the result-release change; includes a finalised day.
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str).encode()
    assert hashlib.sha256(encoded).hexdigest() == '1dc021017df64e155b31a7e220583da9aafdc9d0ff41bf9a65edead9961111b4'
    lane = run.report['lanes'][bt]
    assert lane['replay_digest'] == '39f9807a8071060f4fe98bd9f4a08713d896629fa5ed2a71cf47d235b5f31652'
    assert lane['trade_digest'] == 'de3e4d660a2e3fbdef60cf61f94d806bb3f0445ca089acdf9d7117cc1db7b481'
    assert lane['finalised_days'] == 1
    assert all(s['duckdb_memory_bytes'] == 0 for s in run.report['stream']['per_day_rss'])


def test_midnight_close_and_predecessor_tie_warmup(tmp_path):
    trades = make_trades(days=3, per_day=100)
    t = trades.trade_time_ms.to_numpy().copy()
    midnight = day_ms(DAY0)+86400000
    t[98:100] = midnight-100
    t[100:102] = midnight
    trades['trade_time_ms'] = t
    ds, _ = setup(tmp_path, trades=trades, ends={TYPES[0]: [50,99,100,101,200,270], TYPES[1]: [45,100,102,250]})
    full = replay(ds,tmp_path)
    expected = list(full)
    win = replay(ds,tmp_path,'2023-01-02','2023-01-03')
    actual = list(win)
    for bt in TYPES:
        before = [r for name,r in expected if name==bt and midnight <= r.bar.timestamp_ms < midnight+86400000]
        after = [r for name,r in actual if name==bt]
        for x,y in zip(before,after,strict=True):
            assert (x.trade_stats,x.trade_footprint,x.trade_profiles) == (y.trade_stats,y.trade_footprint,y.trade_profiles)
        assert win.report['lanes'][bt]['warm_n']==100


@pytest.mark.parametrize('field,value', [('tick_count', 999), ('open', 1.), ('high', 90000.), ('low', 1.),
    ('close', 1.), ('timestamp_ms', 1), ('volume', 900.), ('dollar_value', 1.), ('seq', 5)])
def test_each_bar_reconciliation_refuses(tmp_path, field, value):
    ds, _ = setup(tmp_path)
    reader = open_bars(ds.bars_root,'BTCUSDT',TYPES[0],version=STAMP)
    class CorruptReader:
        def __getattr__(self, name): return getattr(reader,name)
        def __iter__(self):
            for i,b in enumerate(reader):
                yield replace(b, **{field:value}) if i==0 else b
    r = TradeReplay(ds, [(CorruptReader(), InfoBarPipeline(get_btcusd_config()))], temp_dir=tmp_path/'spill')
    with pytest.raises(DataIntegrityError): list(r)


def test_end_recheck_midrun_artifact_replacement(tmp_path):
    ds, _ = setup(tmp_path)
    r = replay(ds,tmp_path)
    it=iter(r)
    next(it)
    path=ds.directory/'trade_manifest.json'
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(DataIntegrityError, match='changed after verification'):
        list(it)


@pytest.mark.parametrize('args', [ ['--trades'], ['--trades','--paired-dataset','v0','--persist'],
                                   ['--trades','--paired-dataset','v0','--alerts']])
def test_cli_refuses_invalid_trade_modes_before_bar_read(args):
    from orderflow_system.main import parse_args
    with pytest.raises(SystemExit): parse_args(args)


def test_main_authenticates_before_bar_discovery(tmp_path, monkeypatch):
    from orderflow_system import main as app
    from orderflow_system.config.settings import TradeFlowConfig
    calls=[]
    def refuse(*a, **kw):
        calls.append('authenticate')
        raise DataIntegrityError('unregistered')
    monkeypatch.setattr('orderflow_system.data.paired_dataset.open_registered', refuse)
    monkeypatch.setattr(app, 'open_bars', lambda *a, **kw: pytest.fail('bar read before authentication'))
    with pytest.raises(DataIntegrityError, match='unregistered'):
        asyncio.run(app.OrderflowSystem(paired_dataset='v0',
            trade_config=TradeFlowConfig(report_dir=str(tmp_path/'reports'))).start())
    assert calls==['authenticate']


def test_replay_script_authentication_report_and_certificate(tmp_path, monkeypatch):
    import json
    from scripts import replay_trades as script
    ds, _ = setup(tmp_path)
    monkeypatch.setattr(script, 'open_registered', lambda *a,**kw: ds)
    cfg=script.TradeFlowConfig(duckdb_temp_dir=str(tmp_path/'spill'), report_dir=str(tmp_path/'reports'))
    result=script.run(ds.dataset_id, config=cfg, bar_types=TYPES)
    assert result['full_history']
    assert result['trade_signals'] == asdict(cfg.signals)
    assert all(x['identities']['I4'] for x in result['lanes'].values())
    cert=tmp_path/'cert.json'
    cert.write_text(json.dumps({'certification':'dataset-pipeline', 'pipeline_equivalent_default':True,
        'v1': {'dataset_id':ds.dataset_id,'root_hash':ds.root_hash}, 'types': {
        bt: {'L4': {'passed':True,'configs': [{'config':'default','params':None,'window':[None,None],
             'v1':result['lanes'][bt]['replay_digest'],'n_bars':result['lanes'][bt]['bars'],'equal':True}]}} for bt in TYPES}}))
    again=script.run(ds.dataset_id,config=cfg,bar_types=TYPES,certificate=cert, windows=20,seed=7)
    assert again['certificate_match'] and len(again['windows'])==20
    assert all(w['equal'] for w in again['windows'])
    cert.write_text(cert.read_text().replace(ds.root_hash, '0'*64))
    monkeypatch.setattr(script, 'open_bars', lambda *a,**kw: pytest.fail('bar read before certificate validation'))
    with pytest.raises(DataIntegrityError,match='certificate'):
        script.run(ds.dataset_id,config=cfg,bar_types=TYPES,certificate=cert)


def test_script_unregistered_refuses_before_bars(monkeypatch):
    from scripts import replay_trades as script
    monkeypatch.setattr(script,'open_bars',lambda *a,**kw:pytest.fail('bar read before authentication'))
    with pytest.raises(DataIntegrityError,match='not registered'):
        script.run('v0')


def test_warmup_predecessor_bar_spans_days_and_lanes_have_different_w(tmp_path):
    ds, _ = setup(tmp_path, ends={TYPES[0]: [150, 210, 270], TYPES[1]: [50, 99, 210, 250]})
    full = replay(ds, tmp_path)
    before = list(full)
    window = replay(ds, tmp_path, '2023-01-03', '2023-01-04')
    after = list(window)
    assert window.lanes[TYPES[0]].w == 101
    assert window.lanes[TYPES[1]].w == 1
    for bt in TYPES:
        expected = [r for name,r in before if name==bt and r.bar.timestamp_ms >= day_ms(DAY0)+2*86400000]
        actual = [r for name,r in after if name==bt]
        for x,y in zip(expected, actual, strict=True):
            assert x.trade_stats == y.trade_stats
            assert x.trade_footprint == y.trade_footprint
            assert x.trade_profiles == y.trade_profiles


def test_batch_size_trade_digest_invariance(tmp_path):
    ds, _ = setup(tmp_path)
    digests=[]
    for batch in (1,7,997,1000000):
        r=replay(ds,tmp_path,batch=batch)
        for _ in r: pass
        digests.append({bt: lane['trade_digest'] for bt,lane in r.report['lanes'].items()})
    assert all(d==digests[0] for d in digests)


def test_midnight_window_end_keeps_new_day_developing(tmp_path):
    trades=make_trades(days=3,per_day=100)
    t=trades.trade_time_ms.to_numpy().copy()
    midnight=day_ms(DAY0)+86400000
    t[100]=midnight
    trades['trade_time_ms']=t
    ds,_=setup(tmp_path,trades=trades,ends={TYPES[0]:[99,101,190,270]})
    end=datetime.fromtimestamp((midnight+1)/1000,timezone.utc).isoformat()
    r=replay(ds,tmp_path,end=end,types=[TYPES[0]])
    out=list(r)
    assert len(out[-1][1].trade_profiles)==1
    assert out[-1][1].trade_profiles[0].session_date=='2023-01-01'
    lane=r.report['lanes'][TYPES[0]]
    assert lane['developing_lots']==int(canonical(trades.iloc[100:101])['qty_lots'].sum())
    assert lane['n_tail']==0


def test_mutation_never_reaches_pipeline(tmp_path):
    from tests.test_trade_stream import dataset, mutate
    ds,_=dataset(tmp_path)
    ds=mutate(ds,'maker')
    reader=open_bars(ds.bars_root,'BTCUSDT',TYPES[0],version=STAMP)
    pipe=InfoBarPipeline(get_btcusd_config())
    r=TradeReplay(ds,[(reader,pipe)],temp_dir=tmp_path/'spill')
    with pytest.raises(DataIntegrityError): list(r)
    assert pipe.bars_processed==0 and pipe.trade_view is None


@pytest.mark.parametrize('signals_enabled', [False, True])
def test_main_replay_finishes_and_rechecks_without_double_processing(tmp_path, monkeypatch, signals_enabled):
    from orderflow_system import main as app
    from orderflow_system.config.settings import TradeFlowConfig, TradeSignalConfig
    ds,_=setup(tmp_path,ends={TYPES[0]:[4,99,101,270]})
    cfg=TradeFlowConfig(duckdb_temp_dir=str(tmp_path/'spill'), report_dir=str(tmp_path/'reports'),
                        signals=TradeSignalConfig(absorption_levels=signals_enabled))
    system=app.OrderflowSystem(paired_dataset=ds.dataset_id,trade_config=cfg)
    system.dataset=ds
    system.reader=open_bars(ds.bars_root,'BTCUSDT',TYPES[0],version=STAMP)
    system._running=True
    monkeypatch.setattr(app.DASHBOARD,'enabled',False)
    asyncio.run(system._replay())
    assert system.replay_finished
    assert system.pipelines['BTCUSDT'].bars_processed==4
    assert system.trade_report['lanes'][TYPES[0]]['identities']['I4']
    assert system.trade_report['trade_signals'] == asdict(cfg.signals)
    assert not list((tmp_path/'spill').iterdir())


def test_main_trade_replay_rechecks_artifact_after_processing(tmp_path, monkeypatch):
    from orderflow_system import main as app
    from orderflow_system.config.settings import TradeFlowConfig
    ds,_=setup(tmp_path,ends={TYPES[0]:[4,99,101,270]})
    system=app.OrderflowSystem(paired_dataset=ds.dataset_id,
                              trade_config=TradeFlowConfig(duckdb_temp_dir=str(tmp_path/'spill'), report_dir=str(tmp_path/'reports')))
    system.dataset=ds
    system.reader=open_bars(ds.bars_root,'BTCUSDT',TYPES[0],version=STAMP)
    system._running=True
    original=system._on_bar
    async def mutate(bar,result=None):
        await original(bar,result)
        path=ds.directory/'trade_manifest.json'
        path.write_bytes(path.read_bytes()+b' ')
    monkeypatch.setattr(system,'_on_bar',mutate)
    with pytest.raises(DataIntegrityError,match='changed after verification'):
        asyncio.run(system._replay())
    assert not system.replay_finished


def test_window_end_lookup_prunes_later_bar_row_groups(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    ds,_=setup(tmp_path,ends={TYPES[0]:list(range(4,299,3))})
    reads=[]
    original=pq.ParquetFile.read_row_group
    def read(self, index, *args, **kw):
        if kw.get('columns')==['Date Time','Tick Number']:
            reads.append(index)
        return original(self,index,*args,**kw)
    monkeypatch.setattr(pq.ParquetFile,'read_row_group',read)
    replay(ds,tmp_path,'2023-01-01','2023-01-02',types=[TYPES[0]])
    assert len(reads)==1


def test_script_rechecks_after_midrun_artifact_change(tmp_path, monkeypatch):
    from scripts import replay_trades as script
    ds,_=setup(tmp_path)
    monkeypatch.setattr(script,'open_registered',lambda *a,**kw:ds)
    original=script.InfoBarPipeline.process_bar
    def mutate(self, bar, trade=None):
        result=original(self,bar,trade)
        path=ds.directory/'trade_manifest.json'
        path.write_bytes(path.read_bytes()+b' ')
        return result
    monkeypatch.setattr(script.InfoBarPipeline,'process_bar',mutate)
    with pytest.raises(DataIntegrityError,match='changed after verification'):
        script.run(ds.dataset_id, config=script.TradeFlowConfig(duckdb_temp_dir=str(tmp_path/'spill'), report_dir=str(tmp_path/'reports')))


def test_main_start_authenticates_and_uses_manifest_version(tmp_path, monkeypatch):
    from orderflow_system import main as app
    from orderflow_system.config.settings import TradeFlowConfig
    ds,_=setup(tmp_path,ends={TYPES[0]:[4,99,101,270]})
    called=[]
    def authenticate(*a,**kw):
        called.append('authenticate')
        return ds
    def bars(root,symbol,bar_type,**kw):
        assert called==['authenticate']
        assert str(root)==str(ds.bars_root) and kw['version']==ds.manifest['stamp']
        assert kw['strict']
        return open_bars(root,symbol,bar_type,**kw)
    monkeypatch.setattr('orderflow_system.data.paired_dataset.open_registered',authenticate)
    monkeypatch.setattr(app,'open_bars',bars)
    monkeypatch.setattr(app.DASHBOARD,'enabled',False)
    system=app.OrderflowSystem(paired_dataset=ds.dataset_id,
                              trade_config=TradeFlowConfig(duckdb_temp_dir=str(tmp_path/'spill'), report_dir=str(tmp_path/'reports')))
    async def no_network(): pass
    monkeypatch.setattr(system.telegram,'initialize',no_network)
    asyncio.run(system.start())
    assert system.replay_finished
    assert system.trade_report['lanes'][TYPES[0]]['bars'] == 4


def _rewrite_bar_timestamp_type(ds, unit, tz, remainder=0):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tests.paired_fixture import rewrite_artifact
    path = ds.bar_path(TYPES[0])
    table = pq.read_table(path)
    ms = table['Date Time'].cast(pa.int64()).to_numpy()//1000
    raw = ms*{'us': 1000, 'ns': 1000000}[unit]+remainder
    table = table.set_column(table.schema.get_field_index('Date Time'), 'Date Time',
                             pa.array(raw, type=pa.timestamp(unit, tz=tz)))
    pq.write_table(table, path, row_group_size=7)
    root = rewrite_artifact(ds.directory, path.relative_to(ds.directory).as_posix(), path.read_bytes(), consistent=True)
    return PairedDataset.open(ds.directory, root)


def test_lane_end_naive_timestamps_ignore_process_timezone(tmp_path, monkeypatch):
    import os
    import time
    ds, _ = setup(tmp_path, ends={TYPES[0]: list(range(4, 299, 3))})
    ds = _rewrite_bar_timestamp_type(ds, 'us', None)
    original_tz = os.environ.get('TZ')
    try:
        ends = []
        for zone in ('UTC', 'America/New_York'):
            monkeypatch.setenv('TZ', zone)
            time.tzset()
            expected = list(open_bars(ds.bars_root, 'BTCUSDT', TYPES[0], '2023-01-01', '2023-01-02', version=STAMP))[-1].seq
            lane = replay(ds, tmp_path, '2023-01-01', '2023-01-02', types=[TYPES[0]]).lanes[TYPES[0]]
            assert lane.last_tn == expected
            ends.append(lane.last_tn)
        assert ends[0] == ends[1]
    finally:
        if original_tz is None:
            monkeypatch.delenv('TZ', raising=False)
        else:
            monkeypatch.setenv('TZ', original_tz)
        time.tzset()


@pytest.mark.parametrize('unit', ['us', 'ns'])
def test_lane_end_preserves_reader_submillisecond_conversion(tmp_path, unit):
    ds, _ = setup(tmp_path, ends={TYPES[0]: list(range(4, 299, 3))})
    ds = _rewrite_bar_timestamp_type(ds, unit, 'UTC', remainder=321)
    reader = open_bars(ds.bars_root, 'BTCUSDT', TYPES[0], '2023-01-01', '2023-01-02', version=STAMP)
    expected = list(reader)[-1].seq
    r = replay(ds, tmp_path, '2023-01-01', '2023-01-02', types=[TYPES[0]])
    assert r.lanes[TYPES[0]].last_tn == expected
    list(r)


def test_window_warmup_verifies_leading_empty_hours(tmp_path):
    import hashlib
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pathlib import Path
    from tests.paired_fixture import rewrite_artifact
    trades = make_trades(days=3, per_day=100)
    times = trades.trade_time_ms.to_numpy().copy()
    day = day_ms(DAY0)+86400000
    times[100:200] = day+6*3600000+(times[100:200]-day)//2
    trades['trade_time_ms'] = times
    ds, _ = setup(tmp_path, trades, ends={TYPES[0]: [50, 150, 210, 270]})
    manifest = ds.manifest
    f = manifest['files'][1]
    path = Path(manifest['source_root'])/f['path']
    df = pq.read_table(path).to_pandas().drop(columns=['date', 'symbol'], errors='ignore')
    extra = df.iloc[0].copy()
    extra['trade_time_ms'], extra['trade_id'] = day+1000, 10000
    df.loc[len(df)] = extra
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    f.update(size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), min_time=day+1000)
    root = rewrite_artifact(ds.directory, 'trade_manifest.json', json.dumps(manifest).encode(), consistent=True)
    ds = PairedDataset.open(ds.directory, root)
    r = replay(ds, tmp_path, '2023-01-03', '2023-01-04', types=[TYPES[0]])
    with pytest.raises(DataIntegrityError, match='hour'):
        list(r)


@pytest.mark.parametrize('window', [{'start': '2023-01-02'}, {'end': '2023-01-03'}])
def test_certificate_refuses_window_before_reading(monkeypatch, window):
    from scripts import replay_trades as script
    monkeypatch.setattr(script, 'open_registered', lambda *a, **kw: pytest.fail('dataset read before argument refusal'))
    monkeypatch.setattr(script, 'open_bars', lambda *a, **kw: pytest.fail('bar read before argument refusal'))
    with pytest.raises(ValueError, match='certificate'):
        script.run('dataset', certificate='certificate.json', **window)


def test_trade_cli_requires_no_dashboard():
    from orderflow_system.main import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--trades', '--paired-dataset', 'dataset'])
    args = parse_args(['--trades', '--paired-dataset', 'dataset', '--no-dashboard'])
    assert args.trades and args.no_dashboard


def test_replay_script_loads_and_runs_without_psutil(tmp_path, monkeypatch):
    import runpy
    import sys
    from pathlib import Path
    monkeypatch.setitem(sys.modules, 'psutil', None)
    script = Path(__file__).resolve().parents[1]/'scripts'/'replay_trades.py'
    namespace = runpy.run_path(str(script), run_name='replay_without_psutil')
    ds, _ = setup(tmp_path)
    run = namespace['run']
    monkeypatch.setitem(run.__globals__, 'open_registered', lambda *a, **kw: ds)
    config = namespace['TradeFlowConfig'](duckdb_temp_dir=str(tmp_path/'spill'))
    report = run(ds.dataset_id, config=config, bar_types=TYPES)
    assert report['status'] == 'passed'
    assert report['rss_kind'] == report['stream']['rss_kind'] == 'peak'
    assert all(s['rss_kind'] == 'peak' and s['bytes'] > 0 for s in report['stream']['per_day_rss'])
    assert report['versions']['psutil'] is None
    assert report['hardware']['cores'] is None and report['hardware']['ram_bytes'] is None
    assert set(report['unavailable']) == {'versions.psutil', 'hardware.cores', 'hardware.ram_bytes'}


def test_trade_main_import_is_lazy_in_isolated_process(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    script = '''
import asyncio
import sys
sys.path.insert(0, sys.argv[1])
blocked = {'resource', 'orderflow_system.data.trade_stream',
           'orderflow_system.data.trade_alignment', 'orderflow_system.data.paired_dataset'}
attempts = []
class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            attempts.append(fullname)
            raise ModuleNotFoundError('blocked: '+fullname, name=fullname)
sys.meta_path.insert(0, Blocker())
from orderflow_system import main
assert not main.parse_args([]).trades
assert not (blocked & sys.modules.keys())
args = main.parse_args(['--trades', '--paired-dataset', 'fixture', '--no-dashboard'])
try:
    asyncio.run(main.OrderflowSystem(paired_dataset=args.paired_dataset,
        trade_config=main.TradeFlowConfig(report_dir=sys.argv[2])).start())
except ModuleNotFoundError as exc:
    assert exc.name == 'orderflow_system.data.paired_dataset'
else:
    raise AssertionError('trade execution did not exercise the lazy import')
assert 'orderflow_system.data.paired_dataset' in attempts
'''
    result = subprocess.run([sys.executable, '-I', '-c', script, str(Path(__file__).resolve().parents[1]),
                             str(tmp_path/'reports')],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr


def _main_trade_fixture(tmp_path, monkeypatch):
    from orderflow_system import main as app
    from orderflow_system.config.settings import TradeFlowConfig
    from orderflow_system.data import paired_dataset
    ds, _ = setup(tmp_path, ends={TYPES[0]: [4, 99, 101, 270]})
    monkeypatch.setattr(paired_dataset, 'open_registered', lambda *a, **kw: ds)
    monkeypatch.setattr(app.DASHBOARD, 'enabled', False)
    system = app.OrderflowSystem(
        info_bars=replace(app.INFO_BARS, root=str(ds.bars_root), version=STAMP),
        paired_dataset=ds.dataset_id,
        trade_config=TradeFlowConfig(duckdb_temp_dir=str(tmp_path/'spill'), report_dir=str(tmp_path/'reports')),
    )
    async def no_network():
        pass
    monkeypatch.setattr(system.telegram, 'initialize', no_network)
    return app, ds, system, tmp_path/'reports'


def test_trade_report_is_initially_none(tmp_path, monkeypatch):
    _, _, system, _ = _main_trade_fixture(tmp_path, monkeypatch)
    assert system.trade_report is None


@pytest.mark.parametrize('outcome', ['passed', 'stopped', 'initially_stopped', 'integrity_failed', 'callback_failed'])
def test_trade_main_writes_one_terminal_report(tmp_path, monkeypatch, caplog, outcome):
    import json
    import logging
    app, ds, system, output = _main_trade_fixture(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO, logger=app.logger.name)
    system.dataset = ds
    system.reader = open_bars(ds.bars_root, 'BTCUSDT', TYPES[0], version=STAMP)
    system._running = outcome != 'initially_stopped'
    failure = RuntimeError('injected callback failure')
    original = system._on_bar
    if outcome in ('stopped', 'callback_failed'):
        async def on_bar(bar, result=None):
            await original(bar, result)
            if outcome == 'stopped':
                system._running = False
            else:
                raise failure
        monkeypatch.setattr(system, '_on_bar', on_bar)
    if outcome == 'integrity_failed':
        ds.hours.loc[:, 'digest'] = 'f'*64
    elsewhere = tmp_path/'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    caught = None
    if outcome.endswith('_failed'):
        expected = DataIntegrityError if outcome == 'integrity_failed' else RuntimeError
        with pytest.raises(expected) as error:
            asyncio.run(system._replay())
        caught = error.value
        if outcome == 'callback_failed':
            assert caught is failure
    else:
        asyncio.run(system._replay())
    files = list(output.glob(f'main-{ds.dataset_id}-{TYPES[0]}-*.json'))
    assert len(files) == 1
    report = json.loads(files[0].read_text())
    assert system.trade_report == report
    status = 'failed' if caught else ('passed' if outcome == 'passed' else 'stopped')
    assert report['status'] == status
    assert report['dataset_id'] == ds.dataset_id and report['root_hash'] == ds.root_hash
    assert report['bar_type'] == TYPES[0]
    assert report['trade_signals'] == asdict(system.trade_config.signals)
    assert report['effective_root'] == str(ds.bars_root.resolve())
    assert report['effective_version'] == STAMP
    expected_bars = {'passed': 4, 'stopped': 1, 'initially_stopped': 0, 'integrity_failed': 0, 'callback_failed': 1}[outcome]
    assert report['bars_processed'] == expected_bars == system.pipelines['BTCUSDT'].bars_processed
    assert isinstance(report['stream'], dict)
    if status == 'passed':
        lane = report['lanes'][TYPES[0]]
        assert lane['identities'] == {'I1': True, 'I2': True, 'I3': True, 'I4': True}
        assert lane['n_tail'] == 30 and lane['lots_tail'] > 0
        assert len(lane['trade_digest']) == len(lane['replay_digest']) == 64
        assert lane['developing_profiles'] and report['stream']['overscan_n'] == 0
        assert report['duckdb_settings']['threads'] == 4
    elif status == 'stopped':
        assert report['identities'] == 'not evaluated (stopped before end)'
    else:
        assert report['error'] == {'type': type(caught).__name__, 'message': str(caught)}
    summaries = [r.getMessage() for r in caplog.records if r.getMessage().startswith('Trade replay status=')]
    assert len(summaries) == 1 and status in summaries[0] and str(files[0]) in summaries[0]
    assert not list((elsewhere/'output').glob('**/*.json'))


@pytest.mark.parametrize('source', ['cli', 'programmatic'])
@pytest.mark.parametrize('field', ['root', 'version'])
def test_trade_main_rejects_conflicting_dataset_selection(tmp_path, monkeypatch, source, field):
    app, ds, system, _ = _main_trade_fixture(tmp_path, monkeypatch)
    wrong = str(tmp_path/'wrong-bars') if field == 'root' else 'wrong-stamp'
    if source == 'cli':
        args = app.parse_args(['--trades', '--paired-dataset', ds.dataset_id, '--no-dashboard', '--'+field, wrong])
        system.info_bars = replace(app.INFO_BARS, root=args.root, version=args.version)
    else:
        system.info_bars = replace(app.INFO_BARS, **{field: wrong})
    monkeypatch.setattr(app, 'open_bars', lambda *a, **kw: pytest.fail('bar discovery before selection refusal'))
    with pytest.raises(ValueError) as error:
        asyncio.run(system.start())
    expected = str(ds.bars_root.resolve()) if field == 'root' else STAMP
    assert wrong in str(error.value) and expected in str(error.value)


@pytest.mark.parametrize('source', ['cli', 'programmatic'])
@pytest.mark.parametrize('explicit', [False, True])
def test_trade_main_accepts_dataset_selection(tmp_path, monkeypatch, caplog, source, explicit):
    import logging
    app, ds, system, output = _main_trade_fixture(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO, logger=app.logger.name)
    root = str(ds.bars_root/'BTCUSDT'/'..')
    if source == 'cli':
        argv = ['--trades', '--paired-dataset', ds.dataset_id, '--no-dashboard']
        if explicit:
            argv += ['--root', root, '--version', STAMP]
        args = app.parse_args(argv)
        system.info_bars = replace(app.INFO_BARS, root=args.root, version=args.version)
    else:
        system.info_bars = replace(app.INFO_BARS, root=root, version=STAMP) if explicit else replace(app.INFO_BARS)
    asyncio.run(system.start())
    assert system.info_bars.root == str(ds.bars_root.resolve()) and system.info_bars.version == STAMP
    assert system.trade_report['status'] == 'passed'
    assert system.trade_report['effective_root'] == str(ds.bars_root.resolve())
    assert system.trade_report['effective_version'] == STAMP
    assert system.trade_report['root_hash'] == ds.root_hash
    assert len(list(output.glob('*.json'))) == 1
    messages = '\n'.join(r.getMessage() for r in caplog.records)
    assert str(ds.bars_root.resolve()) in messages and STAMP in messages and ds.root_hash in messages


def test_trade_main_reports_stop_during_pause(tmp_path, monkeypatch):
    app, ds, system, output = _main_trade_fixture(tmp_path, monkeypatch)
    system.dataset = ds
    system.reader = open_bars(ds.bars_root, 'BTCUSDT', TYPES[0], version=STAMP)
    system._running = True
    async def stop_during_pause(delay):
        system._running = False
    monkeypatch.setattr(app.asyncio, 'sleep', stop_during_pause)
    asyncio.run(system._replay())
    assert system.trade_report['status'] == 'stopped'
    assert system.trade_report['bars_processed'] == 1
    assert len(list(output.glob('*.json'))) == 1


def test_trade_main_root_conflict_names_original_argument(tmp_path, monkeypatch):
    app, ds, system, _ = _main_trade_fixture(tmp_path, monkeypatch)
    requested = '../wrong-bars'
    system.info_bars = replace(app.INFO_BARS, root=requested)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app, 'open_bars', lambda *a, **kw: pytest.fail('bar discovery before root refusal'))
    with pytest.raises(ValueError) as error:
        asyncio.run(system.start())
    assert requested in str(error.value) and str(ds.bars_root.resolve()) in str(error.value)


def _memory_gate_samples(n_post=40, slope_mib=0):
    mib = 1 << 20
    return [{'day': 1000+i, 'bytes': (1500 if i == 0 else 1000+slope_mib*(i-1))*mib,
             'rss_kind': 'current', 'duckdb_memory_bytes': 0, 'malloc_in_use_bytes': 100*mib}
            for i in range(n_post+1)]


def _evaluate_memory_with_loads(samples, peak_bytes, rss_kind, **kwargs):
    from scripts.replay_trades import evaluate_memory
    loads = {sample['day']: 100 if i == 0 else 1 for i, sample in enumerate(samples)}
    return evaluate_memory(samples, peak_bytes, rss_kind, day_loads=loads, **kwargs)


def test_memory_criteria_flat_rss_passes_and_heap_is_diagnostic():
    mib = 1 << 20
    samples = _memory_gate_samples()
    samples[-1]['malloc_in_use_bytes'] = 200*mib
    result = _evaluate_memory_with_loads(samples, replay_script.PEAK_RSS_LIMIT_BYTES, 'current')
    assert result['verdict'] == 'pass'
    assert result['gate_peak_rss']['status'] == 'pass'
    assert result['gate_duckdb']['status'] == 'pass'
    assert result['gate_stability']['status'] == 'pass'
    assert result['gate_stability']['n_post_anchor'] == 40
    assert result['gate_stability']['slope_bytes_per_day'] == 0
    assert len(result['gate_stability']['rolling_30_day_slopes']) == 11
    assert result['diagnostic_live_heap']['flagged'] is True
    assert result['diagnostic_live_heap']['first_bytes'] == 100*mib
    assert result['diagnostic_live_heap']['last_bytes'] == 200*mib


def test_memory_criteria_positive_two_mib_daily_slope_fails():
    result = _evaluate_memory_with_loads(_memory_gate_samples(slope_mib=2), 1600*(1 << 20), 'current')
    assert result['verdict'] == 'fail'
    assert result['gate_stability']['status'] == 'fail'
    assert result['gate_stability']['slope_bytes_per_day'] == pytest.approx(2*(1 << 20))


@pytest.mark.parametrize('n_post', [0, 29])
def test_memory_criteria_requires_thirty_post_anchor_samples(n_post):
    result = _evaluate_memory_with_loads(_memory_gate_samples(n_post=n_post), 1600*(1 << 20), 'current')
    assert result['verdict'] == 'fail'
    assert result['gate_stability']['status'] == 'not_evaluable'
    assert result['gate_stability']['n_post_anchor'] == n_post
    assert result['gate_stability']['reason']


def test_memory_criteria_peak_samples_are_not_evaluable():
    samples = _memory_gate_samples()
    for sample in samples:
        sample['rss_kind'] = 'peak'
    result = _evaluate_memory_with_loads(samples, 1600*(1 << 20), 'peak')
    assert result['verdict'] == 'fail'
    assert result['gate_stability']['status'] == 'not_evaluable'
    assert result['gate_stability']['reason']


def test_memory_criteria_duckdb_sixty_five_mib_fails():
    samples = _memory_gate_samples()
    samples[-1]['duckdb_memory_bytes'] = 65*(1 << 20)
    result = _evaluate_memory_with_loads(samples, 1600*(1 << 20), 'current')
    assert result['verdict'] == 'fail'
    assert result['gate_duckdb']['status'] == 'fail'
    assert result['gate_duckdb']['max_bytes'] == 65*(1 << 20)


def test_memory_criteria_missing_duckdb_is_not_evaluable():
    samples = _memory_gate_samples()
    del samples[-1]['duckdb_memory_bytes']
    result = _evaluate_memory_with_loads(samples, 1600*(1 << 20), 'current')
    assert result['verdict'] == 'fail'
    assert result['gate_duckdb']['status'] == 'not_evaluable'
    assert result['gate_duckdb']['reason']


def test_memory_criteria_observed_attempt_two_peak_passes():
    result = _evaluate_memory_with_loads(_memory_gate_samples(), 2_181_775_360, 'current')
    assert result['verdict'] == 'pass'
    assert result['gate_peak_rss']['status'] == 'pass'
    assert result['gate_peak_rss']['limit_bytes'] == 4_294_967_296


def test_memory_criteria_peak_breach_cannot_be_excused_by_stability():
    result = _evaluate_memory_with_loads(_memory_gate_samples(), replay_script.PEAK_RSS_LIMIT_BYTES+1, 'current')
    assert result['verdict'] == 'fail'
    assert result['gate_stability']['status'] == 'pass'
    assert result['gate_peak_rss']['status'] == 'fail'
    assert result['gate_peak_rss']['reason'] == 'process peak RSS exceeds 4 GiB'


@pytest.mark.parametrize('verdict,exit_code', [('pass', 0), ('fail', 1)])
def test_replay_cli_enforces_memory_after_writing_report(tmp_path, monkeypatch, verdict, exit_code):
    import json
    from scripts import replay_trades as script
    monkeypatch.setattr(script, '__file__', str(tmp_path/'scripts'/'replay_trades.py'))
    report = {'status': 'passed', 'wall_seconds': 1., 'memory_criteria': {'verdict': verdict}}
    monkeypatch.setattr(script, 'run', lambda *a, **kw: report)
    if exit_code:
        with pytest.raises(SystemExit) as error:
            script.main(['--paired-dataset', 'fixture', '--enforce-memory-gates'])
        assert error.value.code == exit_code
    else:
        assert script.main(['--paired-dataset', 'fixture', '--enforce-memory-gates']) == report
    files = list((tmp_path/'output'/'trade_replay').glob('*.json'))
    assert len(files) == 1 and json.loads(files[0].read_text()) == report


def test_replay_cli_reports_memory_without_enforcement(tmp_path, monkeypatch):
    import json
    from scripts import replay_trades as script
    ds, _ = setup(tmp_path)
    config_class = script.TradeFlowConfig
    monkeypatch.setattr(script, 'TradeFlowConfig', lambda **kw: config_class(
        duckdb_temp_dir=str(tmp_path/'spill'), report_dir=str(tmp_path/'reports'), **kw))
    monkeypatch.setattr(script, '__file__', str(tmp_path/'scripts'/'replay_trades.py'))
    monkeypatch.setattr(script, 'open_registered', lambda *a, **kw: ds)
    monkeypatch.setattr(script, 'peak_rss_bytes', lambda: 123456, raising=False)
    report = script.main(['--paired-dataset', ds.dataset_id])
    files = list((tmp_path/'output'/'trade_replay').glob('*.json'))
    assert len(files) == 1
    assert json.loads(files[0].read_text()) == report
    assert report['memory_criteria']['gate_peak_rss']['peak_rss_bytes'] == 123456
    assert report['memory_criteria']['gate_duckdb']['status'] == 'pass'
    assert report['memory_criteria']['gate_stability']['status'] == 'not_evaluable'
    assert report['memory_criteria']['verdict'] == 'fail'


def test_memory_criteria_exact_limits_and_heaviest_anchor_pass():
    samples = _memory_gate_samples(n_post=30)
    for sample in samples:
        sample['bytes'] = 1000*(1 << 20)
        sample['duckdb_memory_bytes'] = 64*(1 << 20)
    samples[-1]['malloc_in_use_bytes'] = 160*(1 << 20)
    result = _evaluate_memory_with_loads(samples, replay_script.PEAK_RSS_LIMIT_BYTES, 'current')
    assert result['verdict'] == 'pass'
    assert result['gate_peak_rss']['status'] == 'pass'
    assert result['gate_peak_rss']['limit_bytes'] == replay_script.PEAK_RSS_LIMIT_BYTES == 4_294_967_296
    assert result['gate_duckdb']['limit_bytes'] == 64*(1 << 20)
    assert result['gate_stability']['limit_bytes_per_day'] == 1 << 20
    assert result['gate_stability']['n_post_anchor'] == 30
    assert result['diagnostic_live_heap']['flagged'] is False


def test_memory_criteria_rolling_slopes_are_diagnostic_only():
    import math
    samples = _memory_gate_samples(n_post=120)
    for i, sample in enumerate(samples[1:]):
        sample['bytes'] += round(60*math.sin(2*math.pi*i/40)*(1 << 20))
    result = _evaluate_memory_with_loads(samples, replay_script.PEAK_RSS_LIMIT_BYTES, 'current')
    assert result['verdict'] == 'pass'
    assert any(s['slope_bytes_per_day'] > (1 << 20) for s in result['gate_stability']['rolling_30_day_slopes'])


def test_memory_criteria_ols_uses_elapsed_utc_days():
    samples = _memory_gate_samples(slope_mib=1.5)
    for i, sample in enumerate(samples):
        sample['day'] = 1000+2*i
    result = _evaluate_memory_with_loads(samples, replay_script.PEAK_RSS_LIMIT_BYTES, 'current')
    assert result['verdict'] == 'pass'
    assert result['gate_stability']['slope_bytes_per_day'] == pytest.approx(.75*(1 << 20))


def test_memory_criteria_window_memory_is_included():
    extra = [{'duckdb_memory_bytes': 65*(1 << 20), 'malloc_in_use_bytes': 200*(1 << 20)}]
    result = _evaluate_memory_with_loads(_memory_gate_samples(), replay_script.PEAK_RSS_LIMIT_BYTES, 'current', extra_samples=extra)
    assert result['verdict'] == 'fail'
    assert result['gate_duckdb']['status'] == 'fail'
    assert result['gate_stability']['status'] == 'pass'
    assert result['gate_stability']['n_post_anchor'] == 40
    assert result['gate_duckdb']['sample_scope'] == 'main_replay_and_windows'
    assert result['diagnostic_live_heap']['flagged'] is False
    assert result['diagnostic_live_heap']['first_bytes'] == 100*(1 << 20)
    assert result['diagnostic_live_heap']['last_bytes'] == 100*(1 << 20)
    assert result['diagnostic_live_heap']['windows_max_malloc_in_use_bytes'] == 200*(1 << 20)


def test_live_heap_main_endpoints_and_window_max_are_separate():
    samples = _memory_gate_samples()
    samples[-1]['malloc_in_use_bytes'] = 200*(1 << 20)
    extra = [{'duckdb_memory_bytes': 0, 'malloc_in_use_bytes': value}
             for value in [300*(1 << 20), None, 50*(1 << 20)]]
    result = _evaluate_memory_with_loads(samples, replay_script.PEAK_RSS_LIMIT_BYTES, 'current', extra_samples=extra)
    heap = result['diagnostic_live_heap']
    assert heap['first_bytes'] == 100*(1 << 20)
    assert heap['last_bytes'] == 200*(1 << 20)
    assert heap['flagged'] is True
    assert heap['windows_max_malloc_in_use_bytes'] == 300*(1 << 20)
    assert result['verdict'] == 'pass'


def test_live_heap_without_main_samples_keeps_endpoints_unavailable():
    extra = [{'duckdb_memory_bytes': 0, 'malloc_in_use_bytes': 123}]
    result = _evaluate_memory_with_loads([], replay_script.PEAK_RSS_LIMIT_BYTES, 'current', extra_samples=extra)
    heap = result['diagnostic_live_heap']
    assert heap['first_bytes'] is None and heap['last_bytes'] is None
    assert heap['flagged'] is None
    assert heap['windows_max_malloc_in_use_bytes'] == 123


@pytest.mark.parametrize('extra', [[], [{'duckdb_memory_bytes': 0, 'malloc_in_use_bytes': None}]])
def test_live_heap_window_max_unavailable_without_window_malloc(extra):
    result = _evaluate_memory_with_loads(_memory_gate_samples(), replay_script.PEAK_RSS_LIMIT_BYTES, 'current', extra_samples=extra)
    assert result['diagnostic_live_heap']['windows_max_malloc_in_use_bytes'] is None


def test_memory_criteria_missing_day_samples_cannot_pass():
    result = _evaluate_memory_with_loads([], 1 << 30, 'current')
    assert result['verdict'] == 'fail'
    assert result['gate_duckdb']['status'] == 'not_evaluable'
    assert result['gate_stability']['status'] == 'not_evaluable'
    assert result['diagnostic_live_heap']['flagged'] is None


@pytest.mark.parametrize('gate', ['gate_peak_rss', 'gate_duckdb', 'gate_stability'])
def test_memory_criteria_exceeded_gate_has_reason(gate):
    samples = _memory_gate_samples(slope_mib=2 if gate == 'gate_stability' else 0)
    if gate == 'gate_duckdb':
        samples[-1]['duckdb_memory_bytes'] = 65*(1 << 20)
    peak = replay_script.PEAK_RSS_LIMIT_BYTES+1 if gate == 'gate_peak_rss' else replay_script.PEAK_RSS_LIMIT_BYTES
    result = _evaluate_memory_with_loads(samples, peak, 'current')
    assert result[gate]['status'] == 'fail'
    assert result[gate]['reason']


def test_memory_stability_early_heaviest_day_with_late_rss_maximum():
    from scripts.replay_trades import evaluate_memory
    samples = _memory_gate_samples(n_post=60)
    samples[-1]['bytes'] = 1600*(1 << 20)
    loads = {s['day']: 100 if i == 0 else 1 for i, s in enumerate(samples)}
    result = evaluate_memory(samples, replay_script.PEAK_RSS_LIMIT_BYTES, 'current', day_loads=loads)
    stability = result['gate_stability']
    assert stability['anchor_day'] == datetime.fromtimestamp(1000*86400, timezone.utc).date().isoformat()
    assert stability['anchor_n_trades'] == 100
    assert stability['anchor_source'] == 'hours.parquet n_trades'
    assert stability['sample_scope'] == 'main_replay'
    assert stability['n_post_anchor'] == 60
    assert stability['status'] == 'pass' and result['verdict'] == 'pass'
    assert result['rss_max_anchor']['n_post_peak'] == 0
    assert result['rss_max_anchor']['status'] == 'not_evaluable'
    assert result['rss_max_anchor']['reason']


@pytest.mark.parametrize('n_post', [0, 28, 29])
def test_memory_stability_late_heaviest_day_not_evaluable(n_post):
    from scripts.replay_trades import evaluate_memory
    samples = _memory_gate_samples(n_post=60)
    loads = {s['day']: 1 for s in samples}
    anchor = samples[-n_post-1]['day']
    loads[anchor] = 100
    result = evaluate_memory(samples, replay_script.PEAK_RSS_LIMIT_BYTES, 'current', day_loads=loads)
    stability = result['gate_stability']
    assert stability['anchor_day'] == datetime.fromtimestamp(anchor*86400, timezone.utc).date().isoformat()
    assert stability['n_post_anchor'] == n_post
    assert stability['status'] == 'not_evaluable' and stability['reason']
    assert result['verdict'] == 'fail'
    assert result['rss_max_anchor']['status'] == 'pass'


def test_memory_stability_load_tie_chooses_earliest_day():
    from scripts.replay_trades import evaluate_memory
    samples = _memory_gate_samples()
    loads = {s['day']: 1 for s in samples}
    loads[1002] = loads[1005] = 200
    result = evaluate_memory(samples, replay_script.PEAK_RSS_LIMIT_BYTES, 'current', day_loads=loads)
    stability = result['gate_stability']
    assert stability['anchor_day'] == datetime.fromtimestamp(1002*86400, timezone.utc).date().isoformat()
    assert stability['anchor_n_trades'] == 200
    assert stability['n_post_anchor'] == 38
    assert stability['status'] == 'pass'


def test_memory_stability_light_day_rss_spike_does_not_move_anchor():
    from scripts.replay_trades import evaluate_memory
    samples = _memory_gate_samples()
    samples[10]['bytes'] = 1900*(1 << 20)
    loads = {s['day']: 1 for s in samples}
    loads[1003] = 500
    loads[999] = loads[5000] = 1000000
    extra = [{'day': 5000, 'bytes': 2000*(1 << 20), 'duckdb_memory_bytes': 0, 'malloc_in_use_bytes': 100}]
    result = evaluate_memory(samples, replay_script.PEAK_RSS_LIMIT_BYTES, 'current', day_loads=loads, extra_samples=extra)
    stability = result['gate_stability']
    assert stability['anchor_day'] == datetime.fromtimestamp(1003*86400, timezone.utc).date().isoformat()
    assert stability['anchor_n_trades'] == 500
    assert stability['n_post_anchor'] == 37
    assert stability['status'] == 'pass'
    assert result['rss_max_anchor']['peak_day'] == 1010
    assert result['rss_max_anchor']['n_post_peak'] == 30


@pytest.mark.parametrize('missing', ['all', 'one'])
def test_memory_stability_missing_authenticated_loads_cannot_pass(missing):
    from scripts.replay_trades import evaluate_memory
    samples = _memory_gate_samples()
    loads = None if missing == 'all' else {s['day']: 1 for s in samples[:-1]}
    result = evaluate_memory(samples, replay_script.PEAK_RSS_LIMIT_BYTES, 'current', day_loads=loads)
    stability = result['gate_stability']
    assert stability['status'] == 'not_evaluable'
    assert stability['anchor_day'] is None and stability['anchor_n_trades'] is None
    assert stability['reason'] and result['verdict'] == 'fail'
    assert result['rss_max_anchor']['status'] == 'pass'


def test_replay_memory_anchor_sums_authenticated_hour_trade_counts(tmp_path, monkeypatch):
    from scripts import replay_trades as script
    trades = make_trades(days=3, per_day=120)
    trades = trades.drop(index=list(range(80, 120))+list(range(340, 360))).reset_index(drop=True)
    ds, _ = setup(tmp_path, trades=trades)
    monkeypatch.setattr(script, 'open_registered', lambda *a, **kw: ds)
    config = script.TradeFlowConfig(duckdb_temp_dir=str(tmp_path/'spill'), report_dir=str(tmp_path/'reports'))
    report = script.run(ds.dataset_id, config=config)
    stability = report['memory_criteria']['gate_stability']
    assert stability['anchor_day'] == '2023-01-02'
    assert stability['anchor_n_trades'] == 120
    assert stability['anchor_source'] == 'hours.parquet n_trades'
    assert stability['n_post_anchor'] == 1
    assert stability['status'] == 'not_evaluable'


def test_flags_off_noncanonical_details_preserve_certified_digests(tmp_path, monkeypatch):
    import json
    import numpy as np
    from unittest.mock import Mock
    from orderflow_system.data.models import Signal, SignalType, Side

    ds, _ = setup(tmp_path, ends={TYPES[0]: [4, 99, 101, 270]})
    digests = []
    for noncanonical in (True, False):
        pipe = InfoBarPipeline(get_btcusd_config())
        def emit(bar, *args, **kwargs):
            details = {'type': 'fixture'}
            if noncanonical:
                details.update(numpy_int=np.int64(7), nan=float('nan'))
            return Signal(bar.timestamp_ms, SignalType.ABSORPTION, Side.SELL,
                          bar.close, strength=50., details=details)
        monkeypatch.setattr(pipe.absorption, 'check_bar', emit)
        reader = open_bars(ds.bars_root, 'BTCUSDT', TYPES[0], version=STAMP)
        run = TradeReplay(ds, [(reader, pipe)], temp_dir=tmp_path/'spill')
        with monkeypatch.context() as scoped:
            serialize = Mock(wraps=json.dumps)
            scoped.setattr(json, 'dumps', serialize)
            results = list(run)
            serialize.assert_not_called()
        assert len(results) == 4
        if noncanonical:
            assert isinstance(results[0][1].signals[0].details['numpy_int'], np.int64)
            assert np.isnan(results[0][1].signals[0].details['nan'])
        lane = run.report['lanes'][TYPES[0]]
        assert not {'raw_signal_digest', 'raw_signal_digests', 'signal_detail_counts'} & lane.keys()
        digests.append((lane['replay_digest'], lane['trade_digest']))
    assert digests[0] == digests[1]


def test_flags_off_windows_never_build_raw_signal_digests(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from orderflow_system import pipeline
    from scripts import replay_trades as script

    ds, _ = setup(tmp_path)
    monkeypatch.setattr(script, 'open_registered', lambda *args: ds)
    raw = Mock(side_effect=AssertionError('flags-off raw digest construction'))
    monkeypatch.setattr(pipeline, 'RawSignalDigest', raw)
    report = script.run(ds.dataset_id, windows=2)
    raw.assert_not_called()
    for lane in report['lanes'].values():
        assert not {'raw_signal_digest', 'raw_signal_digests', 'signal_detail_counts'} & lane.keys()
        assert isinstance(lane['signal_counts'], dict)
    assert all(w['equal'] and all(not lane['raw_signal_equal'] for lane in w['lanes'].values())
               for w in report['windows'])


@pytest.mark.parametrize('detail', ['numpy_int', 'nan'])
def test_enabled_raw_signal_details_remain_strict(tmp_path, monkeypatch, detail):
    import numpy as np
    from orderflow_system.config.settings import TradeFlowConfig, TradeSignalConfig
    from orderflow_system.data.models import Signal, SignalType, Side

    ds, _ = setup(tmp_path, ends={TYPES[0]: [4, 99, 101, 270]})
    cfg = TradeFlowConfig(signals=TradeSignalConfig(absorption_levels=True))
    pipe = InfoBarPipeline(get_btcusd_config(), trade_config=cfg)
    value = np.int64(7) if detail == 'numpy_int' else float('nan')
    def emit(bar, *args, **kwargs):
        return Signal(bar.timestamp_ms, SignalType.ABSORPTION, Side.SELL,
                      bar.close, strength=50., details={detail: value})
    monkeypatch.setattr(pipe.absorption, 'check_bar', emit)
    reader = open_bars(ds.bars_root, 'BTCUSDT', TYPES[0], version=STAMP)
    run = TradeReplay(ds, [(reader, pipe)], temp_dir=tmp_path/'spill')
    with pytest.raises(TypeError if detail == 'numpy_int' else ValueError):
        list(run)


def test_raw_signal_work_is_enabled_per_lane(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from orderflow_system import pipeline
    from orderflow_system.config.settings import TradeFlowConfig, TradeSignalConfig

    ds, _ = setup(tmp_path)
    lanes = [(open_bars(ds.bars_root, 'BTCUSDT', bt, version=STAMP),
              InfoBarPipeline(get_btcusd_config(), trade_config=TradeFlowConfig(
                  signals=TradeSignalConfig(absorption_levels=bt == TYPES[0])))) for bt in TYPES]
    raw = Mock(wraps=pipeline.RawSignalDigest)
    monkeypatch.setattr(pipeline, 'RawSignalDigest', raw)
    run = TradeReplay(ds, lanes, temp_dir=tmp_path/'spill')
    list(run)
    assert raw.call_count == 3  # One full digest and two detector digests for the enabled lane.
    raw_keys = {'raw_signal_digest', 'raw_signal_digests', 'signal_detail_counts'}
    assert raw_keys <= run.report['lanes'][TYPES[0]].keys()
    assert not raw_keys & run.report['lanes'][TYPES[1]].keys()

"""Bounded development-only real proofs. No studies, tail looks, or full-history replay."""
import builtins
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from orderflow_system.config.settings import TradeFlowConfig, get_btcusd_config, parse_trade_signals
from orderflow_system.data.infobar_feed import open_bars
from orderflow_system.data.models import BarType
from orderflow_system.data.paired_dataset import open_registered
from orderflow_system.data.trade_stream import TradeReplay
from orderflow_system.pipeline import InfoBarPipeline
from scripts import optimize_orderflow as opt, trade_feature_cache as tfc
from scripts.phase4_policy import ROOT, canonical_bytes, bounded_preflight, protected_state

DATASET = '20261103T011635Z-d3a7b9603b37'
pytestmark = [pytest.mark.real_data,pytest.mark.skipif(not (Path(TradeFlowConfig().paired_root)/DATASET).is_dir(),
                                                     reason='certified paired dataset not mounted')]
ALL = 'absorption_levels,exhaustion_bonus,framing_trades'


@pytest.mark.parametrize('start,end', [('2021-05-19','2021-05-20'),('2019-09-11','2019-09-12'),
                                      ('2023-10-16','2023-10-17'),('2023-12-30','2024-01-01')])
def test_bounded_real_cache_and_live(start,end):
    protected_before = bounded_preflight()
    ds = open_registered(Path(TradeFlowConfig().paired_root),DATASET)
    forbidden = {str(Path(ds.manifest['source_root'])/f['path']) for f in ds.manifest['files'] if f['min_time']>=1704067200000}
    opened,queries,hashed = set(),[],set()
    def intercepted(original):
        def checked(path,*args,**kwargs):
            if not isinstance(path,int):
                opened.add(str(path))
                assert str(path) not in forbidden
            return original(path,*args,**kwargs)
        return checked
    import orderflow_system.data.trade_stream as stream
    original_hash = stream._check_file
    def hashing(f,seen):
        hashed.add(str(f['abs']))
        assert str(f['abs']) not in forbidden
        return original_hash(f,seen)
    output = ROOT/'output/phase4/real'/start
    with patch.object(builtins,'open',intercepted(builtins.open)), patch.object(io,'open',intercepted(io.open)), \
            patch.object(os,'open',intercepted(os.open)), patch.object(stream,'_check_file',hashing):
        paths,report = tfc.build(ds,output/'cache',start=start,end=end,query_observer=queries.extend)
        assert not forbidden.intersection(queries)
        assert set(queries)<=hashed
        assert len(report['stream']['excluded_by_policy'])==1
        refs = json.loads((ROOT/'tests/fixtures/phase4_reference_digests.json').read_text())
        verification = {}
        for bt,path in paths.items():
            cache = tfc.CachedTradeUpdates(path)
            bars = cache.bars()
            cache.validate_bars(bars)
            assert cache.manifest['trade_digest']==report['lanes'][bt]['trade_digest']
            # Compare against v0 bars on identical ordinals/timestamps, not just paired bars.
            v0 = list(open_bars(TradeFlowConfig().paired_root.replace('info-bars-paired','info-bars'),
                                 'BTCUSDT',bt,start,end,version='20261026T135000Z',strict=True))
            assert [(b.seq,b.timestamp_ms) for b in v0]==[(b.seq,b.timestamp_ms) for b in bars]
            v0_path=output/f'v0-{bt}.bin'
            opt.write_bar_stream(v0_path,iter(v0))
            cache.validate_bars(opt.CachedBars(v0_path,bt))
            verification[bt] = {}
            import struct
            maximum=int(cache.arrays['fp_0'].max(initial=0))
            assert struct.pack('d',round(maximum/100,10))==struct.pack('d',maximum/100)
            verification[bt]['max_tick_round_equality']={'max_tick':maximum,'equal_float_bits':True}
            for label,mode in [('off',''),('all',ALL)]:
                cfg = TradeFlowConfig(signals=parse_trade_signals(mode))
                result = opt.run_backtest(bars,get_btcusd_config(),.002,BarType(bt),trade_updates=cache,trade_config=cfg)
                if start=='2021-05-19':
                    expected = refs['bounded_'+label]['lanes'][bt]
                    assert len(bars)==expected['bars']
                    assert cache.manifest['trade_digest']==expected['trade_digest']
                    assert result['replay_digest']==expected['replay_digest']
                    assert result['signal_counts']==expected['signal_counts']
                    if label=='all':
                        assert result['raw_signal_digest']==expected['raw_signal_digest']
                else:
                    reader = open_bars(ds.bars_root,'BTCUSDT',bt,start,end,version=ds.manifest['stamp'])
                    live = TradeReplay(ds,[(reader,InfoBarPipeline(get_btcusd_config(),BarType(bt),trade_config=cfg))],
                        temp_dir=ROOT/'output/duckdb_tmp',development_only=True,query_observer=queries.extend)
                    for _ in live:
                        pass
                    expected = live.report['lanes'][bt]
                    assert cache.manifest['trade_digest']==expected['trade_digest']
                    assert result['replay_digest']==expected['replay_digest']
                    if mode:
                        assert result['raw_signal_digest']==expected['raw_signal_digest']
                verification[bt][label] = {k:result[k] for k in ('replay_digest','raw_signal_digest','signal_counts')}
            assert opt.run_backtest(v0,get_btcusd_config(),.002,BarType(bt))['replay_digest']==verification[bt]['off']['replay_digest']
        assert not forbidden.intersection(queries+list(opened))
    report['verification'] = verification
    report['protected_unchanged'] = protected_state() == protected_before
    assert report['protected_unchanged']
    report['open_interception'] = {'python_paths':sorted(opened),'duckdb_paths':sorted(set(queries)),
                                   'hashed_paths':sorted(hashed),'forbidden_paths':sorted(forbidden),
                                   'excluded_open_count':0,'all_queried_files_hashed':True}
    (output/'report.json').write_bytes(canonical_bytes(report))

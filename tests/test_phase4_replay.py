"""Live/cache equality and bar-only compatibility, red before T2."""
from pathlib import Path

import pytest

from orderflow_system.config.settings import TradeFlowConfig, get_btcusd_config, parse_trade_signals
from orderflow_system.data.infobar_feed import open_bars
from orderflow_system.data.models import BarType
from orderflow_system.data.trade_stream import TradeReplay
from orderflow_system.pipeline import InfoBarPipeline, ReplayDigest
from scripts import optimize_orderflow as opt, trade_feature_cache as tfc
from tests.test_phase4_cache import paired, BT


@pytest.mark.parametrize('mode', ['', 'absorption_levels,exhaustion_bonus,framing_trades'])
def test_cached_equals_live(tmp_path, monkeypatch, mode):
    ds = paired(tmp_path)
    monkeypatch.setattr(tfc, 'pinned_root', lambda _: ds.root_hash)
    monkeypatch.setattr(tfc, 'checked_write', lambda p: Path(p))
    paths, _ = tfc.build(ds, tmp_path/'cache', start='2023-01-01', end='2023-01-04')
    cache = tfc.CachedTradeUpdates(paths[BT])
    cfg = TradeFlowConfig(signals=parse_trade_signals(mode))
    reader = open_bars(ds.bars_root, 'BTCUSDT', BT, '2023-01-01', '2023-01-04', version=ds.manifest['stamp'])
    live = TradeReplay(ds, [(reader, InfoBarPipeline(get_btcusd_config(), BarType(BT), trade_config=cfg))],
                       temp_dir=tmp_path/'spill', development_only=True)
    for _ in live:
        pass
    result = opt.run_backtest(cache.bars(), get_btcusd_config(), .005, BarType(BT),
                               trade_updates=cache, trade_config=cfg)
    assert result['replay_digest'] == live.report['lanes'][BT]['replay_digest']
    if mode:
        assert result['raw_signal_digest'] == live.report['lanes'][BT]['raw_signal_digest']
    assert result['signal_counts'] == live.report['lanes'][BT]['signal_counts']
    old = InfoBarPipeline(get_btcusd_config(), BarType(BT))
    old.aggregator.price_proximity_pct = .005
    digest = ReplayDigest()
    for bar in cache.bars():
        digest.update(old.process_bar(bar))
    assert opt.run_backtest(cache.bars(), get_btcusd_config(), .005, BarType(BT),
                             trade_updates=None)['replay_digest'] == digest.hexdigest()


def test_bar_only_matches_base_commit_function():
    import ast
    import subprocess
    from test_pipeline import synth_bars
    source = subprocess.check_output(['git','show','390e29c:scripts/optimize_orderflow.py'],text=True)
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_backtest')
    namespace = dict(vars(opt))
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<base run_backtest>','exec'),namespace)
    bars = synth_bars(300)
    current = opt.run_backtest(bars,get_btcusd_config(),.002,BarType(BT))
    base = namespace['run_backtest'](bars,get_btcusd_config(),.002,BarType(BT))
    assert current['replay_digest']==base['replay_digest']
    assert current['prefix_digests']==base['prefix_digests']
    assert opt.closed_execution_rows(current['ledger']).equals(opt.closed_execution_rows(base['ledger']))

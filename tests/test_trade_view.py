"""Trade side outputs cannot feed legacy strategy state."""
from dataclasses import asdict

import numpy as np
import pytest

from orderflow_system.analytics.footprint import FootprintEngine
from orderflow_system.analytics.trade_accumulator import TradeAccumulator
from orderflow_system.analytics.volume_profile import VolumeProfileEngine
from orderflow_system.config.settings import get_btcusd_config, VolumeProfileConfig
from orderflow_system.data.infobar_feed import open_bars
from orderflow_system.data.models import Side, Tick
from orderflow_system.pipeline import InfoBarPipeline, ReplayDigest, TradeDigest, TradeView
from tests.paired_fixture import make_trades, write_paired, canonical, STAMP
from tests.test_trade_accumulator import rows


@pytest.mark.parametrize('width', [100, 500, 1000, 2500])
def test_legacy_whole_dollar_histogram_oracles(width):
    # Every tie and +/- two cents around each bucket boundary.
    p = np.array([q*width+offset for q in range(100, 120) for offset in
                  [-width//2-2, -width//2-1, -width//2, -width//2+1, -width//2+2, -2, -1, 0, 1, 2]])
    data = rows(p, maker=np.arange(len(p))%2)
    a = TradeAccumulator(width, width, 0)
    a.add(1, data)
    u = a.close_bar(len(p), int(data['time_ms'][-1]))
    ticks = [Tick(int(r['time_ms']), int(r['price_ticks'])/100, 1., Side.SELL if r['is_buyer_maker'] else Side.BUY)
             for r in data]
    legacy = FootprintEngine(width/100).build_from_ticks(ticks)
    assert {int(k): (int(b), int(s)) for k,b,s in zip(*u.footprint_levels)} == {
        round(k*100): (round(v.ask_volume*1000), round(v.bid_volume*1000)) for k,v in legacy.levels.items()}
    profile = VolumeProfileEngine(VolumeProfileConfig(tick_size=width/100)).compute_from_ticks(ticks)
    assert {int(k): int(v) for k,v in zip(*a.developing_profiles()[0][1:])} == {
        round(k*100): round(v*1000) for k,v in profile.volume_at_price.items()}


def test_pipeline_isolation_and_digest(tmp_path):
    trades = make_trades(days=3, per_day=100)
    directory, _ = write_paired(tmp_path, trades, {'fixed_dollar_imbalance': list(range(4, 291, 7))})
    bars = list(open_bars(directory, 'BTCUSDT', 'fixed_dollar_imbalance', version=STAMP))
    off, on = InfoBarPipeline(get_btcusd_config()), InfoBarPipeline(get_btcusd_config())
    od, nd, td = ReplayDigest(), ReplayDigest(), TradeDigest(ranges=[(bars[2].seq, bars[4].seq)])
    subset = TradeDigest()
    a = TradeAccumulator(100, 1000, 0)
    prev = 0
    for i, bar in enumerate(bars):
        a.add(prev+1, canonical(trades.iloc[prev:bar.seq]))
        update = a.close_bar(bar.seq, bar.timestamp_ms)
        plain, enriched = off.process_bar(bar), on.process_bar(bar, update)
        od.update(plain)
        nd.update(enriched)
        td.update(enriched, update)
        if 2 <= i <= 4:
            subset.update(enriched, update)
        assert enriched.trade_stats == update.stats
        assert enriched.trade_footprint.has_price_levels
        assert all(p.source == 'trades' for p in enriched.trade_profiles)
        # Check the complete legacy output, including actions and rollover bias.
        for key in ('trade_stats', 'trade_profiles', 'trade_footprint'):
            setattr(enriched, key, getattr(plain, key))
        assert asdict(plain) == asdict(enriched)
        prev = bar.seq
    assert od.hexdigest() == nd.hexdigest()
    assert td.range_hexdigest(bars[2].seq, bars[4].seq) == subset.hexdigest()
    assert on.footprint_engine.history == off.footprint_engine.history
    assert on.profile_framing._profile_history == off.profile_framing._profile_history
    assert on.delta_engine.cumulative_delta == off.delta_engine.cumulative_delta
    assert on.aggregator._watched_levels == off.aggregator._watched_levels
    assert on.aggregator._signal_history == off.aggregator._signal_history
    assert all(x._signal_history == y._signal_history for x,y in zip(on.detectors, off.detectors))
    assert not {'profile_framing', 'aggregator', 'detectors', 'delta_engine', 'bars', '_recent', '_session_bars'} & vars(on.trade_view).keys()


def test_garbage_trade_outputs_do_not_affect_strategy(tmp_path, monkeypatch):
    from orderflow_system.data.models import InfoBar
    bar = InfoBar(timestamp_ms=1, open=100., high=101., low=99., close=100., volume=1., buy_volume=.5,
                  sell_volume=.5, tick_count=1, dollar_value=100., seq=1)
    a = TradeAccumulator(100, 1000, 0)
    a.add(1, rows([10000], times=[1]))
    update = a.close_bar(1, 1)
    def garbage(self, bar, trade):
        return object(), object(), [object()]
    monkeypatch.setattr(TradeView, 'update', garbage)
    off, on = InfoBarPipeline(get_btcusd_config()), InfoBarPipeline(get_btcusd_config())
    d0, d1 = ReplayDigest(), ReplayDigest()
    d0.update(off.process_bar(bar)); d1.update(on.process_bar(bar, update))
    assert d0.hexdigest() == d1.hexdigest()


def test_garbage_side_outputs_during_active_strategy_and_rollovers(monkeypatch):
    from tests.test_pipeline import synth_bars
    # Exercise actual signals, actions and daily bias; a quiet single bar is insufficient.
    monkeypatch.setattr(TradeView, 'update', lambda *args: (object(), object(), [object()]))
    off,on=InfoBarPipeline(get_btcusd_config()),InfoBarPipeline(get_btcusd_config())
    d0,d1=ReplayDigest(),ReplayDigest()
    for bar in synth_bars(6000,seed=3):
        plain,enriched=off.process_bar(bar),on.process_bar(bar,object())
        d0.update(plain)
        d1.update(enriched)
        assert plain.signals==enriched.signals and plain.aggregated==enriched.aggregated
        assert plain.bias==enriched.bias and plain.closed_profile==enriched.closed_profile
        assert off.aggregator._watched_levels==on.aggregator._watched_levels
    assert off.signal_counts and off.action_counts and off.sessions_closed
    assert d0.hexdigest()==d1.hexdigest()
    assert off.footprint_engine.history==on.footprint_engine.history
    assert off.profile_framing._profile_history==on.profile_framing._profile_history
    assert off.delta_engine.cumulative_delta==on.delta_engine.cumulative_delta
    assert all(x._signal_history==y._signal_history for x,y in zip(off.detectors,on.detectors))


def _trade_profile_case(tmp_path, lots, pct, finalised=True, lvn=1.5):
    from dataclasses import replace
    from orderflow_system.config.settings import TradeFlowConfig
    from orderflow_system.data.paired_dataset import PairedDataset
    from orderflow_system.data.trade_stream import TradeReplay
    from tests.paired_fixture import DAY0, day_ms
    trades = make_trades(days=1, per_day=4).copy()
    trades['price'] = [100., 110., 120., 130.]
    trades['quantity'] = np.array([*lots, 1])/1000
    start = day_ms(DAY0)
    trades['trade_time_ms'] = [start, start+1, start+2, start+86400000]
    end = 4 if finalised else 3
    directory, root = write_paired(tmp_path, trades, {'fixed_dollar_imbalance': [end]}, days=2)
    ds = PairedDataset.open(directory, root)
    config = replace(get_btcusd_config(), volume_profile=VolumeProfileConfig(value_area_pct=pct, lvn_stddev_factor=lvn))
    pipe = InfoBarPipeline(config, trade_config=TradeFlowConfig())
    reader = open_bars(directory, 'BTCUSDT', 'fixed_dollar_imbalance', version=STAMP)
    replay = TradeReplay(ds, [(reader, pipe)], temp_dir=tmp_path/'spill')
    results = list(replay)
    return pipe, replay, results[0][1]


def test_trade_profile_inherits_pipeline_profile_settings(tmp_path):
    pipe, _, result = _trade_profile_case(tmp_path, [100, 200, 400], .5, lvn=1.)
    assert pipe.trade_view.profile_engine.config.value_area_pct == .5
    assert pipe.trade_view.profile_engine.config.lvn_stddev_factor == 1.
    profile = result.trade_profiles[0]
    assert (profile.poc, profile.vah, profile.val) == (120., 120., 120.)
    assert pipe.trade_view.profile_engine.config.tick_size == 10.
    assert pipe.trade_view.profile_engine.config is not pipe.config.volume_profile


def test_finalised_trade_value_area_uses_exact_decimal_threshold(tmp_path):
    _, _, result = _trade_profile_case(tmp_path, [100, 300, 850], .68)
    # The certified legacy float path is deliberately unchanged.
    legacy = VolumeProfileEngine(VolumeProfileConfig()).compute_from_volume_at_price(
        {100.: .1, 110.: .3, 120.: .85}, '2023-01-01', source='bar_vwap')
    assert (legacy.poc, legacy.vah, legacy.val) == (120., 120., 110.)
    profile = result.trade_profiles[0]
    assert (profile.poc, profile.vah, profile.val) == (120., 120., 120.)
    assert profile.total_volume == legacy.total_volume
    assert profile.shape == legacy.shape and profile.lvn_levels == legacy.lvn_levels
    assert profile.poc_position_pct == legacy.poc_position_pct


def test_developing_trade_value_area_uses_exact_decimal_threshold(tmp_path):
    _, replay, _ = _trade_profile_case(tmp_path, [100, 300, 850], .68, finalised=False)
    profile = replay.report['lanes']['fixed_dollar_imbalance']['developing_profiles'][0]
    assert (profile['poc'], profile['vah'], profile['val']) == (120., 120., 120.)
    assert profile['final'] is False


@pytest.mark.parametrize('finalised', [True, False])
def test_custom_half_value_area_is_exact_for_finalised_and_developing(tmp_path, finalised):
    _, replay, result = _trade_profile_case(tmp_path, [100, 200, 300], .5, finalised=finalised)
    if finalised:
        profile = result.trade_profiles[0]
        actual = profile.poc, profile.vah, profile.val
    else:
        profile = replay.report['lanes']['fixed_dollar_imbalance']['developing_profiles'][0]
        actual = profile['poc'], profile['vah'], profile['val']
    assert actual == (120., 120., 120.)


def test_trade_profile_refuses_float_poc_disagreement():
    from orderflow_system.config.settings import TradeFlowConfig
    from orderflow_system.data.infobar_feed import DataIntegrityError
    view = TradeView(TradeFlowConfig())
    # Float conversion collapses these distinct integer lots to a tie.
    with pytest.raises(DataIntegrityError, match='POC'):
        view.compute_profile(0, np.array([10000, 11000]), np.array([2**62, 2**62+1], dtype=np.int64))

"""Phase 4b C1–C5: synthetic opt-in strategy and integer geometry proofs."""
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from orderflow_system.analytics import footprint as geometry
from orderflow_system.analytics.delta import DeltaEngine
from orderflow_system.config.settings import TradeSignalConfig, TradeFlowConfig, get_btcusd_config
from orderflow_system.patterns.absorption import AbsorptionDetector
from orderflow_system.patterns.initiative import InitiativeDetector
from orderflow_system.pipeline import InfoBarPipeline
from scripts import optimize_orderflow as opt
from tests.test_trade_signals import bar, fp


@pytest.mark.parametrize('values', [
    {'absorption_source':'other'}, {'absorption_source':'trade'},
    {'initiative_imbalance':True}, {'initiative_imbalance':'other'},
    {'initiative_imbalance':'required'},
    *[{name:value} for name in ('imbalance_ratio','imbalance_min_level_share')
      for value in (True, np.bool_(True), '3', float('nan'), float('inf'), 0, -1)],
    {'imbalance_ratio':1}, {'imbalance_min_level_share':1.01},
    *[{'imbalance_bucket_mult':v} for v in (True, 1., 0, 3, 20)],
    *[{'stacked_min_levels':v} for v in (True, 1., -1, 5)],
])
def test_c1_validation(values):
    with pytest.raises(ValueError):
        TradeSignalConfig(**values)


def test_c1_defaults_and_initiative_only_guards():
    defaults = TradeSignalConfig()
    assert defaults.absorption_source == 'bar' and defaults.initiative_imbalance == 'off'
    assert (defaults.imbalance_ratio, defaults.imbalance_min_level_share,
            defaults.imbalance_bucket_mult, defaults.stacked_min_levels) == (3., .01, 1, 0)
    cfg = TradeFlowConfig(signals=TradeSignalConfig(initiative_imbalance='required', stacked_min_levels=1))
    assert cfg.signals.any_enabled
    assert opt.trade_signal_identity(cfg) != 'disabled'
    with pytest.raises(ValueError, match='trade input'):
        InfoBarPipeline(get_btcusd_config(), trade_config=cfg).process_bar(bar())


def test_c2_integer_aggregation_and_diagonal_oracle():
    # Raw cache geometry is cents, buy lots, sell lots. Above float's exact range.
    huge = 2**53 + 1
    levels = (np.array([100, 199, 200, -1]), [huge, 2, 9, 4], [1, 3, 8, 5])
    aggregated = geometry.aggregate_levels(levels, 1)
    assert aggregated == {-1:(5, 4), 1:(4, huge+2), 2:(8, 9)}  # bid, ask
    assert geometry.aggregate_levels(([], [], []), 10) == {}
    levels = {0:(10, 0), 1:(1, 40), 2:(15, 8), 4:(50, 0), 5:(0, 1)}
    # Interior gaps have zero volume; no pair crosses the present outer edges.
    assert geometry.diagonal_imbalances(levels, 3., .1, 125) == [(1,'buy'), (3,'sell'), (5,'sell')]
    assert geometry.diagonal_imbalances({0:(0, 0),1:(0, 1)}, 3., .1, 100) == []
    assert geometry.diagonal_imbalances({0:(0, 0),1:(0, 10)}, 3., .1, 100) == [(1,'buy')]
    assert geometry.diagonal_imbalances({}, 3., .1, 0) == []
    assert geometry.diagonal_imbalances({1:(0, 100)}, 3., .1, 100) == []


def test_c2_stacked_adjacency_oracle():
    pairs = [(0,'buy'),(1,'buy'),(2,'sell'),(4,'buy'),(5,'buy'),(6,'buy')]
    assert geometry.stacked_runs(pairs, 1) == {'buy':3, 'sell':1}
    assert geometry.stacked_runs(pairs, 2) == {'buy':3, 'sell':0}
    assert geometry.stacked_runs([], 0) == {'buy':0, 'sell':0}


def test_c2_diagonal_large_integer_boundary_is_exact():
    bid=2**53+1
    levels={0:(bid,0),1:(0,3*bid-1)}
    assert geometry.diagonal_imbalances(levels,3.,.01,4*bid)==[]
    levels[1]=(0,3*bid)
    assert geometry.diagonal_imbalances(levels,3.,.01,4*bid)==[(1,'buy')]


def initiative(levels, mode='required', k=2):
    b = bar(open=10000., close=10010., buy_volume=90., sell_volume=10.)
    cfg = replace(get_btcusd_config().initiative, volume_acceleration_min=1., min_body_bps=1.)
    detector = InitiativeDetector(cfg, signals=TradeSignalConfig(initiative_imbalance=mode, stacked_min_levels=k))
    class ForbiddenBarFootprint:
        @property
        def has_price_levels(self):
            pytest.fail('read bar footprint')
    return detector.check_bar(b, DeltaEngine().compute_from_bar(b), ForbiddenBarFootprint(),
        trade_footprint=fp(levels), trade_bucket_ticks=100)


def test_c3_trade_required_and_bonus_never_read_bar_footprint():
    assert initiative([(10000.,1,1),(10001.,40,1),(10002.,40,1)])
    assert initiative([(10000.,1,1),(10001.,1,1),(10002.,1,1)]) is None
    s = initiative([(10000.,1,1),(10001.,40,1),(10002.,40,1)])
    assert s.details['imbalance_source'] == 'trade'
    assert s.details['stacked_run'] == s.details['imbalance_levels'] == 2
    with pytest.raises(ValueError, match='trade footprint'):
        InitiativeDetector(get_btcusd_config().initiative, signals=TradeSignalConfig(
            initiative_imbalance='bonus')).check_bar(bar(), None, None)


def test_c4_initiative_bucket_integrity():
    from orderflow_system.data.infobar_feed import DataIntegrityError
    from types import SimpleNamespace
    cfg = TradeFlowConfig(signals=TradeSignalConfig(initiative_imbalance='bonus'))
    with pytest.raises(DataIntegrityError, match='bucket'):
        InfoBarPipeline(get_btcusd_config(), trade_config=cfg).process_bar(bar(), SimpleNamespace(fp_bucket_ticks=200))


def test_c5_trade_source_skips_m1_and_m2_and_matches_independent_a2():
    b = bar(open=10009., buy_volume=90., sell_volume=10.)
    delta, levels = DeltaEngine().compute_from_bar(b), fp([(10010.,90,10)])
    config = get_btcusd_config().absorption
    detector = AbsorptionDetector(config, signals=TradeSignalConfig(absorption_levels=True, absorption_source='trade'))
    oracle = AbsorptionDetector(config, signals=TradeSignalConfig(absorption_levels=True))
    expected = oracle._check_trade_absorption(b, levels, 100)
    detector._check_delta_close_mismatch = Mock(side_effect=AssertionError('M1'))
    detector._check_level_absorption = Mock(side_effect=AssertionError('M2'))
    actual = detector.check_bar(b, fp(), delta, b.close, trade_footprint=levels, trade_bucket_ticks=100)
    assert actual == expected and actual.details['type'] == 'trade_level_absorption'
    assert AbsorptionDetector(config).check_bar(b, fp(), delta, b.close).details['type'] != 'trade_level_absorption'


def test_c1_initiative_only_range_and_certificate_refusal():
    from orderflow_system.config.settings import validate_trade_signal_range
    from scripts.replay_trades import run
    cfg=TradeSignalConfig(initiative_imbalance='bonus')
    with pytest.raises(ValueError,match='development'):
        validate_trade_signal_range(cfg,'2024-01-01','2024-01-02')
    with pytest.raises(ValueError,match='certificate'):
        run('unused',config=TradeFlowConfig(signals=cfg),certificate='not-opened')


def test_c3_footprint_changes_break_even_trail_and_ledger_exit():
    from orderflow_system.analytics.trade_accumulator import TradeBarStats,TradeBarUpdate
    from orderflow_system.data.models import Signal,SignalType,Side
    from orderflow_system.ledger import TradeLedger
    from orderflow_system.signals.profile_framing import DailyBias,QualifiedLevel,LevelType
    def replay(stacked):
        base=get_btcusd_config()
        base=replace(base,initiative=replace(base.initiative,min_body_bps=1.,volume_acceleration_min=1.))
        pipeline=InfoBarPipeline(base,trade_config=TradeFlowConfig(signals=TradeSignalConfig(
            initiative_imbalance='required',stacked_min_levels=2)))
        pipeline.aggregator.min_composite_score=0
        pipeline.aggregator.signal_cooldown_seconds=0
        level=QualifiedLevel(10000.,LevelType.POC,Side.BUY,100)
        pipeline.profile_framing._bias_history.append(DailyBias('d',qualified_levels=[level]))
        pipeline.absorption.check_bar=lambda b,*a,**kw:Signal(b.timestamp_ms,SignalType.ABSORPTION,Side.BUY,10000.,100) if b.seq==1 else None
        pipeline.exhaustion.check_bar=lambda *a,**kw:None
        pipeline.divergence.check_bar=lambda *a,**kw:None
        ohlc=[(10000,10001,9999,10000),(10002,10011,10001,10010),
              (10012,10021,10011,10020),(10020,10021,9990,9995),(9995,10000,9960,9970)]
        ledger=TradeLedger();actions=[]
        for seq,(o,h,l,c) in enumerate(ohlc,1):
            b=bar(seq=seq,timestamp_ms=seq*1000,open=o,high=h,low=l,close=c,
                  buy_volume=90.,sell_volume=10.)
            lots=[10000,40000,40000] if stacked else [90000,0,0]
            update=TradeBarUpdate(TradeBarStats(*([0]*13)),
                (np.array([int(l*100),int(l*100)+100,int(l*100)+200]),
                 np.array(lots),np.array([3334,3333,3333])),[],100)
            result=pipeline.process_bar(b,update);ledger.process_result(result)
            actions.extend(a.action for a in result.aggregated)
        return actions,ledger
    active,yes=replay(True);inactive,no=replay(False)
    assert active==['enter','break_even','trail','stop']
    assert inactive==['enter','stop']
    assert yes.closes[0].execution.fill_price==10011.
    assert no.closes[0].execution.fill_price==9970.

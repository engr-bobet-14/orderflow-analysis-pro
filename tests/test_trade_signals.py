"""Frozen Phase 3: conservative aggressor geometry and opt-in strategy inputs."""
from dataclasses import asdict, fields, replace
from collections import UserString
from types import SimpleNamespace
from itertools import product
from unittest.mock import Mock

import numpy as np
import pytest

from orderflow_system.analytics.delta import DeltaEngine
from orderflow_system.analytics.footprint import FootprintBar, FootprintEngine
from orderflow_system.analytics.trade_accumulator import TradeAccumulator
from orderflow_system.config.settings import (
    AbsorptionConfig, ExhaustionConfig, TradeFlowConfig, TradeSignalConfig, get_btcusd_config,
)
from orderflow_system.data.infobar_feed import DataIntegrityError, open_bars
from orderflow_system.data.models import FootprintLevel, InfoBar, Side, SignalType
from orderflow_system.patterns.absorption import AbsorptionDetector
from orderflow_system.patterns.exhaustion import ExhaustionDetector
from orderflow_system.pipeline import InfoBarPipeline, RawSignalDigest, ReplayDigest, TradeView
from tests.paired_fixture import STAMP, canonical, make_trades, write_paired
from tests.test_pipeline import synth_bars


def bar(**kw):
    return replace(InfoBar(timestamp_ms=1, seq=1, open=10000., high=10010., low=10000.,
        close=10005., volume=100., buy_volume=50., sell_volume=50., tick_count=2,
        dollar_value=1000500., duration_ms=1000), **kw)


def fp(levels=()):
    return FootprintBar(levels={p: FootprintLevel(price=p, ask_volume=a, bid_volume=b) for p,a,b in levels})


def a2(b, levels, cfg=None, width=100):
    detector = AbsorptionDetector(AbsorptionConfig(), signals=cfg or TradeSignalConfig(absorption_levels=True))
    delta = DeltaEngine().compute_from_bar(b)
    return detector.check_bar(b, FootprintEngine().build_from_bar(b), delta, b.close,
                             trade_footprint=fp(levels), trade_bucket_ticks=width)


def mirror(b, levels):
    axis = 20010.
    return replace(b, open=axis-b.open, close=axis-b.close, high=axis-b.low, low=axis-b.high), [
        (axis-p, s, a) for p,a,s in levels]


def test_config_defaults_and_identity():
    assert not TradeFlowConfig().signals.any_enabled
    assert 'signals' not in asdict(get_btcusd_config())
    for cfg in (TradeSignalConfig(absorption_levels=True), TradeSignalConfig(exhaustion_contrarian='bonus'),
                TradeSignalConfig(exhaustion_contrarian='required'), TradeSignalConfig(framing_profile_source='trades')):
        assert cfg.any_enabled
        with pytest.raises(ValueError, match='trade'):
            InfoBarPipeline(get_btcusd_config(), trade_config=TradeFlowConfig(signals=cfg)).process_bar(bar())


@pytest.mark.parametrize('name,value', [
    ('exhaustion_contrarian','yes'), ('framing_profile_source','developing'),
    *[(name, v) for name in ('level_zone_frac','contrarian_zone_frac','min_level_share',
       'contrarian_min_level_share') for v in (0, -1, 1.01)],
    ('min_side_ratio',.5), ('min_side_ratio',1.1), ('max_through_bps',-1),
    ('contrarian_ratio',1), ('min_reject_bps',0), ('min_reject_bps',-1),
    *[(name,v) for name in ('level_zone_frac','contrarian_zone_frac','min_level_share','min_side_ratio',
       'min_reject_bps','max_through_bps','contrarian_ratio','contrarian_min_level_share') for v in (float('nan'),float('inf'))],
])
def test_config_rejects_invalid(name, value):
    with pytest.raises(ValueError):
        TradeSignalConfig(**{name:value})


@pytest.mark.parametrize('mirrored', [False, True])
@pytest.mark.parametrize('scale', [1e-3,1,1e3])
def test_a2_signal_and_volume_scale(mirrored, scale):
    b, levels = bar(), [(10010.,90*scale,10*scale)]
    if mirrored: b, levels = mirror(b,levels)
    s = a2(b,levels)
    assert s and s.direction == (Side.BUY if mirrored else Side.SELL)
    assert s.price_level == (10000.5 if mirrored else 10009.5)
    assert s.strength == 100
    assert s.details['delta_source']=='aggressor' and s.details['bucket']==1
    assert s.details['share']==1 and s.details['side_ratio']==pytest.approx(.9)
    assert b.low <= s.price_level <= b.high


@pytest.mark.parametrize('b,levels', [
    (bar(), []), (bar(), [(10010.,0,0)]), (bar(high=10000.,close=10000.), [(10000.,100,0)]),
    (bar(), [(10010.,1,0),(10005.,99,0)]), (bar(),[(10010.,60,40)]),
    (bar(close=10009.),[(10010.,100,0)]), (bar(high=10012.),[(10010.,100,0)]),
    (bar(high=10009.60,close=10007.80,low=10007.80),[(10010.,100,0)]),
    (bar(high=10010.49,low=10009.51,close=10009.51),[(10010.,100,0)]),
])
@pytest.mark.parametrize('mirrored', [False,True])
def test_a2_rejections(b,levels,mirrored):
    if mirrored: b,levels=mirror(b,levels)
    assert a2(b,levels) is None


@pytest.mark.parametrize('width', [None,0,-1,1.5,True])
@pytest.mark.parametrize('detector', ['a2','e2'])
def test_width_fails_closed(width,detector):
    b=bar()
    d=DeltaEngine(); delta=d.compute_from_bar(b)
    with pytest.raises(ValueError,match='bucket'):
        if detector=='a2': a2(b,[(10010.,100,0)],width=width)
        else:
            ExhaustionDetector(ExhaustionConfig(),signals=TradeSignalConfig(exhaustion_contrarian='bonus')).check_bar(
                b,delta,d,fp(),[b],trade_footprint=fp(),trade_bucket_ticks=width)


@pytest.mark.parametrize('mirrored',[False,True])
def test_wide_bucket_cannot_fabricate_rejection(mirrored):
    b,levels=bar(high=10010.4,close=10007.4),[(10010.,100,0)]
    if mirrored: b,levels=mirror(b,levels)
    assert a2(b,levels,width=1000) is None
    assert a2(b,levels,width=100) is not None  # the former hard-coded geometry is wrong


def test_odd_width_and_rejection_threshold():
    b=bar(high=10010.02)
    s=a2(b,[(10010.,100,0)],width=5)
    assert s.price_level==10009.975
    boundary=(10009.5-b.close)/b.close*1e4
    assert a2(b,[(10010.,100,0)],TradeSignalConfig(absorption_levels=True,min_reject_bps=boundary))
    assert a2(b,[(10010.,100,0)],TradeSignalConfig(absorption_levels=True,min_reject_bps=boundary+1e-6)) is None
    assert a2(b,[(10010.,100,0)],TradeSignalConfig(absorption_levels=True,min_reject_bps=boundary-1e-6))
    assert np.isfinite(a2(b,[(10010.,100,0)],TradeSignalConfig(absorption_levels=True,min_reject_bps=5e-324)).strength)


def test_a2_selection_ambiguity_and_m1_precedence():
    cfg=TradeSignalConfig(absorption_levels=True,max_through_bps=10)
    s=a2(bar(),[(10009.,50,0),(10010.,50,0)],cfg)
    assert s.details['bucket_centre']==10010.
    assert a2(bar(),[(10009.,60,0),(10010.,40,0)],cfg).details['bucket_centre']==10009.
    assert a2(bar(),[(10010.,50,0),(10000.,0,50)],cfg) is None
    b=bar(open=10010.,buy_volume=90.,sell_volume=10.)
    d=DeltaEngine().compute_from_bar(b); f=FootprintEngine().build_from_bar(b)
    legacy=AbsorptionDetector(AbsorptionConfig()).check_bar(b,f,d,b.close)
    assert a2(b,[(10000.,0,100)])==legacy


def exhaustion(mode, levels, mirrored=False, width=100, scale=1, legacy=False):
    bars=[bar(seq=i+1,timestamp_ms=(i+1)*1000,high=10006.+i,low=9999.+i,close=10005.+i,
              volume=(100.-15*i)*scale,buy_volume=(70.-11*i)*scale,
              sell_volume=(30.-4*i)*scale) for i in range(5)]
    if mirrored: bars=[mirror(b,[])[0] for b in bars]; levels=mirror(bar(),levels)[1] if levels is not None else None
    engine=DeltaEngine()
    for b in bars: delta=engine.compute_from_bar(b)
    det=ExhaustionDetector(ExhaustionConfig(min_bars_declining=3,requires_contrarian_imbalance=legacy),
                          signals=TradeSignalConfig(exhaustion_contrarian=mode))
    return det.check_bar(bars[-1],delta,engine,fp(),bars,
                        trade_footprint=None if levels is None else fp([(p,a*scale,b*scale) for p,a,b in levels]),
                        trade_bucket_ticks=width)


@pytest.mark.parametrize('mirrored',[False,True])
@pytest.mark.parametrize('scale',[1e-3,1,1e3])
def test_e2_bonus_required_unknown_and_scale(mirrored,scale):
    levels=[(10010.,0,100)]
    base=exhaustion('off',None,mirrored,scale=scale)
    assert base
    on=exhaustion('bonus',levels,mirrored,scale=scale)
    assert on.strength==pytest.approx(base.strength+20)
    assert on.details['delta_source']=='tick_rule' and on.details['contrarian_source']=='aggressor'
    assert on.details['contrarian_levels']==1 and on.details['contrarian'] is True
    assert exhaustion('required',levels,mirrored,scale=scale)
    for missing in (None,[]):
        assert exhaustion('bonus',missing,mirrored,scale=scale).strength==base.strength
        assert exhaustion('bonus',missing,mirrored,scale=scale).details['contrarian'] is None
        assert exhaustion('required',missing,mirrored,scale=scale) is None
    assert exhaustion('required',[(10010.,100,0)],mirrored,scale=scale) is None
    assert exhaustion('off',None,mirrored,scale=scale,legacy=True)==base


@pytest.mark.parametrize('mirrored',[False,True])
@pytest.mark.parametrize('width',[100,1000])
def test_e2_straddling_bucket_never_confirms(mirrored,width):
    # A bucket covering the entire narrow bar cannot establish aggression at its extreme.
    b=bar(low=10009.51,high=10010.49,close=10010.)
    if mirrored: b=mirror(b,[])[0]
    d=ExhaustionDetector(ExhaustionConfig(),signals=TradeSignalConfig(exhaustion_contrarian='bonus'))
    levels=[(10010.,1,100)]
    if mirrored: levels=mirror(bar(),levels)[1]
    assert d._trade_contrarian(b,fp(levels),width,Side.BUY if mirrored else Side.SELL)==0


def test_pipeline_only_enabled_detector_receives_trade(monkeypatch):
    for a,e in product((False,True),('off','bonus','required')):
        cfg=TradeFlowConfig(footprint_bucket=.05,signals=TradeSignalConfig(absorption_levels=a,exhaustion_contrarian=e))
        p=InfoBarPipeline(get_btcusd_config(),trade_config=cfg)
        for detector in p.detectors: monkeypatch.setattr(detector,'check_bar',Mock(return_value=None))
        update=SimpleNamespace(fp_bucket_ticks=5); footprint=fp([(10010.,100,0)])
        spy=Mock(return_value=(footprint,object(),[])); monkeypatch.setattr(TradeView,'update',spy)
        p.process_bar(bar(),update)
        spy.assert_called_once()
        for detector,enabled in ((p.absorption,a),(p.exhaustion,e!='off'),(p.initiative,False),(p.divergence,False)):
            kw=detector.check_bar.call_args.kwargs
            assert kw==({'trade_footprint':footprint,'trade_bucket_ticks':5} if enabled else {})


def paired_updates(tmp_path):
    trades=make_trades(days=4,per_day=40)
    directory,_=write_paired(tmp_path,trades,{'fixed_dollar_imbalance':[45,60,125,130,150]},days=4)
    bars=list(open_bars(directory,'BTCUSDT','fixed_dollar_imbalance',version=STAMP))
    accumulator=TradeAccumulator(100,1000,0); previous=0; pairs=[]
    for b in bars:
        accumulator.add(previous+1,canonical(trades.iloc[previous:b.seq]))
        pairs.append((b,accumulator.close_bar(b.seq,b.timestamp_ms))); previous=b.seq
    return pairs


@pytest.mark.parametrize('a,e,p2',list(product((False,True),('off','bonus','required'),(False,True))))
def test_prefix_invariance_and_delta_separation(tmp_path,a,e,p2):
    pairs=paired_updates(tmp_path)
    config=TradeFlowConfig(signals=TradeSignalConfig(absorption_levels=a,exhaustion_contrarian=e,
        framing_profile_source='trades' if p2 else 'bar_vwap'))
    def run(pairs):
        p=InfoBarPipeline(get_btcusd_config(),trade_config=config)
        return [p.process_bar(b,u) for b,u in pairs]
    full=run(pairs)
    for cut in (1,3,4): assert full[:cut]==run(pairs[:cut])
    legacy=InfoBarPipeline(get_btcusd_config())
    assert [r.delta for r in full]==[legacy.process_bar(b).delta for b,_ in pairs]


def test_p2_order_timing_first_and_later_multiday(tmp_path,monkeypatch):
    pairs=paired_updates(tmp_path)
    p=InfoBarPipeline(get_btcusd_config(),trade_config=TradeFlowConfig(signals=TradeSignalConfig(framing_profile_source='trades')))
    add=Mock(wraps=p.profile_framing.add_profile); analyze=Mock(wraps=p.profile_framing.analyze)
    monkeypatch.setattr(p.profile_framing,'add_profile',add); monkeypatch.setattr(p.profile_framing,'analyze',analyze)
    monkeypatch.setattr(TradeAccumulator,'developing_profiles',Mock(side_effect=AssertionError('developing read')))
    for b,u in pairs:
        before=analyze.call_count
        monkeypatch.setattr(p.absorption,'check_bar',Mock(side_effect=lambda *a,**kw: (None if analyze.call_count==before+bool(u.finalised_days) else pytest.fail('bias late'))))
        result=p.process_bar(b,u)
        assert analyze.call_count==before+bool(u.finalised_days)
        if u.finalised_days:
            assert result.closed_profile==result.trade_profiles[-1] and result.closed_profile.source=='trades'
            assert result.bias is p.profile_framing.current_bias
        else: assert result.closed_profile is None
    assert [c.args[0].session_date for c in add.call_args_list]==['2023-01-01','2023-01-02','2023-01-03']
    assert len(pairs[2][1].finalised_days)==2
    # First processed bar can also finalise several days.
    q=InfoBarPipeline(get_btcusd_config(),trade_config=p.trade_config)
    q.process_bar(*pairs[2])
    assert len(q.profile_framing._profile_history)==2
    assert not hasattr(pairs[0][1],'developing_profiles')


@pytest.mark.parametrize('unexpected_final',[False,True])
def test_p2_rollover_invariant_from_second_bar(tmp_path,unexpected_final):
    pairs=paired_updates(tmp_path)
    p=InfoBarPipeline(get_btcusd_config(),trade_config=TradeFlowConfig(signals=TradeSignalConfig(framing_profile_source='trades')))
    p.process_bar(*pairs[0])
    b,u=pairs[1] if unexpected_final else pairs[2]
    u=replace(u,finalised_days=pairs[0][1].finalised_days if unexpected_final else [])
    with pytest.raises(DataIntegrityError): p.process_bar(b,u)


def test_flags_off_6000_bar_complete_parity(monkeypatch):
    monkeypatch.setattr(TradeView,'update',lambda *a:(object(),object(),[object()]))
    a,b=InfoBarPipeline(get_btcusd_config()),InfoBarPipeline(get_btcusd_config())
    d0,d1=ReplayDigest(),ReplayDigest()
    for bar_ in synth_bars(6000,seed=3):
        x,y=a.process_bar(bar_),b.process_bar(bar_,object())
        d0.update(x); d1.update(y)
        y.trade_footprint=y.trade_stats=None; y.trade_profiles=[]
        assert x==y
    assert d0.hexdigest()==d1.hexdigest()


def test_raw_digest_includes_details_and_ranges():
    b=bar(); signal=a2(b,[(10010.,100,0)])
    result=InfoBarPipeline(get_btcusd_config()).process_bar(b); result.signals=[signal]
    digest=RawSignalDigest(ranges=[(1,1)],detector=SignalType.ABSORPTION); digest.update(result)
    other=RawSignalDigest(detector=SignalType.ABSORPTION); other.update(result)
    assert digest.range_hexdigest(1,1)==other.hexdigest()
    signal.details=dict(reversed(list(signal.details.items())))
    reordered=RawSignalDigest(detector=SignalType.ABSORPTION); reordered.update(result)
    assert reordered.hexdigest()==other.hexdigest()
    signal.details['delta_source']='tick_rule'
    changed=RawSignalDigest(detector=SignalType.ABSORPTION); changed.update(result)
    assert changed.hexdigest()!=other.hexdigest()


@pytest.mark.parametrize('value',['bad','exhaustion_bonus,exhaustion_required','absorption_levels,absorption_levels'])
def test_cli_invalid_modes(value):
    from orderflow_system.main import parse_args
    from scripts.replay_trades import main
    for cli,args in ((parse_args,[]),(main,['--paired-dataset','unused'])):
        with pytest.raises(SystemExit) as exc: cli([*args,'--trade-signals',value])
        assert exc.value.code==2


@pytest.mark.parametrize('value',['absorption_levels','exhaustion_bonus','exhaustion_required','framing_trades'])
def test_cli_trade_required_and_certificate_refused(value):
    from orderflow_system.main import OrderflowSystem, parse_args
    from orderflow_system.config.settings import parse_trade_signals
    from scripts.replay_trades import main,run
    with pytest.raises(SystemExit) as exc: parse_args(['--trade-signals',value])
    assert exc.value.code==2
    with pytest.raises(SystemExit) as exc:
        main(['--paired-dataset','unused','--certificate','missing','--trade-signals',value])
    assert exc.value.code==2
    cfg=TradeFlowConfig(signals=parse_trade_signals(value))
    with pytest.raises(ValueError,match='certificate'): run('unused',config=cfg,certificate='missing')
    with pytest.raises(ValueError,match='trades'): OrderflowSystem(trade_config=cfg)
    args=parse_args(['--trades','--paired-dataset','v0','--no-dashboard','--trade-signals',value])
    assert args.trade_signals==cfg.signals


@pytest.mark.parametrize('start,end',[('2019-09-09',None),(None,'2024-01-02'),('2024-01-01',None),
                                       (None,'2024-01-01T00:00:00-01:00')])
def test_development_date_guard_before_open(start,end,monkeypatch):
    from scripts import replay_trades as script
    monkeypatch.setattr(script,'open_registered',Mock(side_effect=AssertionError('dataset opened')))
    with pytest.raises(ValueError,match='development'):
        script.run('unused',start=start,end=end,config=TradeFlowConfig(signals=TradeSignalConfig(absorption_levels=True)))


def test_signal_range_windows_on_replay_path(tmp_path,monkeypatch):
    from scripts import replay_trades as script
    from orderflow_system.data.paired_dataset import PairedDataset
    trades=make_trades(days=3,per_day=60)
    directory,root=write_paired(tmp_path,trades,{'fixed_dollar_imbalance':list(range(3,178,3))})
    ds=PairedDataset.open(directory,root)
    monkeypatch.setattr(script,'open_registered',lambda *a:ds)
    cfg=TradeFlowConfig(footprint_bucket=.05,signals=TradeSignalConfig(absorption_levels=True,exhaustion_contrarian='bonus',framing_profile_source='trades'))
    report=script.run(ds.dataset_id,config=cfg,windows=2)
    assert report['certificate_match']=='not_applicable: trade signals enabled'
    assert report['trade_signals']==asdict(cfg.signals)
    for window in report['windows']:
        assert window['equal']
        assert window['lanes']['fixed_dollar_imbalance']['raw_signal_equal']=={'absorption':True,'exhaustion':True}
    # Verify the actual authenticated accumulator width agrees with detector arguments.
    from orderflow_system.data.trade_stream import TradeReplay
    p=InfoBarPipeline(get_btcusd_config(),trade_config=cfg)
    spy=Mock(wraps=p.absorption.check_bar); monkeypatch.setattr(p.absorption,'check_bar',spy)
    replay=TradeReplay(ds,[(open_bars(directory,'BTCUSDT','fixed_dollar_imbalance',version=STAMP),p)],temp_dir=tmp_path/'spill')
    list(replay)
    assert {c.kwargs['trade_bucket_ticks'] for c in spy.call_args_list}=={replay.lanes['fixed_dollar_imbalance'].accumulator.fp_bucket_ticks}=={5}


def test_e2_window_warmup_exclusion_is_needed_and_sufficient():
    # A fresh window cannot reproduce an exhaustion on its first bar.
    bars=[bar(seq=i+1,timestamp_ms=(i+1)*1000,high=10006.+i,low=9999.+i,close=10005.+i,
              volume=100.-10*i,buy_volume=70.-7*i,sell_volume=30.-3*i) for i in range(9)]
    cfg=replace(get_btcusd_config(),exhaustion=ExhaustionConfig(min_bars_declining=3,volume_decline_pct=.25))
    trade_cfg=TradeFlowConfig(signals=TradeSignalConfig(exhaustion_contrarian='bonus'))
    def run(bars):
        p=InfoBarPipeline(cfg,trade_config=trade_cfg)
        # This test isolates history dependence from the separately tested TradeView conversion.
        p.trade_view=Mock(update=lambda *a:(fp([(10010.,0,100)]),None,[]))
        return [p.process_bar(b,SimpleNamespace(fp_bucket_ticks=100)) for b in bars]
    full=run(bars); window=run(bars[4:])
    assert any(s.signal_type==SignalType.EXHAUSTION for s in full[4].signals)
    assert not any(s.signal_type==SignalType.EXHAUSTION for s in window[0].signals)
    a,b=RawSignalDigest(detector=SignalType.EXHAUSTION),RawSignalDigest(detector=SignalType.EXHAUSTION)
    for r in full[4:]: a.update(r)
    for r in window: b.update(r)
    assert a.hexdigest()!=b.hexdigest()
    a,b=RawSignalDigest(detector=SignalType.EXHAUSTION),RawSignalDigest(detector=SignalType.EXHAUSTION,skip_bars=4)
    for r in full[8:]: a.update(r)
    for r in window: b.update(r)
    assert a.hexdigest()==b.hexdigest()


@pytest.mark.parametrize('offset,centre',[(49,10000),(50,10000),(51,10001),(149,10001),(150,10002),(151,10002)])
def test_half_even_bucket_bounds_are_conservative(offset,centre):
    from orderflow_system.analytics.footprint import clipped_trade_bounds
    from orderflow_system.analytics.trade_accumulator import bucket_index
    price=1000000+offset
    c=int(bucket_index(price,100))*100
    assert c==centre*100
    b=bar(low=9999.,high=10003.,close=10001.)
    lo,hi=clipped_trade_bounds(b,c/100,100)
    assert lo<=price/100<=hi


@pytest.mark.parametrize('mirrored',[False,True])
def test_a2_whole_interval_zone_membership(mirrored):
    b,levels=bar(low=10008.,high=10010.49,close=10008.),[(10010.,100,0)]
    if mirrored: b,levels=mirror(b,levels)
    cfg=TradeSignalConfig(absorption_levels=True,min_reject_bps=1.)
    assert a2(b,levels,cfg) is None
    # Enlarging only the zone proves that the zone boundary is the failing gate.
    assert a2(b,levels,replace(cfg,level_zone_frac=1.)) is not None


def test_flags_off_certificate_still_enforces_legacy_digest(tmp_path,monkeypatch):
    import json
    from scripts import replay_trades as script
    from tests.test_trade_replay import setup
    ds,_=setup(tmp_path)
    monkeypatch.setattr(script,'open_registered',lambda *a:ds)
    report=script.run(ds.dataset_id)
    certificate={'v1':{'dataset_id':ds.dataset_id,'root_hash':ds.root_hash},
                 'certification':'dataset-pipeline','pipeline_equivalent_default':True,'types':{}}
    for bt,lane in report['lanes'].items():
        certificate['types'][bt]={'L4':{'passed':True,'configs':[dict(config='default',params=None,
            window=[None,None],equal=True,v1=lane['replay_digest'],n_bars=lane['bars'])]}}
    path=tmp_path/'certificate.json'
    path.write_text(json.dumps(certificate))
    assert script.run(ds.dataset_id,certificate=path)['certificate_match'] is True
    for row in certificate['types'].values(): row['L4']['configs'][0]['v1']='wrong'
    path.write_text(json.dumps(certificate))
    with pytest.raises(DataIntegrityError,match='ReplayDigest'):
        script.run(ds.dataset_id,certificate=path)


@pytest.mark.parametrize('mirrored',[False,True])
@pytest.mark.parametrize('scale',[1e-3,1,1e3])
def test_e2_minimum_level_share(mirrored,scale):
    b=bar()
    levels=[(10010.,0,1*scale),(10005.,99*scale,0)]
    if mirrored: b,levels=mirror(b,levels)
    detector=ExhaustionDetector(ExhaustionConfig(),signals=TradeSignalConfig(exhaustion_contrarian='bonus'))
    side=Side.BUY if mirrored else Side.SELL
    assert detector._trade_contrarian(b,fp(levels),100,side)==0
    detector.signals=replace(detector.signals,contrarian_min_level_share=.005)
    assert detector._trade_contrarian(b,fp(levels),100,side)==1


@pytest.mark.parametrize('value', [0, 1, 'false', None, np.bool_(True)])
def test_absorption_flag_requires_bool(value):
    with pytest.raises(ValueError):
        TradeSignalConfig(absorption_levels=value)


@pytest.mark.parametrize('name,valid', [('exhaustion_contrarian', 'off'), ('framing_profile_source', 'bar_vwap')])
@pytest.mark.parametrize('kind', ['string_like', 'none', 'bool', 'list'])
def test_trade_signal_enums_require_strings(name, valid, kind):
    value = {'string_like': UserString(valid), 'none': None, 'bool': False, 'list': []}[kind]
    with pytest.raises(ValueError):
        TradeSignalConfig(**{name: value})


@pytest.mark.parametrize('name', ['level_zone_frac', 'min_level_share', 'min_side_ratio', 'min_reject_bps',
                                'max_through_bps', 'contrarian_ratio', 'contrarian_zone_frac', 'contrarian_min_level_share'])
@pytest.mark.parametrize('value', [True, '0.5', None, []])
def test_trade_signal_numeric_types_raise_value_error(name, value):
    with pytest.raises(ValueError):
        TradeSignalConfig(**{name: value})


def test_trade_flow_original_positional_order(tmp_path):
    args = ('paired-root', 1., 10., 3, '128MB', str(tmp_path/'spill'), str(tmp_path/'reports'))
    original_fields = ('paired_root', 'footprint_bucket', 'profile_bucket', 'duckdb_threads',
                       'duckdb_memory_limit', 'duckdb_temp_dir', 'report_dir')
    cfg = TradeFlowConfig(*args)
    assert tuple(getattr(cfg, name) for name in original_fields) == args
    assert not cfg.signals.any_enabled
    signals = TradeSignalConfig(absorption_levels=True)
    assert TradeFlowConfig(*args, signals).signals is signals
    assert tuple(f.name for f in fields(TradeFlowConfig)) == (*original_fields, 'signals')


def bucket_update(width):
    from tests.test_trade_accumulator import rows
    acc = TradeAccumulator(width, 1000, 0)
    acc.add(1, rows([1000000, 1001000, 1000500], times=[0, 0, 1], lots=[25000, 25000, 50000]))
    return acc.close_bar(3, 1)


@pytest.mark.parametrize('width', [100, 1000])
def test_update_carries_bucket_width_as_optional_last_field(width):
    from orderflow_system.analytics.trade_accumulator import TradeBarUpdate
    update = bucket_update(width)
    assert update.fp_bucket_ticks == width
    assert fields(TradeBarUpdate)[-1].name == 'fp_bucket_ticks'
    legacy = TradeBarUpdate(update.stats, update.footprint_levels, update.finalised_days)
    assert legacy.fp_bucket_ticks is None


@pytest.mark.parametrize('mode', ['a2', 'bonus', 'required'])
@pytest.mark.parametrize('width', [None, 1000])
def test_pipeline_rejects_bucket_width_before_state_change(mode, width):
    import pickle
    from orderflow_system.analytics.trade_accumulator import TradeBarUpdate
    cfg = TradeFlowConfig(signals=TradeSignalConfig(absorption_levels=mode == 'a2',
        exhaustion_contrarian='off' if mode == 'a2' else mode))
    update = bucket_update(1000)
    if width is None:
        update = TradeBarUpdate(update.stats, update.footprint_levels, update.finalised_days)
    for started in (False, True):
        pipe = InfoBarPipeline(get_btcusd_config(), trade_config=cfg)
        if started:
            pipe.process_bar(bar(), bucket_update(100))
        before = pickle.dumps(pipe)
        with pytest.raises(DataIntegrityError, match='bucket'):
            pipe.process_bar(bar(timestamp_ms=2, seq=6), update)
        assert pickle.dumps(pipe) == before


@pytest.mark.parametrize('source', ['bar_vwap', 'trades'])
@pytest.mark.parametrize('width', [None, 1000])
def test_pipeline_without_level_flags_ignores_width(source, width):
    cfg = TradeFlowConfig(signals=TradeSignalConfig(framing_profile_source=source))
    update = bucket_update(1000)
    reference = InfoBarPipeline(get_btcusd_config(), trade_config=cfg).process_bar(
        bar(), replace(update, fp_bucket_ticks=100))
    actual = InfoBarPipeline(get_btcusd_config(), trade_config=cfg).process_bar(
        bar(), replace(update, fp_bucket_ticks=width))
    assert actual == reference


@pytest.mark.parametrize('bucket', [1., 10.])
def test_trade_replay_update_width_matches_geometry(tmp_path, monkeypatch, bucket):
    from orderflow_system.data.trade_stream import TradeReplay
    from tests.test_trade_replay import setup, TYPES
    ds, _ = setup(tmp_path, ends={TYPES[0]: [4, 99, 101, 270]})
    cfg = TradeFlowConfig(footprint_bucket=bucket, signals=TradeSignalConfig(
        absorption_levels=True, exhaustion_contrarian='bonus'))
    pipe = InfoBarPipeline(get_btcusd_config(), trade_config=cfg)
    process = Mock(wraps=pipe.process_bar)
    monkeypatch.setattr(pipe, 'process_bar', process)
    reader = open_bars(ds.bars_root, 'BTCUSDT', TYPES[0], version=STAMP)
    run = TradeReplay(ds, [(reader, pipe)], temp_dir=tmp_path/'spill')
    list(run)
    assert len(process.call_args_list) == 4
    assert {call.args[1].fp_bucket_ticks for call in process.call_args_list} == {round(bucket*100)}


@pytest.mark.parametrize('b', [bar(), bar(open=10010., buy_volume=90., sell_volume=10.), bar(volume=0.)])
def test_direct_a2_requires_trade_footprint(b):
    detector = AbsorptionDetector(AbsorptionConfig(), signals=TradeSignalConfig(absorption_levels=True))
    with pytest.raises(ValueError, match='trade.*footprint'):
        detector.check_bar(b, fp(), DeltaEngine().compute_from_bar(b), b.close)


@pytest.mark.parametrize('enabled', [False, True])
def test_a2_replaces_m2_only_when_enabled(monkeypatch, enabled):
    b = bar()
    detector = AbsorptionDetector(AbsorptionConfig(), signals=TradeSignalConfig(absorption_levels=enabled))
    legacy_result = object()
    m2 = Mock(return_value=legacy_result)
    monkeypatch.setattr(detector, '_check_level_absorption', m2)
    args = {'trade_footprint': fp(), 'trade_bucket_ticks': 100} if enabled else {}
    result = detector.check_bar(b, fp(), DeltaEngine().compute_from_bar(b), b.close, **args)
    if enabled:
        assert result is None
        m2.assert_not_called()
    else:
        assert result is legacy_result
        m2.assert_called_once()


@pytest.mark.parametrize('name', ['level_zone_frac', 'min_level_share', 'min_side_ratio', 'min_reject_bps',
                                'max_through_bps', 'contrarian_ratio', 'contrarian_zone_frac', 'contrarian_min_level_share'])
@pytest.mark.parametrize('number_type', [np.float32, np.float64, np.int64, int])
@pytest.mark.parametrize('use_replace', [False, True])
def test_real_thresholds_are_normalized_to_python_floats(name, number_type, use_replace):
    value = number_type(2 if name == 'contrarian_ratio' else 1)
    kwargs = {name: value}
    cfg = replace(TradeSignalConfig(), **kwargs) if use_replace else TradeSignalConfig(**kwargs)
    assert type(getattr(cfg, name)) is float
    assert getattr(cfg, name) == float(value)


@pytest.mark.parametrize('name', ['level_zone_frac', 'min_level_share', 'min_side_ratio', 'min_reject_bps',
                                'max_through_bps', 'contrarian_ratio', 'contrarian_zone_frac', 'contrarian_min_level_share'])
@pytest.mark.parametrize('value', [np.bool_(True), True, np.float32('nan'), 1+0j, '1', None])
def test_nonreal_or_nonfinite_thresholds_raise_value_error(name, value):
    with pytest.raises(ValueError):
        TradeSignalConfig(**{name: value})

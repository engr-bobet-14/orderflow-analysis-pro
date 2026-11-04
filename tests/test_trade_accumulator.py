"""Exact trade units, independent price buckets and close-only day visibility."""
from fractions import Fraction

import numpy as np
import pytest

from orderflow_system.analytics.trade_accumulator import TradeAccumulator, bucket_index
from orderflow_system.config.settings import TradeFlowConfig
from orderflow_system.data.infobar_feed import DataIntegrityError
from orderflow_system.data.paired_dataset import ROW_DTYPE

DAY = 86_400_000


def rows(prices, times=None, lots=None, maker=None):
    a = np.zeros(len(prices), dtype=ROW_DTYPE)
    a['price_ticks'] = prices
    a['qty_lots'] = lots if lots is not None else 1000
    a['time_ms'] = times if times is not None else np.arange(len(a))
    a['trade_id'] = np.arange(len(a))
    a['is_buyer_maker'] = maker if maker is not None else 0
    return a


@pytest.mark.parametrize('width', [1, 10, 50, 100, 500, 1000, 2500])
def test_bucket_fraction_oracle(width):
    prices = np.array([q * width + r for q in range(-20, 21) for r in range(width)], dtype=np.int64)
    assert np.array_equal(bucket_index(prices, width), [round(Fraction(int(p), width)) for p in prices])
    assert bucket_index(15, 10) == 2


@pytest.mark.parametrize('width', [0, -1, .001, .015, float('nan'), float('inf')])
def test_refuse_invalid_width(width):
    with pytest.raises(ValueError):
        TradeFlowConfig(footprint_bucket=width)


def test_original_price_used_for_both_histograms():
    a = TradeAccumulator(100, 1000, cursor=0)
    a.add(1, rows([1000510, 1000490], lots=[100000, 1000]))
    u = a.close_bar(2, 1)
    assert tuple(map(list, u.footprint_levels)) == ([1000500], [101000], [0])
    assert tuple(map(list, a.developing_profiles()[0][1:])) == ([1000000, 1001000], [1000, 100000])
    assert u.stats.lots == u.stats.buy_lots + u.stats.sell_lots == 101000


def test_multiday_close_only_and_no_eof_rollover():
    a = TradeAccumulator(100, 1000, cursor=0)
    a.add(1, rows([10000] * 3, times=[23*3600000, DAY+1, 2*DAY+3600000]))
    assert len(a.developing_profiles()) == 3
    u = a.close_bar(3, 2*DAY+3600000)
    assert [d for d, _, _ in u.finalised_days] == [0, 1]
    assert [d for d, _, _ in a.developing_profiles()] == [2]
    assert a.developing_profiles()[0][2].sum() == 1000


@pytest.mark.parametrize('size', [1, 7, 997, 1000000])
def test_slice_invariance(size):
    data = rows(np.arange(3000) + 2000000, times=np.arange(3000)*100000, maker=np.arange(3000)%2)
    a = TradeAccumulator(100, 1000, cursor=0)
    for i in range(0, len(data), size):
        a.add(i+1, data[i:i+size])
    u = a.close_bar(len(data), int(data['time_ms'][-1]))
    b = TradeAccumulator(100, 1000, cursor=0)
    b.add(1, data)
    v = b.close_bar(len(data), int(data['time_ms'][-1]))
    assert u.stats == v.stats
    for x, y in zip(u.footprint_levels, v.footprint_levels):
        assert np.array_equal(x, y)
    for x, y in zip(u.finalised_days, v.finalised_days):
        assert x[0] == y[0]
        assert np.array_equal(x[1], y[1]) and np.array_equal(x[2], y[2])


@pytest.mark.parametrize('method', ['add', 'warm'])
@pytest.mark.parametrize('start', [0, 2])
def test_continuity(method, start):
    with pytest.raises(DataIntegrityError, match='ordinal'):
        getattr(TradeAccumulator(100, 1000, 0), method)(start, rows([10000]))


def test_close_and_domain_guards():
    a = TradeAccumulator(100, 1000, 0)
    with pytest.raises(DataIntegrityError):
        a.close_bar(0, 0)
    a.warm_end = 1
    a.pending_end = 2
    with pytest.raises(DataIntegrityError):
        a.warm(1, rows([10000, 10000]))
    with pytest.raises(DataIntegrityError):
        a.add(1, rows([10000]*3))
    a.add(1, rows([10000]))
    with pytest.raises(DataIntegrityError):
        a.close_bar(2, 0)


def test_dollar_overflow_fails():
    a = TradeAccumulator(100, 1000, 0)
    with pytest.raises(DataIntegrityError, match='overflow'):
        a.add(1, rows([2**62], lots=[4]))


def test_rounded_bucket_overflow_refuses_instead_of_wrapping():
    a = TradeAccumulator(100, 1000, 0)
    with pytest.raises(DataIntegrityError, match='overflow'):
        a.add(1, rows([np.iinfo(np.int64).max], lots=[1]))


def test_quantities_sum_exactly_beyond_float_integer_precision():
    # Weighted bincount would silently lose these low bits.
    q = 2**53+1
    a = TradeAccumulator(1, 1, 0)
    a.add(1, rows([1, 1], lots=[q, 1]))
    u = a.close_bar(2, 1)
    assert u.stats.lots == q+1
    assert int(u.footprint_levels[1][0]) == q+1
    assert int(a.developing_profiles()[0][2][0]) == q+1


@pytest.mark.slow
def test_synthetic_five_million_trade_workload(record_property):
    from time import perf_counter
    n = 5_000_000
    a = TradeAccumulator(100, 1000, 0)
    data = rows(2000000+np.arange(n)%10000, times=np.arange(n), lots=np.arange(n)%1000+1,
                maker=np.arange(n)%2)
    started = perf_counter()
    for i in range(0, n, 1_000_000):
        a.add(i+1, data[i:i+1_000_000])
    u = a.close_bar(n, n-1)
    seconds = perf_counter()-started
    record_property('aggregation_trades_per_second', n/seconds)
    print(f'5M accumulator: {seconds:.3f}s, {n/seconds:,.0f} trades/s')
    assert u.stats.n == n
    assert u.stats.lots == 2502500000
    assert sum(int(x) for x in u.footprint_levels[1])+sum(int(x) for x in u.footprint_levels[2]) == u.stats.lots


def test_bucket_width_tolerance_uses_rounded_tick_minimum():
    assert TradeFlowConfig(footprint_bucket=.01-1e-13).footprint_bucket > 0


def test_relative_spill_path_resolves_from_repo_root(tmp_path, monkeypatch):
    from pathlib import Path
    from orderflow_system.config import settings
    monkeypatch.chdir(tmp_path)
    cfg = TradeFlowConfig()
    expected = Path(settings.__file__).resolve().parents[2]/'output'/'duckdb_tmp'
    assert Path(cfg.duckdb_temp_dir) == expected


def test_spill_config_refuses_volumes():
    with pytest.raises(ValueError, match='local|Volumes'):
        TradeFlowConfig(duckdb_temp_dir='/Volumes/not-local/trade-spill')


def test_exact_value_area_preserves_poc_and_expansion_ties():
    from orderflow_system.analytics.trade_accumulator import exact_value_area
    assert exact_value_area(np.array([100, 100, 100], dtype=np.int64), .5) == (0, 0, 1)
    assert exact_value_area(np.array([100, 300, 100], dtype=np.int64), .8) == (1, 1, 2)
    # Neither totals nor fraction multiplication may overflow int64.
    assert exact_value_area(np.array([2**62, 2**62, 2**62], dtype=np.int64), 1.) == (0, 0, 2)


def test_exact_value_area_matches_legacy_with_exact_float_targets():
    from orderflow_system.analytics.trade_accumulator import exact_value_area
    from orderflow_system.analytics.volume_profile import VolumeProfileEngine
    from orderflow_system.config.settings import VolumeProfileConfig
    rng = np.random.default_rng(20261103)
    for pct in (.25, .5, .75, 1.):
        engine = VolumeProfileEngine(VolumeProfileConfig(value_area_pct=pct))
        for _ in range(100):
            lots = rng.integers(1, 10000, size=int(rng.integers(1, 40)), dtype=np.int64)
            poc, lo, hi = exact_value_area(lots, pct)
            assert poc == int(np.argmax(lots))
            prices = list(range(len(lots)))
            vah, val = engine._compute_value_area(prices, lots.astype(float), poc, float(lots.sum()))
            assert (hi, lo) == (vah, val)


@pytest.mark.parametrize('pct', [0., -.1, 1.1, float('nan'), float('inf')])
def test_exact_value_area_refuses_invalid_fraction(pct):
    from orderflow_system.analytics.trade_accumulator import exact_value_area
    with pytest.raises(ValueError):
        exact_value_area(np.array([100], dtype=np.int64), pct)


@pytest.mark.parametrize('location', ['default', 'relative', 'absolute'])
def test_report_dir_resolves_from_repo_root(tmp_path, monkeypatch, location):
    from pathlib import Path
    from orderflow_system.config import settings
    repo = Path(settings.__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    if location == 'default':
        config = TradeFlowConfig()
        expected = repo/'output'/'trade_replay'
    elif location == 'relative':
        config = TradeFlowConfig(report_dir='output/custom_reports')
        expected = repo/'output'/'custom_reports'
    else:
        config = TradeFlowConfig(report_dir=str(tmp_path/'reports'))
        expected = tmp_path/'reports'
    assert Path(config.report_dir) == expected


def test_report_dir_refuses_volumes():
    with pytest.raises(ValueError, match='local|Volumes'):
        TradeFlowConfig(report_dir='/Volumes/not-local/trade-reports')

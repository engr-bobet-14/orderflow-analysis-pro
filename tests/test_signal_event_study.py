"""Search-only order-flow signal event study: synthetic tests only (no real data, no holdout)."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import signal_event_study as es

MIN = 60_000
DAY = 86_400_000


def series(close, *, step=MIN, start=0, gap=None, high=None, low=None, open_=None, ts=None):
    close = np.asarray(close, float)
    n = len(close)
    return es.PriceSeries(
        ts=np.asarray(ts, dtype=np.int64) if ts is not None else np.arange(n, dtype=np.int64) * step + start,
        open=np.asarray(open_ if open_ is not None else close, float),
        high=np.asarray(high if high is not None else close, float),
        low=np.asarray(low if low is not None else close, float),
        close=close,
        gap=np.zeros(n, bool) if gap is None else np.asarray(gap, bool))


# --- pre-registration (amended 2026-11-10 after the pre-run review) ----------------------------------------------
def test_pre_registered_design_constants():
    assert es.COST_BPS == 12.0
    assert es.HORIZONS_MIN == (5, 15, 60, 240, 1440)
    assert es.FAMILIES == ('absorption', 'initiative_auction', 'exhaustion', 'delta_divergence')
    assert es.ARMS == ('fixed_tick_imbalance', 'fixed_volume_imbalance')
    assert (es.BLOCK_DAYS, es.RESAMPLES, es.ALPHA) == (7, 20_000, 0.05)
    assert (es.MIN_EVENTS, es.MIN_BLOCKS) == (30, 26)


def test_design_record_documents_baseline_rules_deviations_and_limits():
    design = es.design_record()
    text = repr(design).lower()
    assert 'same-day drift' in design['baseline'].lower() and 'not a matched random-entry' in design['baseline'].lower()
    for key in ('entry', 'exit', 'dedup', 'exclusions', 'effective_horizon', 'test', 'decision', 'deviations',
                'interpretation', 'memory'):
        assert key in design
    assert 'raw detector signals' in design['interpretation'].lower()
    assert 'no exit/stop design can be profitable' not in text
    assert any('random-entry' in d for d in design['deviations'])
    assert any('mfe' in d.lower() for d in design['deviations'])


def test_design_record_discloses_skew_sensitivity_and_mean_test_claims_no_exact_validity():
    # m1: the block-t guard is anti-conservative in the extreme tail under negative skew; disclose, do not claim.
    limits = ' '.join(es.design_record()['limitations']).lower()
    assert 'negative skew' in limits and 'anti-conservative' in limits
    assert 'valid on its own terms' not in es.mean_test.__doc__


# --- forward returns: next-bar entry, no look-ahead, search-prefix boundary, gaps, ties -----------------------
def test_entry_is_next_bar_open_and_exit_is_last_close_within_horizon():
    s = series([100, 100, 100, 100], open_=[100, 101, 100, 100], high=[100, 101, 104, 100])
    s.close[:] = [100, 102, 103, 99]
    exit_idx, valid = es.forward_window(s, 2 * MIN)
    ret = es.long_returns_bps(s, exit_idx, valid)
    assert valid[0] and exit_idx[0] == 2
    assert ret[0] == pytest.approx((103 - 101) / 101 * 1e4)


def test_no_lookahead_changing_prices_after_the_horizon_or_the_decision_close_changes_nothing():
    base = series([100, 101, 102, 103, 104, 105])
    exit_idx, valid = es.forward_window(base, 2 * MIN)
    ref = es.long_returns_bps(base, exit_idx, valid)
    later = series([100, 101, 102, 103, 999, 999])
    exit2, valid2 = es.forward_window(later, 2 * MIN)
    assert es.long_returns_bps(later, exit2, valid2)[0] == ref[0]
    moved = series([555, 101, 102, 103, 104, 105], open_=[100, 101, 102, 103, 104, 105])
    exit3, valid3 = es.forward_window(moved, 2 * MIN)
    assert es.long_returns_bps(moved, exit3, valid3)[0] == ref[0]


def test_tied_close_times_exit_at_the_last_tied_bar_and_dedup_treats_ties_as_overlapping():
    s = series([100, 101, 102, 103, 104], ts=[0, MIN, MIN, 2 * MIN, 3 * MIN])
    exit_idx, valid = es.forward_window(s, MIN)
    assert valid[0] and exit_idx[0] == 2               # both bars closing at 1 min are inside the window
    assert list(es.dedup(np.array([1, 2]), s.ts, MIN, exit_idx)) == [1]


def test_dedup_drops_an_event_whose_entry_falls_inside_the_previous_window_on_tied_closes():
    # m3: bar 1 closes exactly at event 0's deadline, but event 0 exits at the tied bar 2, so event 1's entry
    # (open of bar 2) lies inside event 0's window. Bar 2 itself is the exit bar: its entry (bar 3) is outside.
    s = series([100, 101, 102, 103, 104], ts=[0, MIN, MIN, 2 * MIN, 3 * MIN])
    exit_idx, _ = es.forward_window(s, MIN)
    assert list(es.dedup(np.array([0, 1]), s.ts, MIN, exit_idx)) == [0]
    assert list(es.dedup(np.array([0, 2]), s.ts, MIN, exit_idx)) == [0, 2]


def test_events_whose_horizon_passes_the_last_search_bar_are_dropped():
    s = series([100] * 10)
    _, valid = es.forward_window(s, 3 * MIN)
    assert valid[:7].all() and not valid[7:].any()    # last decision bar with a fully observed window is 6


def test_window_spanning_a_data_gap_bar_is_dropped():
    gap = np.zeros(10, bool); gap[4] = True
    s = series([100] * 10, gap=gap)
    _, valid = es.forward_window(s, 2 * MIN)
    assert not valid[2] and not valid[3]
    assert valid[4] and valid[5]


def test_slow_bars_with_no_bar_closing_inside_the_horizon_are_dropped():
    s = series([100] * 5, step=10 * MIN)
    _, valid = es.forward_window(s, 5 * MIN)
    assert not valid.any()


# --- overlapping events -----------------------------------------------------------------------------------------
def test_dedup_keeps_only_non_overlapping_windows():
    s = series([100] * 100)
    exit_idx, _ = es.forward_window(s, 5 * MIN)
    assert list(es.dedup(np.array([0, 1, 4, 5, 9, 10]), s.ts, 5 * MIN, exit_idx)) == [0, 5, 10]


# --- same-day drift baseline -------------------------------------------------------------------------------------
def test_same_day_drift_baseline_is_the_mean_forward_return_of_valid_non_gap_bars_that_day():
    close = np.r_[np.linspace(100, 110, 50), np.linspace(110, 100, 50)]
    close[11] = 150.0                                  # gap decision bar 10 gets an outlying forward return
    gap = np.zeros(100, bool); gap[10] = True
    s = series(close, step=DAY // 50, open_=np.r_[close[0], close[:-1]], gap=gap)
    exit_idx, valid = es.forward_window(s, DAY // 50)
    ret = es.long_returns_bps(s, exit_idx, valid)
    base = es.day_baseline(s, ret, valid)
    days = s.ts // DAY
    for d in (0, 1):
        mask = (days == d) & valid & ~gap
        assert np.allclose(base[days == d], ret[mask].mean())
    assert valid[10] and not np.isclose(base[0], ret[(days == 0) & valid].mean())  # gap decision bar excluded


# --- excursions (oracle bound, labelled) ------------------------------------------------------------------------
def test_excursions_use_bars_after_entry_through_exit():
    s = series([100] * 5, high=[150, 102, 105, 101, 100], low=[50, 99, 97, 100, 100])
    mfe, mae = es.excursions(s, 0, 3, 1)
    assert mfe == pytest.approx(500.0) and mae == pytest.approx(300.0)
    mfe_s, mae_s = es.excursions(s, 0, 3, -1)
    assert mfe_s == pytest.approx(300.0) and mae_s == pytest.approx(500.0)


# --- statistics ------------------------------------------------------------------------------------------------------
def test_block_bootstrap_is_deterministic_and_one_sided():
    days = np.repeat(np.arange(70), 5)
    high = es.block_bootstrap(np.full(350, 40.0) + np.tile([-1., 1.], 175), days, 12.0)
    low = es.block_bootstrap(np.full(350, 2.0) + np.tile([-1., 1.], 175), days, 12.0)
    assert high['p'] == pytest.approx(1 / (es.RESAMPLES + 1)) and low['p'] == 1.0
    assert es.block_bootstrap(np.arange(350.), days, 12.0) == es.block_bootstrap(np.arange(350.), days, 12.0)
    assert high['mc_se'] == pytest.approx(np.sqrt(high['p'] * (1 - high['p']) / es.RESAMPLES))


def test_block_bootstrap_resamples_whole_seven_day_blocks():
    days = np.arange(70)
    values = np.where(days // 7 % 2 == 0, 100.0, -100.0)
    means = np.asarray(es.block_bootstrap(values, days, 0.0)['means']) / 20
    assert np.allclose(means, np.round(means))


def test_block_t_p_uses_cluster_robust_se_with_blocks_minus_one_df():
    from scipy import stats
    rng = np.random.default_rng(1)
    days = np.repeat(np.arange(0, 7 * 40, 7), 3)                    # 40 blocks, 3 events each
    values = 15 + rng.normal(0, 5, len(days))
    block = days // 7 - days.min() // 7
    block_sums = np.bincount(block, weights=values)
    counts = np.bincount(block).astype(float)
    mean = values.sum() / len(values)
    resid = block_sums - mean * counts
    se = np.sqrt(40 / 39 * (resid ** 2).sum()) / len(values)
    expected = stats.t.sf((mean - 12) / se, 39)
    assert es.block_t_p(values, days, 12.0) == pytest.approx(expected)
    assert es.block_t_p(np.full(30, 20.0), np.arange(30) * 7, 12.0) == 0.0     # zero variance, above threshold
    assert es.block_t_p(np.full(30, 5.0), np.arange(30) * 7, 12.0) == 1.0


def test_mean_test_uses_the_larger_of_bootstrap_and_block_t():
    rng = np.random.default_rng(2)
    days = np.repeat(np.arange(0, 7 * 30, 7), 4)
    values = 12.5 + rng.standard_t(3, len(days)) * 8
    result = es.mean_test(values, days, 12.0)
    assert result['p'] == max(result['p_bootstrap'], result['p_block_t'])
    assert 'lower_95_unadjusted_bps' in result and result['blocks'] == 30


def test_mean_test_fails_closed_on_non_finite_values():
    # m2: a NaN made p_block_t NaN, max(p_boot, nan) kept p_boot and NaN resample means counted as rejections.
    days = np.repeat(np.arange(0, 7 * 30, 7), 4)
    for bad in (np.nan, np.inf, -np.inf):
        values = np.full(len(days), -5.0)
        values[3] = bad
        with pytest.raises(ValueError, match='non-finite'):
            es.mean_test(values, days, 12.0)


def test_analyse_cell_fails_closed_when_a_kept_return_is_not_finite():
    s = series(np.full(400, 100.0), step=10 * MIN)
    s.close[50] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        es.analyse_cell(s, np.arange(0, 380, 3), 1, 25 * MIN)


def test_holm_adjustment():
    assert es.holm([0.01, 0.04, 0.03, 0.5]) == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert es.holm([1.0, 1.0]) == [1.0, 1.0]


# --- decision rule (I1: one intersection-union p per cell, one Holm family) ---------------------------------------
def cell(p_gross, p_excess, *, n=100, blocks=60):
    return {'n': n, 'blocks': blocks, 'p_gross': p_gross, 'p_excess': p_excess}


def test_decision_uses_the_larger_component_p_per_cell_then_holm():
    assert es.decide([cell(1e-4, 1e-4)] + [cell(0.9, 0.9)] * 79)['decision'] == 'GO'
    assert es.decide([cell(1e-4, 5e-4)] + [cell(0.9, 0.9)] * 79)['decision'] == 'GO'    # 80 x 5e-4 = 0.04
    assert es.decide([cell(1e-4, 0.9)] + [cell(0.9, 0.9)] * 79)['decision'] == 'NO-GO'
    assert es.decide([cell(0.01, 0.01)] + [cell(0.9, 0.9)] * 79)['decision'] == 'NO-GO'


def test_decision_closes_the_separate_holm_loophole():
    # Separate Holm corrections then AND would pass cell 0 (gross 2e-4, excess 0.03) and report GO.
    result = es.decide([cell(1e-4, 0.03), cell(0.03, 1e-4)])
    assert result['decision'] == 'NO-GO'
    assert [c['p_cell'] for c in result['cells']] == [0.03, 0.03]


def test_insufficient_cells_cannot_pass_and_stay_in_the_holm_family():
    result = es.decide([cell(1e-6, 1e-6, n=29), cell(1e-6, 1e-6, blocks=25), cell(1e-4, 1e-4)])
    assert result['decision'] == 'GO' and result['family_size'] == 3
    assert [c['p_cell_holm'] for c in result['cells'][:2]] == [1.0, 1.0]
    assert result['insufficient_cells'] == 2


def test_cell_statistics_counts_and_effective_horizon_on_a_known_edge():
    rng = np.random.default_rng(0)
    n = 3000
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    s = series(close, step=10 * MIN)
    bars = np.arange(0, n - 20, 7)
    for b in bars:
        for arr in (s.close, s.open, s.high, s.low):
            arr[b + 2:] *= 1.005
    raw = np.r_[bars, bars[:5] + 1, [n - 1]]                      # overlapping and unobservable extras
    result = es.analyse_cell(s, raw, 1, 25 * MIN)
    assert result['n'] > 300 and result['mean_bps'] > 40 and result['p_gross'] < 0.01
    assert result['oracle_mfe_bps_p50_p90'][0] > 40
    assert result['raw_events'] == len(np.unique(raw))
    assert result['dropped_unobservable'] == 1 and result['dropped_overlap'] == 5
    assert result['n'] == result['raw_events'] - result['dropped_unobservable'] - result['dropped_overlap']
    assert result['effective_horizon_min_mean'] == pytest.approx(20.0) and result['effective_horizon_min_p10'] == 20.0
    assert 0 < result['p_excess_bootstrap_mc_se'] < 0.01                        # m5
    assert np.isfinite(result['gross_skew']) and np.isfinite(result['excess_skew'])   # m1 disclosure per cell
    short = es.analyse_cell(s, bars, -1, 25 * MIN)
    assert short['mean_bps'] < -40 and short['p_gross'] == 1.0


# --- signal collection, pipeline equivalence and isolation ---------------------------------------------------------
class FakePipeline:
    def __init__(self, emitted):
        self.emitted = emitted
        self.aggregator = SimpleNamespace(price_proximity_pct=None)

    def process_bar(self, bar, update):
        signals, gap = self.emitted.get(bar.seq, ([], False))
        return SimpleNamespace(bar=bar, signals=[] if gap else signals, gap_skipped=gap, aggregated=[],
                               closed_profile=None, delta=SimpleNamespace(cumulative_delta=0.0))


def sig(kind, side):
    return SimpleNamespace(signal_type=SimpleNamespace(value=kind), direction=SimpleNamespace(value=side),
                           price_level=1.0, strength=1.0)


def test_collect_events_maps_families_and_directions_and_skips_gap_bars():
    bars = [SimpleNamespace(seq=i, timestamp_ms=i * MIN, open=1., high=1., low=1., close=1., data_gap=(i == 2))
            for i in range(4)]
    pipe = FakePipeline({0: ([sig('absorption', 'buy'), sig('book_sweep', 'buy')], False),
                         1: ([sig('exhaustion', 'sell'), sig('exhaustion', 'sell')], False),
                         2: ([sig('absorption', 'sell')], True)})
    s, events, _ = es.collect_events(bars, [None] * 4, pipe, digest=False)
    assert events == {('absorption', 1): [0], ('exhaustion', -1): [1]}
    assert list(s.gap) == [False, False, True, False]


def test_collect_events_reproduces_the_run_backtest_replay_digest_on_synthetic_bars():
    from tests.test_pipeline import synth_bars
    from scripts import optimize_orderflow as opt
    from orderflow_system.config.settings import get_btcusd_config
    from orderflow_system.data.models import BarType
    bars = synth_bars(n=1500)
    expected = opt.run_backtest(bars, get_btcusd_config(), 0.002, BarType.DOLLAR_IMBALANCE)['replay_digest']
    pipe = opt.bar_only_pipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE)
    pipe.aggregator.price_proximity_pct = 0.002
    _, events, digest = es.collect_events(bars, [None] * len(bars), pipe)
    assert digest == expected and events


def test_default_pipeline_is_the_s1_defaults_construction():
    from scripts import phase4_studies as p4
    params = p4.full_strategy_params(p4.full_strategy_defaults())
    pipe = es.build_default_pipeline('fixed_tick_imbalance')
    assert pipe.aggregator.price_proximity_pct == params['price_proximity_pct']
    assert pipe.trade_config.signals.absorption_source == 'trade'


def test_run_refuses_non_search_prefix_and_unsafe_or_existing_output(tmp_path):
    with pytest.raises(TypeError):
        es.require_search_prefix(object())
    checkout = tmp_path / 'campaign'
    (checkout / 'output').mkdir(parents=True)
    with pytest.raises(ValueError, match='outside'):
        es.check_output_location(checkout / 'output' / 'study', checkout)
    with pytest.raises(ValueError, match='Market-Data'):
        es.check_output_location(Path('/Volumes/Market-Data/study'), checkout)
    existing = tmp_path / 'done'
    existing.mkdir()
    with pytest.raises(ValueError, match='exists'):
        es.check_output_location(existing, checkout)
    es.check_output_location(tmp_path / 'fresh', checkout)


def test_output_guard_cannot_be_bypassed_by_letter_case(tmp_path):
    # m4: APFS is case-insensitive and resolve() does not canonicalise case.
    checkout = tmp_path / 'Campaign'
    checkout.mkdir()
    with pytest.raises(ValueError, match='outside'):
        es.check_output_location(tmp_path / 'campaign' / 'study', checkout)
    with pytest.raises(ValueError, match='Market-Data'):
        es.check_output_location(Path('/volumes/market-data/study'), checkout)


def test_run_checks_output_before_opening_any_input(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(es, '_sealed_json', lambda path: opened.append(path))
    with pytest.raises(ValueError, match='outside'):
        es.run(tmp_path, tmp_path / 'inside')
    assert opened == []


def test_benchmark_must_belong_to_the_sealed_manifest():
    with pytest.raises(ValueError, match='benchmark'):
        es.check_benchmark_binding({'campaign_sha256': 'a' * 64}, 'b' * 64)
    es.check_benchmark_binding({'campaign_sha256': 'a' * 64}, 'a' * 64)


def test_pipeline_must_reproduce_the_accepted_s1_defaults_digest():
    with pytest.raises(ValueError, match='S1 defaults digest'):
        es.verify_default_digest('a' * 64, 'b' * 64)
    es.verify_default_digest('a' * 64, 'a' * 64)


def test_code_provenance_and_data_boundaries_are_recorded():
    prov = es.code_provenance()
    assert prov['script_sha256'] == hashlib.sha256(Path(es.__file__).read_bytes()).hexdigest()
    assert set(prov) >= {'script_sha256', 'git_head', 'git_dirty', 'test_sha256'}
    s = series([100] * 3, ts=[DAY, DAY + MIN, DAY + 2 * MIN])
    bounds = es.data_boundaries(s)
    assert bounds == {'bars': 3, 'first_close_utc': '1970-01-02T00:00:00+00:00',
                      'last_close_utc': '1970-01-02T00:02:00+00:00', 'weekly_blocks': 1}

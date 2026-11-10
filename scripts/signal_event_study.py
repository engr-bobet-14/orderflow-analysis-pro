"""Search-only order-flow signal event study (research; outside the sealed Phase 4b campaign).

Question: does any raw detector signal family emitted by the unchanged Phase 4b S1 ``defaults`` pipeline predict
a sign-adjusted price move larger than the 12 bps round-trip cost, independent of stops, targets and sizing?
The full design, its deviations from the recommendation and its limitations are recorded by ``design_record()``
and written into every report. Results are evidence for deciding whether further research is worthwhile, not
proof that profitability is impossible.

Run (only with owner authorization): ``python scripts/signal_event_study.py --checkout <4b worktree> --out <dir>``
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---- Pre-registered design (amended 2026-11-10 after the pre-run review; changing a value is a new experiment) -
COST_BPS = 12.0                                   # 5 bps fee + 1 bps slippage, per side
HORIZONS_MIN = (5, 15, 60, 240, 1440)
FAMILIES = ('absorption', 'initiative_auction', 'exhaustion', 'delta_divergence')
DIRECTIONS = (1, -1)                              # buy, sell
ARMS = ('fixed_tick_imbalance', 'fixed_volume_imbalance')
BLOCK_DAYS = 7
RESAMPLES = 20_000                                # min p 5e-5, 12.5x below the first Holm cutoff 0.05/80
SEED = 20261110
ALPHA = 0.05
MIN_EVENTS = 30
MIN_BLOCKS = 26                                   # about 14 % of the ~191 weekly blocks per search segment
DAY_MS = 86_400_000
MINUTE_MS = 60_000
PROTECTED_OUTPUT_ROOTS = (Path('/Volumes/Market-Data'),)


def design_record():
    return {
        'cost_bps': COST_BPS, 'horizons_min': HORIZONS_MIN, 'families': FAMILIES, 'arms': ARMS,
        'block_days': BLOCK_DAYS, 'resamples': RESAMPLES, 'seed': SEED, 'alpha': ALPHA,
        'min_events': MIN_EVENTS, 'min_blocks': MIN_BLOCKS,
        'data': 'Phase 4b SearchPrefix only (bars and trade updates up to n_search) via the S1 open_prefix path; '
                'opening the production cache decodes and hashes whole files for integrity only (disclosed).',
        'events': 'raw detector signals (BarResult.signals) of the S1 defaults pipeline, per family x direction; '
                  'signals on gap-skipped bars are not emitted; one event per bar per family x direction.',
        'entry': 'open of bar i+1, the first trade after signal bar i closes (zero-latency assumption).',
        'exit': 'close of the last bar closing at or before ts[i] + h (a close-time deadline, not a price at '
                'exactly h).',
        'effective_horizon': 'realised holding time is at most h; on irregular information-driven bars it is '
                             'shorter at short horizons (about 4 min mean at h = 5 for ~60 s bars, negligible at '
                             'h >= 60); per-cell effective_horizon_min_mean/p10 are reported. The 5-minute label '
                             'is approximate.',
        'exclusions': 'windows with no bar closing after entry within h, windows ending after the last '
                      'search-prefix close, and windows containing a data_gap bar are dropped (counted as '
                      'dropped_unobservable); dropping windows without a closing bar conditions on activity.',
        'dedup': 'per family x direction x horizon, greedy in time order: an event is kept only if its decision '
                 'close is at or after the previous kept deadline ts + h and its decision bar is at or after the '
                 'previous exit bar, so its entry bar always lies after the previous window (tied closes at the '
                 'deadline count as overlapping).',
        'baseline': 'same-day drift baseline: the mean forward return, at the same horizon, of every valid '
                    'non-gap decision bar on the same UTC day, sign-adjusted by the event direction. It is not a '
                    'matched random-entry control; it includes the event windows themselves.',
        'test': 'per cell two one-sided tests: gross mean > 12 bps and excess over the baseline > 0. Each p is '
                'the larger of a 7-day block bootstrap p (20 000 resamples, fixed seed; Monte Carlo SE '
                'reported for both tests) and a cluster-robust block-t p (blocks - 1 df). lower_95 bounds are '
                'unadjusted. Any non-finite value stops the run (fail closed).',
        'decision': 'cell p = max(p_gross, p_excess) (intersection-union); Holm over every cell of the run; '
                    'cells with < 30 events or < 26 weekly blocks keep p = 1 and stay in the family. GO if '
                    'any Holm-adjusted cell p < 0.05, else NO-GO.',
        'deviations': [
            'baseline is a same-day drift baseline, not the time-matched random-entry control proposed in the '
            'recommendation section 3',
            'the oracle MFE-minus-MAE stop condition of the recommendation is not part of the decision; MFE and '
            'MAE are reported as separate marginal oracle quantiles only',
            'no trade-level vs bar-level split: under S1 defaults each family has a single source',
            'statistics amended before any run: Holm on max(p_gross, p_excess) instead of separate Holm '
            'families; 20 000 resamples, block-t guard and 26-block minimum instead of 2 000 and 8',
        ],
        'interpretation': 'tests raw detector signals, not S1 trades (entries also need a watched level, '
                          'composite score, cooldown and bias). Evidence for whether further research is '
                          'worthwhile; not proof that profitability is impossible. MFE/MAE are oracle bounds, '
                          'not strategies.',
        'limitations': [
            'both p components are approximate: under negative skew the block-t guard is anti-conservative in '
            'the extreme tail (synthetic null, event skew -0.95 to -3.7, rejection at the first Holm cutoff '
            '6.25e-4: 1.6e-3 to 5.3e-3 at 26 blocks, 9.6e-4 to 1.9e-3 at 191 blocks; the max rule still gave '
            '2.45e-3 at skew -1.75 and 26 blocks); positive skew is conservative. A false '
            'GO needs many cells sitting exactly at their null, so the family-wise effect is small but nonzero; '
            'per-cell event skewness (gross_skew, excess_skew) is reported so any GO cell can be judged.',
            'the MFE/MAE quantiles are marginal oracle bounds, not a strategy.',
        ],
        'memory': 'arms run sequentially in one process; arm 2 is opened through open_prefix with the default '
                  '4 GiB RSS check on the lifetime peak, which fails closed (no output) if exceeded.',
    }


@dataclass
class PriceSeries:
    ts: np.ndarray        # bar close, ms UTC (non-decreasing)
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    gap: np.ndarray       # bar spans missing tick hours


# ---- Forward windows and returns -------------------------------------------------------------------------------
def forward_window(series, horizon_ms):
    """Exit bar per decision bar, and whether the window is fully observed, gap-free and after the entry bar."""
    ts, n = series.ts, len(series.ts)
    decision = np.arange(n)
    end = ts + horizon_ms
    exit_idx = np.searchsorted(ts, end, side='right') - 1
    valid = (decision + 1 < n) & (exit_idx >= decision + 1) & (end <= ts[-1])
    gaps = np.cumsum(series.gap.astype(np.int64))
    safe = np.clip(exit_idx, 0, n - 1)
    valid &= (gaps[safe] - gaps[decision]) == 0       # no gap bar in (i, exit]
    return np.where(valid, exit_idx, -1), valid


def long_returns_bps(series, exit_idx, valid):
    n = len(series.ts)
    entry = np.full(n, np.nan)
    entry[:-1] = series.open[1:]
    out = np.full(n, np.nan)
    j = exit_idx[valid]
    out[valid] = (series.close[j] - entry[valid]) / entry[valid] * 1e4
    return out


def day_baseline(series, ret_long, valid):
    """Same-UTC-day mean forward return of every valid, non-gap decision bar (NaN on days without one)."""
    use = valid & ~series.gap
    days = series.ts // DAY_MS
    uniq, inverse = np.unique(days, return_inverse=True)
    sums = np.bincount(inverse, weights=np.where(use, ret_long, 0.0), minlength=len(uniq))
    counts = np.bincount(inverse, weights=use.astype(float), minlength=len(uniq))
    with np.errstate(invalid='ignore', divide='ignore'):
        means = sums / counts
    return means[inverse]


def dedup(bars, ts, horizon_ms, exit_idx):
    """Greedy non-overlapping windows in time order: past the previous deadline and its (possibly tied) exit bar."""
    kept, window_end, last_exit = [], None, -1
    for bar in np.sort(np.unique(bars)):
        if window_end is None or (ts[bar] >= window_end and bar >= last_exit):
            kept.append(int(bar))
            window_end, last_exit = ts[bar] + horizon_ms, exit_idx[bar]
    return np.asarray(kept, dtype=np.int64)


def excursions(series, i, j, direction):
    """Oracle MFE/MAE in bps over bars i+1..j (decision bar excluded); not a tradable quantity."""
    entry = series.open[i + 1]
    hi, lo = series.high[i + 1:j + 1].max(), series.low[i + 1:j + 1].min()
    up, down = (hi - entry) / entry * 1e4, (entry - lo) / entry * 1e4
    return (up, down) if direction == 1 else (down, up)


# ---- Statistics ---------------------------------------------------------------------------------------------------
def _blocks(values, days, block_days):
    values, days = np.asarray(values, float), np.asarray(days)
    blocks, inverse = np.unique(days // block_days, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(blocks))
    counts = np.bincount(inverse, minlength=len(blocks)).astype(float)
    return sums, counts


def block_bootstrap(values, days, threshold, *, resamples=RESAMPLES, seed=SEED, block_days=BLOCK_DAYS):
    """One-sided p for mean > threshold, resampling whole calendar blocks with replacement."""
    sums, counts = _blocks(values, days, block_days)
    picks = np.random.default_rng(seed).integers(0, len(sums), size=(resamples, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    p = (1 + int(np.count_nonzero(means <= threshold))) / (resamples + 1)
    return {'p': p, 'mc_se': float(np.sqrt(p * (1 - p) / resamples)),
            'lower_95_unadjusted': float(np.quantile(means, 0.05)), 'blocks': len(sums),
            'means': tuple(float(m) for m in means)}


def block_t_p(values, days, threshold, *, block_days=BLOCK_DAYS):
    """One-sided p for mean > threshold from a cluster-robust (CR1, block) standard error, blocks - 1 df."""
    from scipy import stats
    sums, counts = _blocks(values, days, block_days)
    nb, total = len(sums), counts.sum()
    mean = sums.sum() / total
    if nb < 2:
        return 1.0
    resid = sums - mean * counts
    se = np.sqrt(nb / (nb - 1) * float((resid ** 2).sum())) / total
    if se == 0:
        return 0.0 if mean > threshold else 1.0
    return float(stats.t.sf((mean - threshold) / se, nb - 1))


def mean_test(values, days, threshold):
    """The larger of the block bootstrap p and the block-t p; both approximate (see design 'limitations').

    Fails closed on any non-finite value: a NaN would otherwise make p_block_t NaN and be dropped by max().
    """
    if not np.isfinite(np.asarray(values, float)).all():
        raise ValueError('non-finite event value: refusing to compute a p-value')
    boot = block_bootstrap(values, days, threshold)
    p_t = block_t_p(values, days, threshold)
    return {'p': max(boot['p'], p_t), 'p_bootstrap': boot['p'], 'p_bootstrap_mc_se': boot['mc_se'],
            'p_block_t': p_t, 'lower_95_unadjusted_bps': boot['lower_95_unadjusted'], 'blocks': boot['blocks']}


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p, kind='stable')
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return [float(x) for x in adjusted]


def analyse_cell(series, bars, direction, horizon_ms, *, cache=None):
    key = horizon_ms
    if cache is None or key not in cache:
        exit_idx, valid = forward_window(series, horizon_ms)
        ret = long_returns_bps(series, exit_idx, valid)
        computed = (exit_idx, valid, ret, day_baseline(series, ret, valid))
        if cache is not None:
            cache[key] = computed
    exit_idx, valid, ret, base = cache[key] if cache is not None else computed
    raw = np.unique(np.asarray(bars, dtype=np.int64))
    observable = raw[valid[raw]] if len(raw) else raw
    kept = dedup(observable, series.ts, horizon_ms, exit_idx) if len(observable) else observable
    counts = {'raw_events': int(len(raw)), 'dropped_unobservable': int(len(raw) - len(observable)),
              'dropped_overlap': int(len(observable) - len(kept))}
    if len(kept) == 0:
        return {'n': 0, 'blocks': 0, 'p_gross': 1.0, 'p_excess': 1.0, **counts}
    gross = direction * ret[kept]
    excess = gross - direction * base[kept]
    days = series.ts[kept] // DAY_MS
    gross_test = mean_test(gross, days, COST_BPS)
    excess_test = mean_test(excess, days, 0.0)
    held = (series.ts[exit_idx[kept]] - series.ts[kept]) / MINUTE_MS
    mfe, mae = np.array([excursions(series, int(i), int(exit_idx[i]), direction) for i in kept]).T
    return {'n': int(len(kept)), 'blocks': gross_test['blocks'], **counts,
            'mean_bps': float(gross.mean()), 'median_bps': float(np.median(gross)),
            'hit_rate': float((gross > 0).mean()), 'net_of_cost_bps': float(gross.mean() - COST_BPS),
            'lower_95_unadjusted_bps': gross_test['lower_95_unadjusted_bps'], 'p_gross': gross_test['p'],
            'p_gross_bootstrap': gross_test['p_bootstrap'], 'p_gross_bootstrap_mc_se': gross_test['p_bootstrap_mc_se'],
            'p_gross_block_t': gross_test['p_block_t'],
            'mean_excess_bps': float(excess.mean()),
            'lower_95_unadjusted_excess_bps': excess_test['lower_95_unadjusted_bps'], 'p_excess': excess_test['p'],
            'p_excess_bootstrap': excess_test['p_bootstrap'],
            'p_excess_bootstrap_mc_se': excess_test['p_bootstrap_mc_se'], 'p_excess_block_t': excess_test['p_block_t'],
            'gross_skew': _skew(gross), 'excess_skew': _skew(excess),
            'effective_horizon_min_mean': float(held.mean()), 'effective_horizon_min_p10': float(np.quantile(held, .1)),
            'oracle_mfe_bps_p50_p90': [float(np.quantile(mfe, .5)), float(np.quantile(mfe, .9))],
            'oracle_mae_bps_p50_p90': [float(np.quantile(mae, .5)), float(np.quantile(mae, .9))]}


def _skew(values):
    from scipy import stats
    return float(stats.skew(values)) if len(values) > 2 and np.ptp(values) > 0 else 0.0


def decide(cells):
    """Intersection-union p per cell, one Holm family; insufficient cells keep p = 1 and stay in the family."""
    sufficient = [c['n'] >= MIN_EVENTS and c['blocks'] >= MIN_BLOCKS for c in cells]
    p_cell = [max(c['p_gross'], c['p_excess']) if ok else 1.0 for c, ok in zip(cells, sufficient)]
    adjusted = holm(p_cell)
    out = [{**c, 'sufficient': ok, 'p_cell': p, 'p_cell_holm': a}
           for c, ok, p, a in zip(cells, sufficient, p_cell, adjusted)]
    go = [c for c in out if c['p_cell_holm'] < ALPHA]
    return {'decision': 'GO' if go else 'NO-GO', 'family_size': len(cells),
            'insufficient_cells': sufficient.count(False), 'go_cells': go, 'cells': out}


# ---- Pipeline, signal collection, provenance and isolation -----------------------------------------------------
def collect_events(bars, updates, pipeline, *, digest=True):
    """Run the pipeline bar by bar; collect raw detector signals (skipping gap bars) and the replay digest."""
    from orderflow_system.pipeline import ReplayDigest
    n = len(bars)
    arrays = {k: np.empty(n) for k in ('open', 'high', 'low', 'close')}
    ts, gap = np.empty(n, dtype=np.int64), np.zeros(n, bool)
    events, replay = {}, ReplayDigest() if digest else None
    for idx in range(n):
        bar = bars[idx]
        result = pipeline.process_bar(bar, updates[idx])
        if replay is not None:
            replay.update(result)
        ts[idx], gap[idx] = bar.timestamp_ms, bool(getattr(bar, 'data_gap', False))
        for key in arrays:
            arrays[key][idx] = getattr(bar, key)
        if result.gap_skipped:
            continue
        for signal in result.signals:
            family, side = signal.signal_type.value, signal.direction.value
            if family in FAMILIES and side in ('buy', 'sell'):
                bucket = events.setdefault((family, 1 if side == 'buy' else -1), [])
                if not bucket or bucket[-1] != idx:
                    bucket.append(idx)
    series = PriceSeries(ts=ts, gap=gap, **arrays)
    return series, events, (replay.hexdigest() if replay is not None else None)


def build_default_pipeline(bar_type, symbol='BTCUSDT'):
    """Exactly the S1 ``defaults`` construction (phase4_studies.replay_full_strategy -> run_backtest)."""
    from scripts import optimize_orderflow as opt, phase4_studies as p4
    from orderflow_system.pipeline import InfoBarPipeline
    params = p4.full_strategy_params(p4.full_strategy_defaults(symbol), symbol)
    config = p4.effective_trade_config(SimpleNamespace(phase='phase4b'), params)
    p4.validate_phase4b_signals(config.signals, s1=True)
    pipeline = InfoBarPipeline(opt.build_instrument_config(params, symbol), bar_type, trade_config=config)
    pipeline.aggregator.price_proximity_pct = params['price_proximity_pct']
    return pipeline


def require_search_prefix(prefix):
    from scripts.phase4_studies import SearchPrefix
    if not isinstance(prefix, SearchPrefix):
        raise TypeError('the event study reads only a Phase 4b SearchPrefix')
    return prefix


def _under(path, root):
    """Case-insensitive containment (APFS is case-insensitive; resolve() keeps the given case)."""
    path, root = os.path.normpath(str(path)).casefold(), os.path.normpath(str(root)).casefold()
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def check_output_location(out, checkout):
    out = Path(out).resolve()
    if _under(out, Path(checkout).resolve()):
        raise ValueError('event-study output must be outside the sealed campaign checkout')
    for root in PROTECTED_OUTPUT_ROOTS:
        if _under(out, root):
            raise ValueError(f'event-study output must not be under read-only {root} (Market-Data policy)')
    if out.exists():
        raise ValueError(f'event-study output already exists: {out}')


def check_benchmark_binding(benchmark, manifest_sha):
    if benchmark.get('campaign_sha256') != manifest_sha:
        raise ValueError('S1 benchmark does not belong to the sealed S1 manifest')


def verify_default_digest(actual, expected):
    if actual != expected:
        raise ValueError('pipeline does not reproduce the accepted S1 defaults digest')


def code_provenance():
    def git(*args):
        result = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    test = ROOT / 'tests/test_signal_event_study.py'
    return {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'test_sha256': hashlib.sha256(test.read_bytes()).hexdigest() if test.exists() else None,
            'git_head': git('rev-parse', 'HEAD'), 'git_branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
            'git_dirty': bool(git('status', '--porcelain')), 'python': sys.version.split()[0]}


def data_boundaries(series):
    def iso(ms):
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat()
    return {'bars': int(len(series.ts)), 'first_close_utc': iso(series.ts[0]), 'last_close_utc': iso(series.ts[-1]),
            'weekly_blocks': int(len(np.unique(series.ts // DAY_MS // BLOCK_DAYS)))}


def _sealed_json(path):
    raw = Path(path).read_bytes()
    if Path(str(path) + '.sha256').read_text().strip() != hashlib.sha256(raw).hexdigest():
        raise ValueError(f'seal mismatch: {path}')
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def run(checkout, out):
    checkout = Path(checkout).resolve()
    check_output_location(out, checkout)                 # before any input is opened
    from scripts.phase4_policy import PhaseLayout
    from scripts.phase4b_campaign import open_prefix
    layout = PhaseLayout('phase4b-s1', root=checkout)
    manifest, manifest_sha = _sealed_json(layout.manifest)
    benchmark, benchmark_sha = _sealed_json(layout.output / 's1/benchmark.json')
    check_benchmark_binding(benchmark, manifest_sha)
    provenance = code_provenance()
    cells, arms = [], {}
    for bt in ARMS:
        started = time.perf_counter()
        prefix = require_search_prefix(open_prefix(layout, manifest, bt)[0])
        series, events, digest = collect_events(prefix.bars, prefix.trade_updates,
                                               build_default_pipeline(prefix.bars[0].bar_type))
        verify_default_digest(digest, benchmark['lanes'][bt]['replays']['defaults']['replay_digest'])
        cache = {}
        for (family, direction) in [(f, d) for f in FAMILIES for d in DIRECTIONS]:
            for minutes in HORIZONS_MIN:
                cells.append({'arm': bt, 'family': family, 'direction': 'buy' if direction == 1 else 'sell',
                              'horizon_min': minutes,
                              **analyse_cell(series, events.get((family, direction), []), direction,
                                             minutes * MINUTE_MS, cache=cache)})
        arms[bt] = {'n_search': len(prefix.bars), **data_boundaries(series), 'replay_digest': digest,
                    'digest_matches_s1_defaults': True,
                    'raw_events': {f'{f}:{d}': len(v) for (f, d), v in events.items()},
                    'seconds': round(time.perf_counter() - started, 1)}
        del prefix, series, events, cache
        gc.collect()
    result = decide(cells)
    report = {'design': design_record(), 'code': provenance,
              'inputs': {'checkout': str(checkout), 's1_manifest_sha256': manifest_sha,
                         's1_benchmark_sha256': benchmark_sha},
              'arms': arms, **result}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'event_study.json').write_text(json.dumps(report, indent=1, default=float))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--checkout', required=True, help='Phase 4b worktree holding the sealed S1 inputs')
    parser.add_argument('--out', required=True, help='new directory outside the campaign checkout')
    args = parser.parse_args(argv)
    report = run(args.checkout, args.out)
    print(json.dumps({'decision': report['decision'], 'family_size': report['family_size'],
                      'insufficient_cells': report['insufficient_cells'], 'go_cells': len(report['go_cells'])},
                     indent=1))


if __name__ == '__main__':
    main()

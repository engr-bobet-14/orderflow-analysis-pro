# OrderFlow Analysis Pro

> Order-flow strategy on **information-driven bars**: delta analytics, volume-profile framing, pattern detection (absorption, initiative, exhaustion, divergence), a trade state machine, Telegram alerts and a FastAPI dashboard.

Based on Fabio Testa's order-flow methodology. Forked from `mahmoud20138/OrderFlow-Analysis-Pro`; this repository is canonical.

## Status

- **Data source: information-driven bars** (Parquet), replayed chronologically through a deterministic pipeline. The legacy MT5 / Bybit feeds and fixed-time bar builder are removed.
- **Not yet done:** live Binance feed; calibration of detector thresholds against the dataset (defaults are provisional); Telegram entry/exit alerts for *live* use (alerts exist but are off during replay).

## Data

External, read-only dataset (never modified by this project):

```
/Volumes/Market-Data/info-bars/BTCUSDT/<bar_type>/<UTC stamp>.parquet   (+ <UTC stamp>.json sidecar)
```

Confirm the volume is mounted before running. Six bar types, selected with `--bar-type` or `settings.INFO_BARS.bar_type`:

| Bar type | Closes when the running … reaches a threshold |
|---|---|
| `fixed_tick_imbalance` / `fixed_volume_imbalance` / `fixed_dollar_imbalance` | signed tick / volume / dollar imbalance |
| `fixed_tick_run` / `fixed_volume_run` / `fixed_dollar_run` | one-sided tick / volume / dollar run |

### Schema (verified on all six files)

| Column | Meaning |
|---|---|
| `Date Time` | bar **close** time, UTC (µs). Not unique: bars can close in the same ms |
| `Tick Number` | cumulative trade index at bar close — strictly increasing; tie-break for equal timestamps |
| `Open/High/Low/Close` | bar prices |
| `Cumulative Volume`, `… Buy Volume`, `… Sell Volume` | **per-bar** totals despite the name; buy + sell = volume |
| `Cumulative Ticks` | trades in the bar (= diff of `Tick Number`) |
| `Cumulative Dollar Value` | Σ price × size in the bar → bar VWAP = dollars / volume |
| `Threshold` | the bar builder's closing threshold (units depend on bar type) |

Sidecar JSON: bar-builder config, `data_range`, `n_bars`, `missing_tick_hours` (trade-less hours), calibration provenance.

### What the data does **not** contain

No trade-level and no price-level data: no footprint ladder, no horizontal (per-price) delta, no tape, no orderbook. The system **never fabricates them**:

| Feature | On info bars |
|---|---|
| Vertical / cumulative delta | ✅ genuine (bar buy − sell) |
| Footprint | bar totals only; `has_price_levels = False`, `levels = {}`; level queries return `None` (unknown), not 0 |
| Horizontal delta, imbalance prints, level-based absorption | ⛔ unavailable (inert; work only when real ticks are supplied) |
| Volume profile | ⚠ **coarse**: each bar's volume is placed at the bar's own VWAP (`source = "bar_vwap"`). Volume is never spread over the high–low range |
| Sweep detector / orderbook tracker | ⛔ need L2 data; kept for a future depth feed, **not wired** into the info-bar pipeline |

### Data-quality facts (full history, 2019-09-10 → 2023-12-31)

All six sets: no NaN, OHLC consistent, buy + sell = volume, tick counts consistent, timestamps non-decreasing. Notable: millisecond-tied close times (up to ~0.8 % of imbalance bars), heavy-tailed bars (imbalance bars up to ~29 h / 1.4 M trades long), one reported outage (2023-11-21 03:00–08:00 UTC). Bars spanning a reported outage are flagged `data_gap`: all analytical histories advance, but emitted detector signals are discarded. Their participation rate is unknown (`None`). Protective stop/target actions remain active.

## Architecture

```
Parquet ─▶ InfoBarReader ─▶ InfoBarPipeline (sync, no I/O, no wall clock) ─▶ OrderflowSystem
 (data/infobar_feed.py)       (pipeline.py)                                    (main.py)
                              ├─ DeltaEngine        (analytics/delta.py)        ├─ SQLite (optional)
                              ├─ FootprintEngine    (analytics/footprint.py)    ├─ Telegram (off in replay)
                              ├─ Absorption / Initiative / Exhaustion /         └─ Dashboard (FastAPI + WS)
                              │  Divergence detectors (patterns/)
                              ├─ UTC-day session → VolumeProfileEngine → ProfileFramingEngine (bias, levels)
                              └─ SignalAggregator state machine (signals/aggregator.py)
```

Guarantees:

- **Chronological**: bars ordered by (close time, tick number); out-of-order rows are rejected (reader) / raise (pipeline).
- **Deterministic**: every time-dependent component (cooldowns, staleness) runs on an `EventClock` set from each bar's close time. Same bars → identical output (`ReplayDigest`).
- **No look-ahead**: a bar is delivered only after it closed; a UTC-day profile is finalised by the *first bar of the next day* and only then feeds the bias; exits are checked before new entries, so a trade opened on bar N is exposed from N+1; if one bar spans both stop and target the stop is assumed first.
- **Info-bar native "effort"**: bars compress time when activity is high, so participation is measured as volume per second (`InfoBar.volume_rate`), and delta as a share of bar volume (`delta_pct`), not as absolute contracts.

Module map: `orderflow_system/` → `data/` (models, `infobar_feed`, `database`), `analytics/` (delta, footprint, volume_profile, orderbook*), `patterns/`, `signals/` (profile_framing, aggregator), `pipeline.py`, `main.py`, `alerts/`, `dashboard/`, `config/settings.py`, `validate_infobars.py`. (*dormant)

## Patterns on info bars

| Pattern | Rule (bar level) |
|---|---|
| Absorption | `|delta|/volume ≥ min_delta_pct` and the bar closes ≥ `min_adverse_body_bps` *against* the delta (positive delta + down bar = bearish). Level-based method inert without price levels |
| Initiative | strong `delta_pct`, participation rate ≥ `volume_acceleration_min` × rolling average, body ≥ `min_body_bps`, delta and price aligned |
| Exhaustion | highs (lows) non-decreasing (non-increasing) over `min_bars_declining` bars while participation rate falls ≥ `volume_decline_pct`; contrarian-imbalance check skipped when no price levels |
| Divergence | new price extreme by `min_price_new_extreme_bps` while cumulative delta fails to confirm by > `1 − delta_failure_pct` of the window's cumulative-delta range |

Signals carry `details["bar_seq"]` (which bar produced them). **Thresholds are provisional/uncalibrated** — see `InfoBarConfig`/detector configs in `config/settings.py`.

Trade lifecycle (unchanged methodology): `WATCHING → ABSORPTION → POSITION_OPEN → BREAK_EVEN → TRAILING → CLOSED`; exits via strong exhaustion/divergence, **stop** or **target** hit on a closed bar. Composite score = signal weight (absorption 0.30, divergence 0.25, initiative 0.20, other 0.15) + level strength × 0.25 + bias alignment × 0.20; entry needs ≥ 40.

## Setup

```bash
source .venv_orderflow/bin/activate
pip install -e ".[dev]"        # needs pyarrow, fastapi, uvicorn (declared)
```

## Run

```bash
# Replay one bar type (dashboard on http://localhost:8080)
python -m orderflow_system.main --bar-type fixed_dollar_imbalance --start 2023-01-01 --end 2023-04-01

# Options: --speed N (bars/s, 0 = fastest) --persist (SQLite) --no-dashboard --alerts --version <stamp> --root <dir>
python -m orderflow_system.dashboard      # dashboard only (empty until a replay runs)
```

Defaults live in `orderflow_system/config/settings.py` (`INFO_BARS`, `DASHBOARD`, `TELEGRAM`, `DB_PATH`). Telegram alerts are **off during replay** (`--alerts` to force) to avoid sending thousands of historical messages.

## Dashboard / API

The x-axis is bar sequence; `bars=N` selects the most recent N bars. Info bars are never re-aggregated into fixed time buckets. Chart `time` is a strictly-increasing *display* value (several bars can close in the same second); rows also carry exact `time_ms` and `seq`.

`GET /api/instruments · /api/scanner · /api/bars/{sym}?bars= · /api/delta/{sym}?bars= · /api/volume-profile/{sym}?bars= · /api/bias/{sym} · /api/signals/{sym} · /api/markers/{sym}?bars= · /api/trade/{sym} · /api/strategy-status/{sym} · /api/footprint/{sym} (bar totals, no levels) · /api/stats · WS /ws` (channels: `signal`, `trade_state`, `volume_profile`, `bias`, `stats`). Fast replays push only periodic stats; the page polls bars over REST.

## Database (SQLite, optional via `--persist`)

`runs` records dataset identity, source path/version, replay window, configuration digest and creation time. Dataset identity hashes the resolved Parquet path, stem, size, modification time and the consumed JSON sidecar bytes (or their absence). Run identity hashes that identity, bar type, window and configuration. `info_bars` is unique by dataset/instrument/bar type/sequence; `signals` by run/sequence/type/direction/price; `volume_profiles` by run/session. Replaying the same run is idempotent; different windows or datasets remain separate. Reads default to the most recent run for the requested bar type. Partial first-session profiles are marked incomplete and cannot replace complete profiles of the same run. Old databases migrate automatically without deleting or merging historical rows.

## Validation & tests

```bash
pytest                                   # unit + synthetic + real-data tests (real-data skipped if volume absent)
pytest orderflow_system/test_integration.py::test_volume_profile_from_ticks
python -m orderflow_system.validate_infobars --bar-type all [--start 2023-01-01 --end 2023-04-01]
```

`validate_infobars` runs the real files read-only through the whole pipeline and checks: strict schema/row integrity, completion, **deterministic replay** (digest equality over two runs), **no look-ahead** (the first 40 % of bars replayed alone reproduces the same digest as inside the full run), and that no price-level data was produced. The shared `TradeLedger` audit checks entry direction/proximity, management-stop validity, action sequences and executable exit prices; gap-bar protective fills and target gaps have separate counters. First-outage checks verify duration/gap classification independently of replay start.

## Known limitations

- Detector thresholds are uncalibrated; the exhaustion detector is noisy on participation rate. Calibrate on the dataset before trusting signals.
- Bar-construction hyper-parameters in the sidecars were calibrated on the *whole* 2019–2023 history (in-sample for any backtest of this period).
- Volume profile has only bar-VWAP resolution; LVN / shape classification are therefore coarse.
- Within one bar the order of high and low is unknown; stop-vs-target on the same bar resolves pessimistically (stop first). Entry and pattern exits fill at the actionable bar close. Opening gaps through stops fill at the open; targets fill exactly at their resting limit, including opening gaps beyond the target (counted as `target_gap_through`, even when the target lies outside the current bar range). Fees/slippage remain deferred.
- Action-time immutable execution snapshots feed the ledger, alerts and trade broadcasts; current dashboard views read live trade state through the same payload serializer, including watching/reset transitions. The original bracket reward/risk ratio is retained after management changes. Detector extreme references stay separate from fill prices. Break-even is deferred until the stop lies on the protective side of the current close; trailing requires a strictly tighter protective stop. Protective stops/targets do not start a signal cooldown; signal-driven entries, management and exits do.
- Whole bars remain assigned to their close-date UTC session. The first replay session is complete when the first yielded bar has no predecessor in the file or its predecessor closed before that session’s UTC midnight; a predecessor within the same session marks it partial.
- No live feed yet; dashboard replays are single-instrument (BTCUSDT).

## Disclaimer

Research software; not financial advice.

## Orderflow parameter search

Install the research dependencies with `python -m pip install -e '.[tuning]'`, then
run `python scripts/optimize_orderflow.py --help`. The runner ports notebook 02's
six parameter dimensions, plus the absorption/initiative thresholds and the fallback
stop width and target R:R, without changing the pipeline defaults or the notebooks:

```bash
python scripts/optimize_orderflow.py \
  --symbol BTCUSDT --bar-type fixed_dollar_imbalance,fixed_tick_run \
  --start-date 2023-06-01 --end-date 2023-07-31 \
  --n-trials 50 --n-jobs 2 \
  --study-db /tmp/orderflow/studies.db \
  --cache-dir /tmp/orderflow/cache --output /tmp/orderflow/results.json
```

Dates are inclusive UTC days; `START_DATE` and `END_DATE` supply defaults.
`--bar-type all` is the default and selects the six fixed information-bar types.
`all` also expands within comma lists, which are deduplicated in first occurrence
order; use `--bar-type all,time_1m` for all seven types. `time_1m` must be selected
explicitly. Only BTCUSDT has a registered instrument configuration; other symbols
are refused. `--symbol` and `--bar-type` accept comma lists. Defaults for local
artifacts are under ignored `output/optimize/`.
`--data-root` defaults to the configured info-bar root. `--trades-root` defaults
to `/Volumes/Market-Data/orderbook-data_binance/futures_trades` (or
`ORDERFLOW_TRADES_ROOT`). Missing roots are refused before output work begins.
Sources are read-only.

`time_1m` derives candles from those raw trades with DuckDB, one UTC day at a
time, with spills confined to a temporary subdirectory of `--cache-dir`.
Exact duplicate rows are removed first; conflicting trade IDs and invalid
volumes fail closed. Candles become actionable at minute close, skip empty
minutes, and flag gaps longer than one hour. Binary monthly staging files always
cover whole UTC months, including a preceding month for the midnight close and
predecessor. Stages are resumable and keyed by the source inventory and builder
fingerprint. Final range caches publish atomically and workers share read-only
memmaps. Observed coverage is counted from actual bars, rather than inferred
from sidecar dates. A missing day's cause remains unknown.

Rerun with the same arguments to resume. `--n-trials` is the total number of
COMPLETE trials to reach, not an increment. No trial is deleted; stale RUNNING
and failed trials remain visible. Each symbol/bar-type pair gets a study whose
name includes data and strategy identity hashes. Identity includes source
inventory/sidecar identity, reader and transformation code, all execution-affecting
strategy modules, accounting/objective code, non-searched instrument defaults,
search bounds, costs, CV geometry, holdout fraction, minimum trades and seed.
Changing an identity input starts a new study. Changing `--n-trials`, `--n-jobs`,
output paths or the holdout evaluation flag keeps the study. Resume is refused
if Optuna, RiskLabAI or quantstats major.minor versions changed; use a new
study database in that case. Versions and git SHA/dirty state are recorded.

Search-space overrides are a JSON file supplied with `--search-space`, for
example `{"lookback_bars": {"low": 8, "high": 20}}`. Unknown dimensions are
refused. Capital and risk are fixed at $5,000 and 1% equity per trade, with
stop-loss sizing, a 0.05% minimum stop distance, and notional capped at equity.
Default costs are 5 bps taker fee and 1 bp slippage per side; `--fee-bps` and
`--slippage-bps` are part of study identity. There are no capital or risk flags.

The objective is mean daily-return Sharpe (365 periods) across five
chronological, equal-bar-count blocks of the search segment (no model is fitted, so
nothing is purged). One causal replay per trial produces
the same closed trades and sizing as the notebook's repeated prefix replays.
Each fold selects trades by **exit** sequence; entries may precede its scoring
window, and open trades at its end stay open. Accounting compounds over the
prefix, and each fold's daily returns start from the **account equity at the
fold's start** (the equity after the last trade that exited earlier, $5,000 for
the first fold), so returns agree with the position sizing; a fold that opens with
a depleted account is infeasible. (Earlier versions re-anchored every fold at
$5,000, which could report returns below -100%; the study identity is versioned
`continuous_equity_anchor_v2`.) The daily metric anchor
is on the day before the scored window, and daily equity extends through its
last day: this fixes the notebook's loss of day-one PnL and includes trailing
flat days. Undefined metrics become JSON null with a status; a no-loss profit
factor has status `no_losses`.

The objective is two-tier. A trial is feasible when all folds exist and each has
at least `--min-trades-per-fold` trades (default 2), no account depletion and a
finite Sharpe, and the pooled trades reach `--min-total-trades` (default 0). A
feasible trial scores the day-weighted mean fold Sharpe (floored at -50); an
infeasible one scores `-100 + 49 * progress`, so no infeasible trial can outrank a
feasible one. The reported best is the best feasible trial; with none,
`best_degenerate` is true. The provisional floors for `fixed_tick_imbalance` are
105 pooled and 11 per fold: a guard against starved folds, not a significance
test. Daily Sharpe from very few trades is noisy.

This is a rule-based strategy with no fitted training set. The folds are
chronological sub-periods of one continuous replay; nothing is purged and the
bars before a fold serve only as past context. As in notebook 02, `--embargo` (default
0.02) is repurposed as a **scoring blackout** at each test block's start, rather
than AFML's exclusion of future training observations. Fold zero starts cold;
exit attribution and continuous equity make adjacent folds weakly dependent.
Fold scores are in-sample for the tuned parameters and optimistically biased;
the only out-of-sample evidence is 2024+ data (see USAGE.md, dataset policy). CPCV, PBO, nested CV and RiskLabAI's estimator-based HPO wrapper
are intentionally not used.

The chronological tail (`--holdout-fraction 0.2`, or 0 to disable) is never
replayed during search. Add `--evaluate-holdout` for an explicit replay
with the best feasible trial. The holdout starts
scoring after `int(N * embargo)` tail bars; impossible CV/holdout geometry is
refused before trials. Each evaluation gets a distinct timestamp/UUID study
attribute and is retained in JSON history, including `prior_evaluations` and the
COMPLETE trial count `trials_at_evaluation`. Repeated evaluations are allowed and
print a warning that repeated looks erode holdout independence. For information
bars the holdout is **conditional on bars calibrated on the entire history**, not independent
end-to-end validation: `bar_construction_independent: false` and the upstream
hyperparameters/configuration/creation/git metadata are preserved. Time bars
have no calibrated construction parameters.

Holdout daily returns feed RiskLabAI's `deflated_sharpe_gate` as a **diagnostic
only**, never a selection criterion. It is evaluated at two multiplicities, all
COMPLETE trials in this study and all COMPLETE trials for the symbol across studies
(`sensitivity`); the dispersion is the sample standard deviation of finite raw daily
Sharpes from the same trials with unpenalised folds. If the two disagree, `passes`
is `null`. Insufficient or zero dispersion produces `passes: null`,
`status: indeterminate`. `n_nonzero_days`, skew and excess kurtosis are reported
beside it because mostly-zero daily returns strain the gate's assumptions.
Holdouts below the minimum trade count, with account depletion or with non-finite
Sharpe also receive an indeterminate gate and an explicit reason.
The effective number of independent trials is unknown (many trials differ only
in the score threshold), and repeated holdout evaluation after more tuning erodes holdout independence.

SQLite initialization has a database-wide lock. Optimization and holdout have
a non-blocking study admission lock: a second invocation for the same study
exits clearly; different studies may run concurrently. Workers use **spawn**,
open their own SQLite connections, hold shared lifetime locks, handle SIGTERM,
and check for parent death between trials. Admission refuses lingering orphan
workers; wait for their current trial to finish or terminate them. Worker errors
propagate with nonzero exit status and failed trials are retained. Parallel
completion can overshoot the total by at most `n_jobs - 1`; proposals depend on
completion order. With one worker, per-trial seeding makes resumed proposals
reproducible. Parallel worker seeds use separate streams spaced by 1,000,003.
Artifact paths are expanded and resolved before deriving lock paths.
An already-complete study starts no workers. Holdout runs after
all workers join. JSON results merge atomically under a separate lock at
`symbols → symbol → bar_type → study_name`, including best parameters/objective,
scored-bar sample count, trade count, trial states, fingerprints and provenance.

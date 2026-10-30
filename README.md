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

# Orderflow Analysis Pro

## Overview

Order-flow trading strategy: footprint charts, delta analytics, volume profile, and pattern detection on information-driven bars, with Telegram signal alerts and a FastAPI dashboard with WebSocket.

Forked from mahmoud20138/OrderFlow-Analysis-Pro.

## Status

* Migrated to information-driven bars: `Parquet → InfoBarReader → InfoBarPipeline → OrderflowSystem`. MT5/Bybit feeds and the fixed-time bar builder are removed.
* Default source: info-bar replay (`settings.INFO_BARS`, default `fixed_dollar_imbalance`); no MT5 terminal needed.
* Not done: live Binance feed; calibration of order-flow thresholds against the dataset (defaults are provisional); live Telegram entry/exit alerts (off during replay).

## Data

* Primary information-driven bar dataset used for order-flow strategy calibration, backtesting, and validation.
* Path: `/Volumes/Market-Data/info-bars/BTCUSDT/<bar_type>/<UTC timestamp>.parquet` (+ `.json` sidecar)
* Bar types: `fixed_{dollar,tick,volume}_{imbalance,run}`
* External volume: confirm `/Volumes/Market-Data` is mounted before reading. Read-only: never write there.
* Columns: close-time `Date Time` (UTC), `Tick Number`, OHLC, per-bar volume/buy/sell, ticks, dollar value, `Threshold`. No price-level or trade-level data → never fabricate footprint, horizontal delta, tape or book (see README "What the data does not contain").
* Ordering is (close time, `Tick Number`); close times can tie. Bars spanning `missing_tick_hours` get `data_gap`: analytical histories advance, emitted detector signals are discarded, and protective stop/target actions remain active.

## Working Rules

* Be brief and concise.
* Default branch is master.
* Prefer minimal, targeted edits over large refactors.
* Do not modify unrelated files — note them in the final response instead.
* Use Graphify to understand code relationships before making broad changes.
* Activate the project environment (see Commands) before running Python.
* Run the smallest relevant test set after changes.
* Before the final response, check git diff and review all changes for correctness, regressions, incomplete work, and unintended edits.
* Do not stage, commit, amend, rebase, merge, or push Git changes unless explicitly asked — leave changes in the working tree.
* Use /codex:review when available, followed by /codex:adversarial-review to challenge assumptions and catch missed issues.
* Fix issues found during review before responding.
* End completed implementation tasks with `FINAL VERDICT: APPROVED FOR PRODUCTION` or `FINAL VERDICT: NOT APPROVED FOR PRODUCTION`.
* Approve only when relevant validation passes and no known blocking issues remain.

## Commands

* Environment: `source .venv_orderflow/bin/activate`
* Install: `pip install -e ".[dev]"`
* Optional tuning deps: `pip install -e ".[tuning]"`
* Run system: `python -m orderflow_system.main --bar-type fixed_dollar_imbalance --start 2023-01-01 --end 2023-04-01`
* Run dashboard: `python -m orderflow_system.dashboard` (FastAPI/uvicorn, http://localhost:8080)
* Tests: `pytest` (`tests/` = feed, pipeline, real-dataset; `orderflow_system/test_integration.py` = engines/detectors/DB)
* Single test: `pytest orderflow_system/test_integration.py::test_volume_profile_from_ticks`
* Real-data validation (read-only, all pipeline guarantees): `python -m orderflow_system.validate_infobars --bar-type all [--start D --end D]`

## Architecture

* Authoritative architecture: `orderflow_system/pipeline.py` (pure, deterministic analysis) + `orderflow_system/main.py` (I/O)
* Core package: `orderflow_system/`
* Analytics: `orderflow_system/analytics/`
* Patterns: `orderflow_system/patterns/`
* Signals: `orderflow_system/signals/`
* Dashboard: `orderflow_system/dashboard/`
* Info-bar models, Parquet reader/replay, SQLite: `orderflow_system/data/` (`models.py`, `infobar_feed.py`, `database.py`)
* Execution snapshots and shared ledger: `data/models.py`, `ledger.py`; action consumers use immutable snapshots; current dashboard views serialize live trade state with the same schema. Persisted runs retain dataset/window/config provenance.
* Telegram alerts: `orderflow_system/alerts/`
* Configuration: `orderflow_system/config/`
* All settings (incl. INFO_BARS, DASHBOARD host/port): `orderflow_system/config/settings.py`
* Invariants: analytics are synchronous with no wall clock (use `EventClock`); no look-ahead; thresholds are scale-free (delta/volume, bps), not absolute contracts/ticks.
* Dormant (need L2 data, not wired): `analytics/orderbook.py`, `patterns/sweep.py`.

## Caveats

* Treat this repository as canonical, not the upstream fork.
* Do not preserve upstream implementation choices solely for compatibility — choose what fits this repo's design.
* `.claude/worktrees/` contains working copies and is not the canonical source tree.

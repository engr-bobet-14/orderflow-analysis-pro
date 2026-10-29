# Orderflow Analysis Pro

## Overview

Order-flow trading strategy: footprint charts, delta analytics, volume profile, and pattern detection on information-driven bars, with Telegram signal alerts and a FastAPI dashboard with WebSocket.

Forked from mahmoud20138/OrderFlow-Analysis-Pro.

## Status

* Runtime feeds in `orderflow_system/data/` still use legacy Bybit/MT5 implementations, slated for removal or refactoring.
* Default `DATA_SOURCE` is `MT5` (`orderflow_system/config/settings.py:926`); running the system needs an MT5 terminal until the Binance migration lands.
* In progress: migration to Binance data feeds and information-driven bars.
* Next: calibrate the order-flow logic against the info-bar dataset (see Data).
* Planned: Telegram entry/exit trade alerts.

## Data

* Primary information-driven bar dataset used for order-flow strategy calibration, backtesting, and validation.
* Path: `/Volumes/Market-Data/info-bars/BTCUSDT/<bar_type>/<UTC timestamp>.parquet` (+ `.json` sidecar)
* Bar types: `fixed_{dollar,tick,volume}_{imbalance,run}`
* External volume: confirm `/Volumes/Market-Data` is mounted before reading.

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
* Run system: `python -m orderflow_system.main`
* Run dashboard: `python -m orderflow_system.dashboard` (FastAPI/uvicorn, http://localhost:8080)
* Tests: `pytest` (all tests are in `orderflow_system/test_integration.py`; `tests/` is empty)
* Single test: `pytest orderflow_system/test_integration.py::test_volume_profile`

## Architecture

* Authoritative architecture: `orderflow_system/main.py`
* Core package: `orderflow_system/`
* Analytics: `orderflow_system/analytics/`
* Patterns: `orderflow_system/patterns/`
* Signals: `orderflow_system/signals/`
* Dashboard: `orderflow_system/dashboard/`
* Data feeds, candle builder, SQLite: `orderflow_system/data/`
* Telegram alerts: `orderflow_system/alerts/`
* Configuration: `orderflow_system/config/`
* All settings (incl. DATA_SOURCE, DASHBOARD host/port): `orderflow_system/config/settings.py`

## Caveats

* Treat this repository as canonical, not the upstream fork.
* Do not preserve upstream implementation choices solely for compatibility — choose what fits this repo's design.
* `.claude/worktrees/` contains working copies and is not the canonical source tree.

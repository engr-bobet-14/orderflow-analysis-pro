# Usage Guide

Step-by-step use of `scripts/optimize_orderflow.py`: resumable Optuna search of the rule-based orderflow strategy on information-driven bars (or 1-minute candles), scored with RiskLabAI `PurgedKFold` and an optional holdout.
All commands run from the repo root. Methodology and limits are in the README section "Orderflow parameter search".

```
step 0 setup -> step 1 data -> step 2 search -> step 3 resume / parallel -> step 4 holdout -> step 5 read results
```

Whole pipeline:

```bash
source .venv_orderflow/bin/activate
python scripts/optimize_orderflow.py --bar-type fixed_dollar_imbalance --n-trials 50       # steps 2-3
python scripts/optimize_orderflow.py --bar-type fixed_dollar_imbalance --n-trials 50 --evaluate-holdout   # step 4
```

## Step 0 - Setup

```bash
source .venv_orderflow/bin/activate
pip install -e ".[tuning]"                # REQUIRED once per venv: RiskLabAI, optuna, quantstats, duckdb
pytest tests/test_optimize_orderflow.py   # optional sanity check, ~30 s (needs: pip install -e ".[dev,tuning]")
python scripts/optimize_orderflow.py --help
```

`pip install -e ".[tuning]"` installs this project in editable mode (`-e`: code edits take effect without reinstalling) plus the optional `tuning` dependency group from `pyproject.toml`. Without it the script fails at import. Keep the quotes: zsh treats bare `[...]` as a wildcard. `.[dev,tuning]` also adds pytest.

## Step 1 - Data

Raw inputs are read-only; nothing is ever written to them. Confirm `/Volumes/Market-Data` is mounted (missing roots are refused before any work).

| Source | Default | Used for |
|---|---|---|
| Info bars `<root>/<symbol>/<bar_type>/<stamp>.parquet` | `--data-root` (the `INFO_BARS` root) | the six `fixed_{dollar,tick,volume}_{imbalance,run}` types |
| Raw futures trades `date=*/symbol=*/*.parquet` | `--trades-root` or `ORDERFLOW_TRADES_ROOT` | `time_1m` 1-minute baseline candles |

- `--bar-type all` (default) = the six info-bar types. Add the baseline explicitly: `--bar-type all,time_1m`. Any comma list works (`fixed_tick_run,fixed_dollar_imbalance`).
- Only `BTCUSDT` has an instrument config today; other `--symbol` values are refused.
- Range: `--start-date` / `--end-date` (inclusive UTC days; defaults `START_DATE` / `END_DATE` env, else 2019-09-10 .. 2023-12-31).
- First run per range builds a binary bar cache in `--cache-dir` (streamed, memmapped by workers). `time_1m` also builds monthly stage files from raw trades; an interrupted build resumes.
- Defaults for local artifacts: `output/optimize/{studies.db,results.json,cache/}` (git-ignored). Override with `--study-db`, `--output`, `--cache-dir`.

## Step 2 - Search

```bash
python scripts/optimize_orderflow.py --symbol BTCUSDT --bar-type fixed_dollar_imbalance \
  --start-date 2023-01-01 --end-date 2023-12-31 --n-trials 50
```

- One study per (symbol, bar type). Six parameters are searched (price proximity, composite score, cooldown, volume decline, divergence lookback, delta failure); bounds can be overridden with `--search-space file.json`, e.g. `{"lookback_bars": {"low": 8, "high": 20}}`.
- Accounting is fixed: $5,000 capital, no leverage, 1% equity risk per trade sized by the stop. There are no capital or risk flags. Costs: `--fee-bps 5`, `--slippage-bps 1` (part of study identity).
- Objective = mean fold Sharpe over `--n-splits 5` purged blocks; `--embargo 0.02` is a scoring blackout at each block start; folds with fewer than `--min-trades-per-fold 2` trades score -5 (raise it: Sharpe from few trades is noisy).
- The last `--holdout-fraction 0.2` of bars is never replayed during the search (`0` disables).
- Runtime: about 45 s per trial on ~2 M bars. Console and `results.json` report best parameters, objective, scored-bar sample count, trade count, trial counts by state.

## Step 3 - Resume and parallel

```bash
python scripts/optimize_orderflow.py ...same arguments... --n-trials 100            # tops up to 100 total
python scripts/optimize_orderflow.py ...same arguments... --n-jobs 4                # 4 spawn workers, one SQLite study
```

- Rerun with the same identity inputs to resume. `--n-trials` is the **total** to reach, not an increment; an already-complete study adds none. No trial is ever deleted.
- Identity = data, strategy source, search bounds, costs, CV geometry, holdout fraction, min trades, seed. Changing any of them starts a new study (new name in `results.json`). `--n-trials`, `--n-jobs` and output paths do not.
- Resume is refused if Optuna, RiskLabAI or quantstats major.minor changed; use a new `--study-db`.
- A second invocation of the same study exits with an error; different studies may run concurrently. Parallel runs can overshoot by up to `--n-jobs - 1` trials and propose in completion order; `--n-jobs 1` with the same `--seed` is reproducible.

## Step 4 - Holdout (explicit, rare)

**What it is.** The search (steps 2-3) only ever replays the first 80% of bars (`--holdout-fraction 0.2`), scoring trials on purged CV blocks of that segment. Because the same folds both rank and report the trials, the best CV score is optimistic. The last 20% (the holdout) is never replayed during the search, so `--evaluate-holdout` is the one honest "final exam" of the winning parameters. Run it once, after the search has settled.

```bash
python scripts/optimize_orderflow.py ...same arguments... --evaluate-holdout
```

The same arguments resume the same study and add no trials; the only extra cost is one full replay (~1 min per bar type). It needs `--holdout-fraction` above 0 and a tail large enough to score.

- Replays the whole timeline with the best trial that has an unpenalised fold and scores only the held-out tail.
- RiskLabAI `deflated_sharpe_gate` is applied at this study's COMPLETE trial count; with too little dispersion the gate is `indeterminate`, never a silent pass.
- Every evaluation is kept as a record; repeating it prints a warning because repeated looks erode independence.
- For info bars the result is flagged `bar_construction_independent: false`: those bars were calibrated on the full 2019-2023 history, so the holdout is not independent end-to-end validation.

Reading the gate in `results.json` (holdout record):

| `passes` | Meaning |
|---|---|
| `true` | Held-out Sharpe beats the expected best-of-N Sharpe for this study's trial count. Encouraging, not proof. |
| `false` | The tuned result is not distinguishable from picking the luckiest of N trials. |
| `null` (`status: indeterminate`, with a `reason`) | Not enough information: too few trials with unpenalised folds, zero Sharpe dispersion, or a holdout with too few trades, a blown account or a non-finite Sharpe. Never read it as a pass. |

## Step 5 - Read the results

`results.json` is keyed `symbols -> symbol -> bar_type -> study_name`. Each entry holds `best_params`, `best_objective`, `n_scored_bars`, `total_trades`, `trial_counts`, `best_degenerate` (all folds penalised), holdout history, fingerprints, git SHA/dirty and library versions. Full trial history is in the SQLite `--study-db`:

```bash
python - <<'EOF'
import optuna
s = optuna.load_study(study_name="<study_name>", storage="sqlite:///output/optimize/studies.db")
print(s.best_value, s.best_params, len(s.trials))
EOF
```

## Notes

- Scratch outputs: point `--study-db/--cache-dir/--output` outside the repo for experiments.
- Tuning does not create edge: treat the CV score as optimistic and gate any candidate by the holdout and the deflated Sharpe.

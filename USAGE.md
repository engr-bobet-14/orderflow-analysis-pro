# Usage Guide

Step-by-step use of `scripts/optimize_orderflow.py`: resumable Optuna search of the rule-based orderflow strategy on information-driven bars (or 1-minute candles), scored on chronological sub-period folds (no model is fitted, so nothing is purged) and an optional holdout.
All commands run from the repo root. Methodology and limits are in the README section "Orderflow parameter search".

```
step 0 setup -> step 1 data -> step 2 search -> step 3 resume / parallel -> step 4 holdout -> step 5 read results
```

Whole pipeline:

```bash
source .venv_orderflow/bin/activate
python scripts/optimize_orderflow.py --bar-type fixed_dollar_imbalance --n-trials 300       # steps 2-3
python scripts/optimize_orderflow.py --bar-type fixed_dollar_imbalance --n-trials 300 --evaluate-holdout   # step 4
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
  --start-date 2023-01-01 --end-date 2023-12-31 --n-trials 300
```

- One study per (symbol, bar type). Twelve parameters are searched (price proximity, composite score, cooldown, volume decline, divergence lookback, delta failure, absorption and initiative thresholds, fallback stop width and target R:R); bounds can be overridden with `--search-space file.json`, e.g. `{"lookback_bars": {"low": 8, "high": 20}}`.
- Accounting is fixed: $5,000 capital, no leverage, 1% equity risk per trade sized by the stop. There are no capital or risk flags. Costs: `--fee-bps 5`, `--slippage-bps 1` (part of study identity).
- Objective is two-tier. A trial is **feasible** only if all `--n-splits 5` chronological folds exist, each has at least `--min-trades-per-fold` trades (default 2), no blown account and a finite Sharpe, and the pooled trades reach `--min-total-trades` (default 0). A feasible trial scores the day-weighted mean fold Sharpe (floored at -50). An infeasible trial scores `-100 + 49 * progress` (progress is how close the folds and the pooled total are to their floors), so **no infeasible trial can outrank a feasible one**. The best trial reported is the best feasible one; if none is feasible, `best_degenerate` is true and the best-scoring infeasible trial is only named in `best_infeasible_trial_number`.
- The approved provisional floors for `fixed_tick_imbalance` are `--min-total-trades 105 --min-trades-per-fold 11`. They are a guard against starved folds, not a significance test: 105 independent trades give about 80% power to detect an edge equal to the round-trip cost under a normal approximation, but an 11-trade fold alone has about 20%.
- `--embargo 0.02` is a scoring blackout at the start of each fold (warm-up and open-trade carryover), not an AFML post-test embargo. Folds are chronological sub-periods of one continuous replay; no model is fitted, so nothing is purged.
- Fold returns start from the account equity at the fold's start (not a fresh $5,000), so Sharpe, drawdown and position sizing describe the same account; a fold that opens with a depleted account is infeasible.
- The objective definition (tier constants, floors, aggregation, accounting version) is part of the study identity: changing it starts a new study and never mixes trials. Studies created before the accounting fix are a different identity and cannot be resumed or mixed.
- The last `--holdout-fraction 0.2` of bars is never replayed during the search (`0` disables).
- Runtime: about 45 s per trial on ~2 M bars. Console and `results.json` report best parameters, objective, scored-bar sample count, trade count, trial counts by state.

## Step 3 - Resume and parallel

```bash
python scripts/optimize_orderflow.py ...same arguments... --n-trials 600            # tops up to 600 total
python scripts/optimize_orderflow.py ...same arguments... --n-jobs 4                # 4 spawn workers, one SQLite study
```

- Rerun with the same identity inputs to resume. `--n-trials` is the **total** to reach, not an increment; an already-complete study adds none. No trial is ever deleted.
- Identity = data, strategy source, search bounds, costs, CV geometry, holdout fraction, min trades, seed. Changing any of them starts a new study (new name in `results.json`). `--n-trials`, `--n-jobs` and output paths do not.
- Warm start (default on) applies only when a command creates a **new** study: an existing study with the original identity always resumes as before. `--no-warm-start` and `--warm-start-max` change which study a command resolves to (a warm study has a different name from a cold one), and a warm study's seeds are fixed when it is created. See Step 6 for the resolution order.
- Resume is refused if Optuna, RiskLabAI or quantstats major.minor changed; use a new `--study-db`.
- A second invocation of the same study exits with an error; different studies may run concurrently. Parallel runs can overshoot by up to `--n-jobs - 1` trials and propose in completion order; `--n-jobs 1` with the same `--seed` is reproducible.

## Step 4 - Holdout (explicit, rare)

**What it is.** The search (steps 2-3) only ever replays the first 80% of bars (`--holdout-fraction 0.2`), scoring trials on chronological sub-period folds of that segment. Because the same folds both rank and report the trials, the best fold score is optimistic (it is in-sample for the tuned parameters). The last 20% (the holdout) is never replayed during the search, so `--evaluate-holdout` is the one honest "final exam" of the winning parameters. Run it once, after the search has settled.

```bash
python scripts/optimize_orderflow.py ...same arguments... --evaluate-holdout
```

The same arguments resume the same study and add no trials; the only extra cost is one full replay (~1 min per bar type). It needs `--holdout-fraction` above 0 and a tail large enough to score.

- Replays the whole timeline with the best **feasible** trial and scores only the held-out tail. With no feasible trial it reports `not_evaluated` (reason `no feasible trial to evaluate`).
- RiskLabAI `deflated_sharpe_gate` is reported as a **diagnostic only** (never a selection criterion), at this study's COMPLETE trial count and at the symbol-wide count across all studies (`sensitivity`), next to `n_nonzero_days`, skew and excess kurtosis. If the two counts disagree, `passes` is `null`; with too little dispersion the gate is `indeterminate`, never a silent pass.
- Every evaluation is kept as a record; repeating it prints a warning because repeated looks erode independence.
- For info bars the result is flagged `bar_construction_independent: false`: those bars were calibrated on the full 2019-2023 history, so the holdout is not independent end-to-end validation.

Reading the gate in `results.json` (holdout record):

| `passes` | Meaning |
|---|---|
| `true` | Held-out Sharpe beats the expected best-of-N Sharpe for this study's trial count. Encouraging, not proof. |
| `false` | The tuned result is not distinguishable from picking the luckiest of N trials. |
| `null` (`status: indeterminate`, with a `reason`, or sensitivity entries that disagree) | Not enough information: too few trials with unpenalised folds, zero Sharpe dispersion, or a holdout with too few trades, a blown account or a non-finite Sharpe. Never read it as a pass. |

## Step 5 - Read the results

`results.json` is keyed `symbols -> symbol -> bar_type -> study_name`. Each entry holds `best_params`, `best_objective`, `n_scored_bars`, `total_trades`, `trial_counts`, `best_degenerate` (no feasible trial), `best_infeasible_trial_number`, holdout history, fingerprints, git SHA/dirty and library versions. Full trial history is in the SQLite `--study-db`:

```bash
python - <<'EOF'
import optuna
s = optuna.load_study(study_name="<study_name>", storage="sqlite:///output/optimize/studies.db")
print(s.best_value, s.best_params, len(s.trials))
EOF
```

## Step 6 - Search-space v2 campaign (not launched)

Two studies with wider, per-bar-type bounds (`scripts/search_spaces/<bar_type>.json`), warm-started from the best trials already in `output/optimize/main/studies.db`. **Launching requires the user's explicit approval**; run the dry-run first. `--dry-run` still needs `/Volumes/Market-Data` mounted and may write the bar cache under `output/optimize/cache`, but it creates no study, DB or results file. The replays of the best trials (see `scripts/search_spaces/FINDINGS.md`) found no statistically significant gross edge in the best trials of the 200-trial studies, so this campaign tests search-space hygiene, not a known edge.

Shared flags: `--symbol BTCUSDT --start-date 2019-09-10 --end-date 2023-12-31 --n-jobs 1 --n-splits 5 --embargo 0.02 --holdout-fraction 0.2 --min-total-trades 105 --min-trades-per-fold 11 --fee-bps 5 --slippage-bps 1 --seed 43`. Never add `--no-warm-start` to a campaign command: a cold study has a different name, and once it exists the default command resumes it instead of the warm one. Seed 43 (prior runs used 42) decorrelates TPE's startup from the 200-trial runs. `--study-db output/optimize/main/studies.db` is the DB holding the warm-start sources; a separate `--output output/optimize/campaign2/results.json` leaves the old `results.json` untouched. The campaign is staged: **stage 1 is 300 trials per bar type**, then a check decides whether to top up (see "Stage check and top-up" below). Wall-clock at the observed per-trial time with `--n-jobs 1`: about 5.8 h for 300 tick trials (69 s/trial) and about 4.8 h for 300 volume trials (58 s/trial); a top-up to 600 totals about 11.5 h (tick) and 9.7 h (volume), and the ceiling of 800 about 15.3 h (tick) and 12.9 h (volume).

Tick launch (stage 1, 300 trials):

```bash
python scripts/optimize_orderflow.py --bar-type fixed_tick_imbalance --n-trials 300 \
  --search-space scripts/search_spaces/fixed_tick_imbalance.json \
  --symbol BTCUSDT --start-date 2019-09-10 --end-date 2023-12-31 --n-jobs 1 --n-splits 5 \
  --embargo 0.02 --holdout-fraction 0.2 \
  --min-total-trades 105 --min-trades-per-fold 11 --fee-bps 5 --slippage-bps 1 --seed 43 \
  --study-db output/optimize/main/studies.db --output output/optimize/campaign2/results.json
```

Tick dry-run (prints the plan as one JSON line; creates no study, DB or results file):

```bash
python scripts/optimize_orderflow.py --bar-type fixed_tick_imbalance --n-trials 300 \
  --search-space scripts/search_spaces/fixed_tick_imbalance.json \
  --symbol BTCUSDT --start-date 2019-09-10 --end-date 2023-12-31 --n-jobs 1 --n-splits 5 \
  --embargo 0.02 --holdout-fraction 0.2 \
  --min-total-trades 105 --min-trades-per-fold 11 --fee-bps 5 --slippage-bps 1 --seed 43 \
  --study-db output/optimize/main/studies.db --output output/optimize/campaign2/results.json \
  --dry-run
```

Volume launch (stage 1, 300 trials):

```bash
python scripts/optimize_orderflow.py --bar-type fixed_volume_imbalance --n-trials 300 \
  --search-space scripts/search_spaces/fixed_volume_imbalance.json \
  --symbol BTCUSDT --start-date 2019-09-10 --end-date 2023-12-31 --n-jobs 1 --n-splits 5 \
  --embargo 0.02 --holdout-fraction 0.2 \
  --min-total-trades 105 --min-trades-per-fold 11 --fee-bps 5 --slippage-bps 1 --seed 43 \
  --study-db output/optimize/main/studies.db --output output/optimize/campaign2/results.json
```

Volume dry-run:

```bash
python scripts/optimize_orderflow.py --bar-type fixed_volume_imbalance --n-trials 300 \
  --search-space scripts/search_spaces/fixed_volume_imbalance.json \
  --symbol BTCUSDT --start-date 2019-09-10 --end-date 2023-12-31 --n-jobs 1 --n-splits 5 \
  --embargo 0.02 --holdout-fraction 0.2 \
  --min-total-trades 105 --min-trades-per-fold 11 --fee-bps 5 --slippage-bps 1 --seed 43 \
  --study-db output/optimize/main/studies.db --output output/optimize/campaign2/results.json \
  --dry-run
```

Dry-run fields: `mode` (`new_cold`, `new_warm`, `resume_cold`, `resume_warm`), `study_exists`, `widened_vs_default` (dimensions whose bounds exceed the built-in ones), `floors`, `costs`, `window` (bars and search bars), `warm_start` (`seeds`, `skipped` with reasons) and `holdout` (`evaluate_holdout`, `holdout_fraction`, `embargo`, `prior_evaluations` already recorded on an existing study). `main()` writes no `results.json` in dry-run.

### Stage check and top-up

When a stage finishes, read the best trial from `output/optimize/campaign2/results.json` (`best_objective`, `best_trial_number`; the earlier 200-trial bests were tick -0.223 and volume -0.940) and run the read-only boundary audit on each new study (it opens the DB read-only and reads the search space from the study's stored identity):

```bash
python scripts/search_space_audit.py --study-db output/optimize/main/studies.db \
  --study-name BTCUSDT__fixed_tick_imbalance__2019-09-10_2023-12-31__d72a3440a__s02650ab3
python scripts/search_space_audit.py --study-db output/optimize/main/studies.db \
  --study-name BTCUSDT__fixed_volume_imbalance__2019-09-10_2023-12-31__db53f26f0__s4462e3ca
```

The audit prints, per dimension, how many of the 10 best feasible trials sit in the outer 10% of the range (`top_lo`, `top_hi`), the share of the last 50 trials in the outer 5% (`tail_lo`, `tail_hi`), and a `widen` flag (at least 5 of the top 10, or at least 30% of the tail). See `scripts/search_spaces/FINDINGS.md` for the rule and the 200-trial baseline.

Decision rule:

- **Stop at 300** if `best_trial_number` is not in the last 100 trials (the best has plateaued) and no widened dimension is still flagged.
- **Top up to 600** if the best trial is among the last 100 trials, or a widened dimension is still flagged `widen`. Rerun the same launch command with `--n-trials 600` (the total to reach, not an increment); it resumes the same warm study with its persisted seeds, because `--n-trials` is not part of the identity.
- **Go to 800 (tick only)** if the 600-trial result is still improving and some trial has a positive fold-level result.
- If nothing beats the 200-trial bests, stop: the strategy, not the search, is the limit.

Warm start (new studies only; opt out with `--no-warm-start`, cap with `--warm-start-max`, default 3):

- Sources are prior studies of the same symbol and bar type in `--study-db`, fitted on the 2019-2023 window; each contributes at most its best eligible trial.
- Eligible: COMPLETE, feasible, finite, with every parameter inside the new bounds and of the right kind. An out-of-bounds trial is skipped (never clamped) and the next-best trial of that study is tried; every skip is logged with its reason.
- Dimensions missing in old trials are filled from the baseline values (`baseline_extra_params`).
- Studies that saw holdout data, and studies outside the development window, are never sources. A source must also have the same start/end dates and bars (data fingerprint) as the new study and a recorded `holdout_fraction` at least as large, so its search never covered the new study's holdout; a study whose holdout fraction is unknown is skipped.
- A results file refuses a second, different warm start (other seeds, or cold vs warm) under the same study name; use another `--output`.
- Seeds are ordered (value descending, study name, trial number), enqueued once before the first trial, count toward `--n-trials`, and are fully re-evaluated under the new study's costs, folds and bounds; old values are never copied. Source studies are opened read-only.

Resolution order: an existing cold study with the original identity resumes first (selection never affects a resume); then an existing warm study resumes with its **persisted** seeds; otherwise a new study seeds itself from the sources. Derived (warm-started) studies are never sources, so extending a source study never changes a warm study that was already created.

## Smoke test (BTCUSDT, 2019-2023 development data, 5 trials)

A short end-to-end check before any real campaign. It uses development data only, needs `/Volumes/Market-Data` mounted, and takes roughly 7-10 minutes (about 76 s per trial on `fixed_tick_imbalance`).

```bash
cd /Users/bobet/Documents/Code-Repository/Trading/orderflow-analysis-pro
source .venv_orderflow/bin/activate

python scripts/optimize_orderflow.py \
  --symbol BTCUSDT --bar-type fixed_tick_imbalance \
  --start-date 2019-09-10 --end-date 2023-12-31 \
  --n-trials 5 --n-jobs 1 --n-splits 5 \
  --min-total-trades 105 --min-trades-per-fold 11 \
  --study-db output/optimize/smoke/studies.db \
  --output output/optimize/smoke/results.json
```

- It uses its own `smoke/` database and results file, so it never mixes with other studies; the bar cache is shared (`output/optimize/cache`), so nothing is rebuilt.
- No `--evaluate-holdout`: the pseudo-holdout stays unspent.
- Expect `trials={'COMPLETE': 5, ...}`. With 5 random trials and floors of 105 / 11, `best=None` (`best_degenerate: true`) is a valid result: it means no trial was feasible. `best_infeasible_trial_number` names the closest one.

Optional follow-ups on the smoke results:

```bash
NAME=$(python -c "import json;d=json.load(open('output/optimize/smoke/results.json'));print(next(iter(next(iter(d['symbols']['BTCUSDT'].values())))))")

python scripts/compare_bar_types.py --results output/optimize/smoke/results.json \
  --n-splits 5 --min-trades-per-fold 11 --min-total-trades 105

python scripts/walk_forward_selection.py --study-db output/optimize/smoke/studies.db \
  --study-name "$NAME" --min-trades 11

# only if the study has a feasible trial (otherwise it stops with "no feasible trial"):
python scripts/diagnose_trades.py --study-db output/optimize/smoke/studies.db \
  --study-name "$NAME" --floors --stress
```

### Defaults (what happens if you omit a flag)

| Flag | Default | Note |
|---|---|---|
| `--symbol` | `BTCUSDT` | |
| `--bar-type` | **all six** info-bar types | The smoke test pins `fixed_tick_imbalance`; omitting it runs every bar type. |
| `--start-date` / `--end-date` | `2019-09-10` / `2023-12-31` | Inclusive UTC days; env `START_DATE` / `END_DATE` override. The optimizer warns if the end is after 2023-12-31. |
| `--n-trials` | `300` | Total COMPLETE trials per study (not an increment). |
| `--n-jobs` | `1` | |
| `--n-splits` | `5` | Chronological folds. |
| `--min-trades-per-fold` | `2` | **Not** the approved floor: pass `11` explicitly. |
| `--min-total-trades` | `0` | **Not** the approved floor: pass `105` explicitly. |
| `--embargo` | `0.02` | Scoring blackout at each fold start. |
| `--holdout-fraction` | `0.2` | Last 20% of bars never replayed during the search. |
| `--evaluate-holdout` | off | |
| `--no-warm-start` / `--warm-start-max` | warm start **on** / `3` | Applies to new studies only; see Step 6. |
| `--dry-run` | off | Prints the resolved plan; creates no study, DB or results file. |
| `--fee-bps` / `--slippage-bps` | `5.0` / `1.0` | Part of the study identity. |
| `--seed` | `42` | |
| `--study-db` | `output/optimize/studies.db` | Shared with older studies unless overridden (the smoke test overrides it). |
| `--output` | `output/optimize/results.json` | |
| `--cache-dir` | `output/optimize/cache` | |
| `--data-root` | `/Volumes/Market-Data/info-bars` | Read-only. |
| `--search-space` | none | Built-in 12-dimension space. |

The floors default to `2` and `0`, so a run that omits them has effectively no trade-count requirement; always pass `--min-total-trades 105 --min-trades-per-fold 11` for the approved provisional floors.

## Notes

- Scratch outputs: point `--study-db/--cache-dir/--output` outside the repo for experiments.
- Tuning does not create edge: treat the fold score as optimistic and in-sample; the only out-of-sample evidence is 2024+ data via `validate_oos.py`.

## Dataset policy and workflow

Default split (`CLAUDE.md`): 2019-2023 development, hyperparameter optimization and cross-validation; 2024-2025 out-of-sample validation only; 2026 final untouched test set. The optimizer warns (does not fail) if `--end-date` is after 2023-12-31; the analysis scripts below refuse out-of-policy windows.

1. **Search** on 2019-2023 (`optimize_orderflow.py`), passing the floors explicitly. Do not pass `--evaluate-holdout` until a candidate has a feasible trial and a positive day-weighted Sharpe.
2. **Diagnose** a stored trial (`diagnose_trades.py --study-name ... --trial N --floors --stress`): replays only the cached development window, with settings taken from the study's stored identity, and stops if the replay digest does not reproduce the study's. Reports gross vs net expectancy, cost per trade in R, trade-return variability, cost and one-bar-later entry stress, and floors derived from the measured numbers.
3. **Selection stability** (`walk_forward_selection.py`): expanding-window walk-forward over the stored trials, no replay. It measures how stable the winning choice is across folds; the candidate pool was proposed with knowledge of all folds, so it is not an out-of-sample estimate.
4. **Compare** bar types (`compare_bar_types.py --min-trades-per-fold ... --min-total-trades ...`): eligibility is recomputed from each stored fold.
5. **Validate** (`validate_oos.py --candidates study[:trial],...`) once, on 2024-2025.
   - **Preflight, before any validation data is read:** every candidate is resolved to a concrete trial, its stored instrument defaults and library versions are checked against the installed code, and its development-window replay digest must reproduce. The stored **strategy fingerprint must equal the current one**: replay equality on the development inputs cannot prove equivalence on validation inputs, so editing strategy sources after tuning (anything hashed into `STRATEGY_FP`) requires re-tuning. Older fingerprints are accepted only through the reviewed, empty-by-default `COMPATIBLE_LEGACY_STRATEGY_FPS` mapping in `validate_oos.py`, and the digest must verify either way. Any failure aborts with nothing exposed and no manifest written.
   - **Frozen campaign:** the resolved candidates are written to `output/validate/campaign-<id>.json` before the first read; each result is saved as soon as it exists. Candidates are matched by (study, trial, parameter fingerprint) in a canonical order, so the command-line order never matters and duplicates are refused. Re-running the identical command resumes the pending candidates and never re-reads finished ones. Admission (exposure check plus manifest creation) runs under a directory-wide lock, and a campaign executes under an exclusive run lock, so concurrent invocations cannot both be admitted as first and the same candidate cannot be evaluated twice. A different candidate set after exposure is refused (`--acknowledge-prior-exposure` records the prior campaigns and proceeds). The Bonferroni count is the frozen campaign size.
   - **Bar construction** must match the development study (dataset source, calibration hyperparameters, creation time, git SHA, data-processing fingerprint); otherwise the run is refused before reading bars, and the evaluated dataset provenance is stored in each result. Time-bar studies need `--trades-root` (default: the optimizer's, honouring `ORDERFLOW_TRADES_ROOT`).
   - **Confirmation:** a candidate is confirmed only if its account never went bankrupt (missing solvency evidence also fails), its net expectancy is positive and its lower bound is above zero (Bonferroni over the candidates, standard error inflated for positive serial correlation). If none is confirmed the run stops. The independence assumption on trades is unverified. Deflated Sharpe is attached as a diagnostic.
6. **Final test** on 2026 only once, when explicitly decided. No script here reads 2026.

## Phase 4: opt-in trade signal research

Phase 4 uses the existing 2019-09-10 through 2023-12-31 information bars and an
exact, memory-mapped trade feature cache. Defaults remain bar-only. Minute bars,
post-2023 dates, warm starts, multiple workers within a study, and legacy holdout
selection are refused in Phase 4. Trade flags, a trade cache, frozen parameters,
a campaign manifest, a holdout request, or destinations under
`output/optimize/phase4/` imply Phase 4 controls.

All writable paths must resolve inside this checkout's
`output/optimize/phase4/`, `output/optimize/trade_cache/`, `output/phase4/`, or
`output/duckdb_tmp/`. Symlinks cannot grant writes into protected paths or an
external checkout. Existing study databases are read-only sources.

The build approval permits bounded verification only. For example (end exclusive):

```bash
python scripts/trade_feature_cache.py build --start 2021-05-19 --end 2021-05-20
python scripts/trade_feature_cache.py verify --against tests/fixtures/phase4_reference_digests.json
python -m pytest -q -p no:cacheprovider -m real_data tests/test_phase4_real.py
python scripts/phase4_benchmark.py --days 2019-09-11,2021-05-19,2023-10-16 --out output/phase4/benchmark
```

Run from this checkout with the project virtual environment. Bounded builds pin
`source_state` before and after replay and produce `bounded_verification` caches;
the optimizer accepts only clean-tree, full-history `production` caches. Bounded
builds default to `output/phase4/bounded_cache/` and refuse the production
`output/optimize/trade_cache/` root. Production builds refuse the bounded root.
Cache discovery counts only production manifests for the registered dataset.
`verify --against ...` selects certified references by start/end window and
refuses an unknown window; `--start` and `--end` can select a window to verify. Builds
longer than four days require explicit `--full-history --owner-approved TOKEN`,
with no explicit dates. The selector must identify a verified host approval
record; arbitrary strings are refused. That command and all experiments require
later owner approval. The development policy excludes post-2023 manifest entries before raw
file hashing or DuckDB queries, including the terminal full-stream hash sweep.
Legacy unbounded replay behavior is unchanged.

The benchmark requires the protected baseline at
`output/phase4/build/protected-before.sha256` and full-development bar-cache JSON
manifests at `output/phase4/inputs/bar-cache-manifests/` (or
`--bar-cache-manifests DIR`). It authenticates their bar counts against the paired
dataset, uses measured child-process RSS and mapped-page residency, and records
formulas, inputs, uncertainty, and resource gates in `benchmark.json` and
`REPORT.md`. It creates no Optuna study or trial.

For later campaigns, `phase4_campaign.py register` seals the campaign JSON and
writes a draft host driver in `output/phase4/run_scripts/`. `driver` prints a dry
run unless `--execute --stage A|B|C` is specified. Execution requires a separate
owner approval for the selected stage, an
independently inspected driver hash, the preflight, and gates U/FH1–FH4/R. Keep
one worker per study and one study process at a time. B0 uses 200
cold trials per bar type; A2/E2 use 100 feature-only trials on verified frozen B0
parameters; P2 writes one content-addressed candidate and creates no study.

`--freeze-params FILE` expects `study_db`, `study`, `trial`, `budget`, `params`,
and `replay_digest`. Stage B generates and verifies both frozen files after its
two B0 studies finish. The source must be its study's best feasible trial at the exact budget;
the optimizer verifies its stored search replay digest before using its values.

`phase4_report.py` reads stored studies and immutable database snapshots using
SQLite `mode=ro`. It reports CSCV, campaign multiplicity and deflated Sharpe,
joint UTC-day gross-R bootstrap, gross/net decomposition, the cost grid, funding
exposure, frequency multiples, boundaries and parameter neighbourhoods. Pass
`--study-name NAME --baseline-study B0_NAME`, or `--candidate SHA256` for P2,
plus `--study-db`, `--campaign-manifest`, and the appropriate `--trade-cache`.
P2 reports PBO and top-three as `not_applicable` and shows B0's selection risk.

The separate `--holdout-look STUDY:TRIAL` or
`--holdout-look candidate:SHA256` command requires a G2 report. Before replay it
persists a locked reservation. Repeating the identical request returns the stored
result; an interrupted request may finish that candidate; a different candidate
for an exposed arm is refused. The pseudo-holdout remains conditional on
full-history bar calibration. 2024–2025 dataset construction and OOS evaluation
remain outside Phase 4. `phase4_d6_probe.py` is an owner-gated read-only evidence
probe; it never changes fingerprint compatibility.

The frequency gate uses the largest per-signal-type multiple, so other detectors
cannot dilute an absorption increase. Reports include both per-type and total
multiples. A new signal type with no baseline count fails the frequency gate.

Campaign registration records three legacy database paths and SHA256 values
(`register --snapshots FILE`, a JSON list of `{path, sha256}` entries), plus the
Phase 4 database path without a prospective hash. Reporting first refuses any
missing, duplicate, RUNNING/WAITING, or off-budget campaign study. It then uses
SQLite's backup API with a `mode=ro` source to create a content-addressed Phase 4
snapshot under `output/phase4/reports/snapshots/`. The report records that hash
alongside the three legacy hashes and reads the resulting immutable union.
Frozen B0 sources must belong to the attached campaign, use that campaign's B0
budget, and resolve within this checkout's `output/optimize/phase4/`. A second
study for the same arm/campaign is refused; implied activation is a study attr,
not part of its hashed identity.

Host approvals are read only from `~/.orderflow-approvals/phase4/`, resolving home
through `pwd.getpwuid(os.getuid()).pw_dir`; `$HOME` is ignored. Approval directories,
records and gates must belong to the current uid and must not be group/world-writable.
Only the host
may write these records, after explicit owner approval in chat. This build never
creates real approvals. `--owner-approved SHA256` selects `SHA256.json`; the
file must contain canonical sorted compact JSON bytes plus a newline, and its
name must match the SHA256 of those bytes. Required fields are:

```json
{
  "scope": "full_history_tfc or campaign",
  "plan_sha256": "frozen plan SHA256",
  "campaign_sha256": "campaign scope only",
  "command_argv": ["/absolute/python", "scripts/command.py", "arguments"],
  "base_commit": "current HEAD",
  "source_state": {"base": "HEAD commit", "sha256": "source hash", "dirty": false},
  "created_utc": "2026-11-05T00:00:00Z"
}
```

The full-history scope omits `campaign_sha256`. The command is compared with
Python's original argv; only `--owner-approved SHA256` (or its `=` spelling) is
removed, avoiding a self-referential record hash. Plan, campaign, base and source
hashes must match the running checkout. A campaign additionally needs the
host-owned `gates-<campaign_sha256>.json` in the same directory, containing
`plan_sha256`, `campaign_sha256`, `base_commit`, `source_state`, `created_utc`,
and flat `U`, `FH1`, `FH2`, `FH3`, `FH4`, `R` fields equal to `pass`.
`output/phase4/gates.json` cannot authorize execution. The driver requires a
matching host record for each child command; it never mints child approvals.
Verified approval/gate hashes are recorded in manifests, study attrs and logs.
Each actual admission writes an audit line to `<sha256>.consumed` in that directory.
A full-history build consumes its record exclusively before replay; the same record
cannot authorize a second build, including after an interrupted attempt. Campaign
admissions append one line per use. Tests patch the OS home resolver and never
create real approval records. Any process running as this user can write records:
these are consent/audit artefacts, not a security boundary against the same user.
D6 execution also requires a host campaign-scoped command record and
`--campaign-manifest`; it remains code-only in this build.

Production caches are discoverable only after `COMPLETE.json` is written last,
binding every lane manifest hash. A partial multi-type publish cannot attach or
verify. Campaign registration stages both files, replaces the seal first and the
manifest last, and refuses a half-sealed campaign after an interrupted publish.

Daily-return payloads record `blown`. On bankruptcy the actual daily loss is
clipped at -1.0, with zero returns afterwards; these finite bad candidates remain
in CSCV. Calendar mismatches are checked independently from non-finite data.

The benchmark separately times cache-open hashing and digest verification.
Hash time scales by cache bytes and digest time by full-history bars. Each trade
study includes one open per process, amortized over its trial budget; P2 includes
one open for its single evaluation. Stage A opens once per bar type for all six
modes and checks protected hashes before/after that group. Campaign totals and
speed gates include these open costs; B0 has no trade-cache open.

The benchmark fits build wall as fixed plus per-processed-trade cost by ordinary
least squares, shows residuals and the Phase 3 8063–8804 s replay anchor, and
retains the conservative whole-run upper bound. Disk is fitted from separately
measured fixed metadata, per-bar arrays and finalised-day arrays, since the three
one-day windows cannot identify a joint fixed/day coefficient. Fitted and
conservative resource gates are reported separately without changing thresholds.

Production cache build and verification admission uses this order: validate the
command-bound approval without consuming it, run the production preflight, then
consume approval and begin work. Preflight checks the mount, registered paired
root, destination filesystem free space, conflicts, owner protected baseline,
source pin and total-memory admission. Required free space is
`max(100 GiB, 2 * estimated cache bytes)`; production uses the 40 GiB hard artifact
ceiling as its conservative estimate. Process enumeration errors refuse production;
bounded verification may report `unavailable`. Preflight values are saved in the
build report and `cache-build.json`. Protected and source states are checked again
before publishing the completion marker. A failed preflight leaves approval and
the destination untouched; an interruption after admission consumes that approval.

Production `verify` and cache attachment read only the six lanes listed in
`COMPLETE.json`, verify every listed manifest hash, and report unlisted orphan
directories without opening them. A retry after an interrupted publish needs a
new command-bound approval and may use the incomplete destination. A destination
with an existing completion marker is refused; the owner must explicitly move or
delete it. Successful retries retain orphan names for auditing.

Phase 4 study resume holds both exclusive study and worker locks while failing
orphan RUNNING/WAITING rows and running the sequential trial loop. Each orphan gets
`phase4_reconciled` (original state, UTC time, reason and known worker pid/host);
`phase4_reconciliation_audit` appends study-level records and repairs interrupted
audit appends. COMPLETE records are never changed. FAIL trials do not consume the
approved COMPLETE budget. Results, reports and holdout admission records include
FAIL/reconciled counts; admission still requires the exact COMPLETE budget and no
RUNNING/WAITING rows. Legacy studies retain their original resume behavior.

On macOS Phase 4 measures RSS/`ru_maxrss` and private `phys_footprint`, including
the kernel lifetime peak from `proc_pid_rusage(RUSAGE_INFO_V4)`. Child monitoring
also retains `wait4` peak RSS. Non-macOS uses sampled psutil USS. If private
measurement is unavailable, Phase 4 admission fails closed. Every Phase 4
campaign/study process must satisfy **RSS <= 4 GiB and private memory <= 4 GiB**,
including processes mapping a TFC lane. Mapped bytes remain a separate diagnostic;
they grant no RSS credit. The driver's external `wait4` peak check is also strict.
This supersedes the earlier mapped-cache RSS allowance (owner decision 2026-11-06).

Before each campaign child and each study admission in the child, planned private budgets plus
unique mapped files must be <= 24 GiB, and available memory must be >= 8 GiB.
macOS availability is exactly vm_stat free + inactive + speculative pages.
The existing driver launches one study process at a time; Stage A and verification
hold one lane at a time. Dry-run reports its serial plan and whether production
caches are available. Benchmark also reports the hypothetical six-study plan;
that plan is refused if six private budgets plus cache mappings exceed 24 GiB.
This does not change trial budgets or permit extra concurrency.

Benchmark records per-child private/RSS/mapped bytes, memory admission and both
resource bounds. It retains the per-observed-sample <= 3x speed gate and all
previous speed accounting. Release-readiness evidence is provisioned read-only
under `output/phase4/inputs/release-readiness/`; historical bar-only trial durations,
the 8,063 s full replay anchor and the per-level A2 model are diagnostic additions.
The supplied detector prescreen experiment is never executed or used in that model.

### Campaign admission after fix round 4

Phase 4 runs one study process at a time, with `--n-jobs 1`. Campaign manifests
may specify `concurrency: 1` and `n_jobs: 1`; any other value, including an arm
requesting parallelism, is refused. Driver dry-runs report `concurrency: 1`.
An exclusive checkout-wide execution lock serializes optimizer and Stage A
processes; a competing launch is refused. The benchmark's historical six-worker
projection remains evidence, not permission to launch six workers.

Host gate labels alone cannot admit a campaign. In addition to the fields above,
`gates-<campaign_sha256>.json` must bind these exact artifacts by SHA256:

| Gate field | Artifact relative to this checkout |
| --- | --- |
| `benchmark_sha256` | `output/phase4/benchmark/benchmark.json` |
| `complete_sha256` | `output/optimize/trade_cache/20261103T011635Z-d3a7b9603b37/COMPLETE.json` |
| `build_report_sha256` | `output/phase4/cache-build.json` |
| `verify_report_sha256` | `output/phase4/fullhistory/verify.stdout` |

Admission checks every `resource_gates` entry, including conservative-bound
entries, and verifies all six marker-listed lane manifests against the build
report, certified reference trade/replay digests, and off/all verification report.
A missing, changed, incomplete or failing artifact refuses the driver, Stage A
and optimizer children before any study database or cache arrays are opened.

Only the host may waive a failed benchmark gate. The gates record lists waiver
hashes in `waivers: ["<sha256>"]`. Each selects
`~/.orderflow-approvals/phase4/waiver-<sha256>.json`, with the same ownership,
permissions, symlink and canonical content-addressing rules as approvals:

```json
{
  "gate_path": "extrapolation.arms_under_3x",
  "benchmark_sha256": "exact benchmark SHA256",
  "plan_sha256": "frozen plan SHA256",
  "campaign_sha256": "sealed campaign SHA256",
  "created_utc": "2026-11-06T00:00:00Z",
  "replacement_gate": {
    "name": "host-approved replacement measurement",
    "threshold_spec": {"metric": ["trial_seconds"], "operator": "<=", "value": 60},
    "measured_artifact_path": "/absolute/path/to/measurement.json",
    "measured_artifact_sha256": "measurement SHA256"
  }
}
```

`metric` is a nonempty list of JSON object keys; the measured value and threshold
must be finite numbers. Supported comparisons are `<=`, `<`, `>=`, `>`, `==`.
Waivers identify a full gate path, omitting the `resource_gates` container: for
example `extrapolation.conservative_upper_bound.build_under_4h`. If the same gate
name fails in multiple blocks, each path requires its own waiver. Legacy name-only,
unknown, duplicate, stale or mismatched waivers, or failing replacement
measurements, refuse admission. Code never creates waiver records. Gate thresholds
and the benchmark artifact remain unchanged.

Each campaign study arm may optionally contain positive finite
`max_trial_seconds` and `max_study_seconds`. Omission leaves runtime enforcement
disabled. Every Phase 4 trial records `phase4_wall_seconds`; study elapsed
optimization time accumulates across resumes from monotonic loop measurements.
A metadata heartbeat thread checkpoints `phase4_heartbeat_utc` and measured loop
time at trial start, every 30 seconds of work, and trial exit. Failed checkpoints
are retried on each later pulse. `phase4_heartbeat_health` records consecutive and
total failures, the last error, and the longest gap between successful checkpoints.
Transient failures that recover retain the trial as COMPLETE. The final checkpoint
has three bounded attempts; exhaustion raises and stops the loop before any next
trial, without writing a final `phase4_wall_seconds`.

An independent watchdog uses a 90-second stale bound (three heartbeat intervals).
It exits non-zero, after logging to stderr, before that bound expires if successful
checkpoints cease, even if the checkpoint thread is blocked. The active trial stays
RUNNING. When measured trial wall time exceeds `max_trial_seconds`, it persists a
live `phase4_runtime_stop` with trial, measurement, limit and UTC time, then exits.
A fatal path prevents Optuna from finishing the active row during stop persistence.
Stop persistence has a bounded wait so storage trouble cannot disable termination.

On admission, every non-COMPLETE trial with timing but no final wall time is charged
**last heartbeat elapsed + the stale bound**, exactly once. This includes a FAIL
caused by final-checkpoint exhaustion. `phase4_reconciled` records
`accounting: "conservative_upper_bound"`, the charged seconds, bound and cumulative
runtime boundary; the study audit and persisted runtime are repaired idempotently.
The existing checkpoint credit is replaced by the bound, rather than charged twice.
Resume downtime is excluded. This intentionally over-counts unfinished work by up
to the stale bound. A legacy interrupted row without any timing record has no
measured work to charge; its start-to-reconciliation duration is never used.

`max_study_seconds` remains checked between trials; its worst overshoot is one
live trial ceiling when configured. Post-trial checks remain active. Runtime
stops block resume, candidate, holdout and report admission until the existing
host-written runtime decision clears them. `phase4_runtime_policy` stores the
interval, stale bound and accounting rule in the study; `phase4_report` emits them
with reconciliation and preflight diagnostics.

Every heartbeat samples RSS and private memory, recording `phase4_memory` peaks
and the private-memory metric. RSS > 4 GiB, private > 4 GiB, or unavailable private
measurement persists a live memory stop and terminates the process. Private memory
is required in addition to RSS; clean mapped pages do not relax either bound.

The driver and child both run the same system-memory preflight before a study;
the child does so before study writes or bar/trade-cache opens. Each accepted
study admission appends its report and UTC time to `phase4_memory_preflight`.
The campaign log includes every command's preflight, including `status: refused`
when a failed preflight prevents child launch. Execution is **sequential only**,
with `--n-jobs 1` and one study process at a time, enforced by the shared execution
lock. This supersedes the plan's <= 6 concurrent processes (owner decision
2026-11-06). Historical benchmark fields are preserved; the benchmark was not
rerun or rewritten for these conditions.

`owner_approval_sha256` and `gates_sha256` retain their first values. Every
subsequent study admission appends `{approval_sha256, gates_sha256, utc}` to
`phase4_approvals`. Phase 4 sampler seeds count COMPLETE trials, excluding failed
and reconciled attempts, so interrupted resumes reproduce the COMPLETE proposal
sequence. Legacy sampler behavior is unchanged.

Known cache-builder limitations, deferred to preserve the certified builder
fingerprint: P4-RV-02's projected disk/wall early abort is not implemented;
P4-RV-03's `build --bar-type` selector is currently ignored and build captures all
six lanes. The build RSS limit is checked during capture, not during final
write/hash; the observed transient peak was 3.83 GiB against the 4 GiB limit.


### Fix round 4b: registering runtime limits and owner recovery

Register optional limits before sealing the campaign. `ARM` is the full registered
key `SYMBOL:BAR_TYPE:MODE`; only study arms accept limits (P2 is a deterministic
candidate). Both flags are repeatable, require finite positive seconds, and reject
duplicate/unknown arms. For example, registration arguments may include:

```text
--max-trial-seconds BTCUSDT:fixed_tick_imbalance:off=120
--max-study-seconds BTCUSDT:fixed_tick_imbalance:off=24000
```

The sealed manifest and driver dry-run commands retain these limits; optimizer
arguments must match the registered arm. An existing manifest cannot be silently
registered with different limits. Omitted limits remain disabled.

A stopped study can be cleared only by a host-written canonical, content-addressed
`~/.orderflow-approvals/phase4/runtime-decision-<sha256>.json` record, with the same
ownership, mode and symlink checks as other approval records:

```json
{
  "plan_sha256": "frozen plan SHA256",
  "study_identity_sha256": "optimize_orderflow.fingerprint(study identity)",
  "stop_sha256": "optimize_orderflow.fingerprint(exact phase4_runtime_stop)",
  "decision": "clear_and_restart_budget",
  "created_utc": "2026-11-06T00:00:00Z"
}
```

The two identity hashes use the existing `fingerprint()` JSON serialization;
the filename hashes the record's canonical bytes. The code never creates a
decision record. To apply a host decision:

```bash
python scripts/phase4_campaign.py clear-runtime-stop \
  --study-db output/optimize/phase4/studies.db --study-name STUDY \
  --runtime-decision SHA256
```

The command requires an existing stopped Phase 4 study with no pending trials and
holds the exclusive execution/study/worker locks. It verifies the record before
opening writable study storage, appends the full old stop, decision hash, decision,
UTC timestamp and reset boundary to `phase4_runtime_decisions`, then clears the
stop to JSON null. A partial application can be repaired with the identical record.

`clear_and_restart_budget` explicitly acknowledges past overruns and starts a new
runtime-limit window; it does not erase cumulative `phase4_runtime_seconds`,
trials, limits or the audit. Subsequent overruns stop again. Missing, stale,
tampered or mismatched decisions refuse. Campaign approval and benchmark gates
still apply to every later run; this recovery command cannot waive them.


### Fix round 5b: separate stage approvals and checkpoints

A driver invocation executes exactly one stage and exits at its checkpoint.
Preview the entire grouped plan, or just one stage, without execution:

```bash
python scripts/phase4_campaign.py driver --manifest output/phase4/campaign.json
python scripts/phase4_campaign.py driver --manifest output/phase4/campaign.json --stage B
```

| Stage | Ordered commands | Preconditions |
| --- | --- | --- |
| A | One `ablation` child: 36 fixed-default replays across six bar types | No valid A checkpoint already exists |
| B | Two B0 studies, then two registered B0 freeze children | Valid A checkpoint for this campaign |
| C | Six feature studies, then two P2 deterministic optimizer commands | Valid B checkpoint and both frozen B0 sources re-verified before any child starts |

P2 uses the existing optimizer's deterministic candidate path; it writes immutable
candidate artifacts and creates no P2 Optuna study. The registered freeze children
select B0 by campaign/bar type, enforce the exact COMPLETE budget with no pending
trials/runtime stop, write frozen params and re-verify the stored search replay.
Stage C checks the checkpoint chain, all frozen-file hashes, live completed B0
specs/params/budgets and both replay digests before launching any C child.

After reviewing a stage's dry-run, the host writes a fresh argv-bound campaign
approval for that driver command and the existing per-child approval records.
For example, the stage-A driver record authorizes the exact invocation:

```bash
python scripts/phase4_campaign.py driver --manifest output/phase4/campaign.json \
  --execute --stage A --owner-approved SHA256 \
  --driver output/phase4/run_scripts/driver.py --inspected-driver DRIVER_SHA256
```

Review A's checkpoint before requesting the independent B approval, and B's
checkpoint before C. The stage flag is part of the approved driver argv; A's
approval cannot authorize B or C. Missing child approval is an error, rather than
the mechanism that separates stages. Execution remains sequential with one study
process and `--n-jobs 1`; all prior gates and strict RSS/private preflights remain.

Successful checkpoints are atomically published at
`output/phase4/checkpoints/<campaign_sha256>/stage-A.json` (similarly B/C).
They bind campaign, stage, parent/child approval hashes, gates, source state,
protected hashes before/after, every command's status/preflight/wall/RSS/private
measurement, output artifacts with SHA256, study counts and any runtime stops.
Every checkpoint states `next_stage_requires_owner_approval: true`. The campaign
log seals its SHA256, and a content-addressed copy retains the record. Per-study
summary artifacts are immutable; the live campaign database can continue growing
in C without invalidating B's checkpoint. P2 artifacts and frozen files are also
bound by hash.

Child failure (including any non-zero exit), a runtime or memory stop, gate
recheck failure, or source/protected change produces `stage-X.stopped.json`
instead of a completed checkpoint. Stopped attempts and log entries are retained.
A partial stage requires a new driver approval to retry; the existing exact-budget
logic resumes/skips completed studies without duplicates. A `stage_started` log
entry enforces fresh approval even if the driver was killed before it could write
a stopped record. A stage with a valid complete checkpoint refuses to run again.

The owner accepted the round-5 storage-outage termination priority with this
additional safeguard: reconciliation reconstructs a missing live stop before any
new trial if the conservatively charged wall time exceeds the arm's trial ceiling,
or the last durable heartbeat health contains a memory breach. It persists
`phase4_runtime_stop` with `live: false` and `reconciled: true`, blocking admission
until the same host-written runtime-decision mechanism clears it. Historical
reconciled trials acknowledged by an existing clear decision are not blocked
again. No approval/decision records are created by the code.

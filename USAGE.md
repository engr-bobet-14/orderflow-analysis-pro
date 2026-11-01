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

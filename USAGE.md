# Usage Guide

Step-by-step use of `scripts/optimize_orderflow.py`: resumable Optuna search of the rule-based orderflow strategy on information-driven bars (or 1-minute candles), scored on chronological sub-period folds (no model is fitted, so nothing is purged) and an optional holdout.
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
python scripts/optimize_orderflow.py ...same arguments... --n-trials 100            # tops up to 100 total
python scripts/optimize_orderflow.py ...same arguments... --n-jobs 4                # 4 spawn workers, one SQLite study
```

- Rerun with the same identity inputs to resume. `--n-trials` is the **total** to reach, not an increment; an already-complete study adds none. No trial is ever deleted.
- Identity = data, strategy source, search bounds, costs, CV geometry, holdout fraction, min trades, seed. Changing any of them starts a new study (new name in `results.json`). `--n-trials`, `--n-jobs` and output paths do not.
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
| `--n-trials` | `50` | Per study. |
| `--n-jobs` | `1` | |
| `--n-splits` | `5` | Chronological folds. |
| `--min-trades-per-fold` | `2` | **Not** the approved floor: pass `11` explicitly. |
| `--min-total-trades` | `0` | **Not** the approved floor: pass `105` explicitly. |
| `--embargo` | `0.02` | Scoring blackout at each fold start. |
| `--holdout-fraction` | `0.2` | Last 20% of bars never replayed during the search. |
| `--evaluate-holdout` | off | |
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

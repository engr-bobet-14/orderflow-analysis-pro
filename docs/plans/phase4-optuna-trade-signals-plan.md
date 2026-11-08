# Phase 4 — Optuna support for opt-in trade signals (A2, E2, P2): reconnaissance and design

Status: DESIGN ONLY (rev 3 — Codex findings P4-R1…R9 addressed). No code changes, Optuna runs, new studies, dataset changes or deployment are authorised by this plan. Implementation needs a separate owner approval.

Roles (Claudex loop): Claude = planner/coordinator; Codex = plan reviewer and, after approval, sole builder; Claude = final inspector. Rounds ≤ 3, MAX_FIX_ROUNDS 2, MAX_INSPECTION_ROUNDS 2.

## 0. Phase 3 verification (goal 1) — done during recon, 2026-11-05

| Check | Result | Source |
|---|---|---|
| master == origin/master | `390e29c` (merge of PR #13; commits `552e398`, `fef28c1`) | `git rev-parse HEAD origin/master` |
| Full test suite on master | **1107 passed**, 0 failed (PR reported 1106 + 1 skipped; the data-dependent skip now runs because `/Volumes/Market-Data` is mounted) | `pytest -q` (137.8 s) |
| Defaults off | `TradeFlowConfig().signals.any_enabled is False` (`absorption_levels=False`, `exhaustion_contrarian="off"`, `framing_profile_source="bar_vwap"`) | `orderflow_system/config/settings.py:187-233` |
| Compatibility list | `COMPATIBLE_LEGACY_STRATEGY_FPS == frozenset()` (and asserted by `tests/test_validate_oos.py:446`) | `scripts/validate_oos.py:32` |
| Optimizer / OOS bar-only guards | `bar_only_pipeline` (`scripts/optimize_orderflow.py:612`), `frozen_strategy_problems` + `evaluate_candidate` (`scripts/validate_oos.py:44-91`) | code |
| Current fingerprints | `STRATEGY_FP d592fcf9…`, `DATA_PROC_FP f6ef1e93…` | import of `scripts/optimize_orderflow.py` |
| Worktree | clean except 6 pre-existing untracked `.archify/` paths (left untouched) | `git status` |
| Flags-off full-history | byte-identical to Phase 2 (`f68cc72`) — Phase 3 evidence | `output/phase3_detector_plan/phase3_fullhistory/compare.json` (63/63) |
| **Correction** | The Phase 2/3 full-history runs hashed the whole manifest-listed margin file `date=2024-01-01/…683eef76….parquet` (integrity sweep; no trade values used in any output). PR #13's "no 2024+ data read" is therefore imprecise: 2024 bytes were read for hashing only. Phase 4 adds a policy-bounded mode (§4.1) | `trade_stream.py:217-220`, `trade_alignment.py:287-288`, `trade_manifest.json` |

## 1. Goal and acceptance (observable)

Goal: let the existing Optuna optimizer evaluate the opt-in trade signals A2 (trade-level absorption), E2 (exhaustion contrarian, `bonus|required`) and P2 (trade-profile framing) on the 2019–2023 development data, through pre-registered baseline and ablation experiments, with overfitting controls strong enough that a "feature helps" conclusion would be credible — while keeping the bar-only optimizer, stored studies, certified datasets and 2024–2026 holdouts untouched.

Phase 4 is complete when (all evidence under `output/phase4/`, git-ignored):
1. The trade-feature cache (§4.1) reproduces, for all six bar types over the full development history: Phase 2/3 per-lane `trade_digest`, the flags-off `replay_digest`, and the Phase 3 all-on `replay_digest` + `raw_signal_digest` (gates FH1–FH3).
2. Optimizer with `--trade-signals off` produces the same `replay_digest` per trial as master for identical params (byte-identity test), and refuses every unsafe combination listed in §4.3.
3. The pre-registered experiment campaign (§10) has run to its gates, OR the owner has stopped it; every run/stop is in the campaign log.
4. The overfitting report (§9) exists for every Optuna study, with CSCV PBO, campaign-level deflated Sharpe, gross/net R decomposition, cost-stress grid, signal-frequency multiples and parameter-stability tables.
5. Protected set (§12) hashes identical before/after every run; no 2024+ file opened by any Phase 4 command (open-interception test + run report `excluded_by_policy`); existing study DBs unchanged.
6. Test suite ≥ 1107 + new tests, all passing; new tests proven red first.

Non-goals: OOS (2024–2025) evaluation of trade-signal candidates (blocked: no 2024–2025 bars exist, §2 A6); changing bar construction thresholds; enabling A2/E2/P2 by default; live feed/alerts; changing `COMPATIBLE_LEGACY_STRATEGY_FPS`; tuning `footprint_bucket`/`profile_bucket`; new indicators.

## 2. Assumptions ledger (verified during recon)

| # | Fact | Source |
|---|---|---|
| A1 | Optimizer replays `CachedBars` (v0 info-bars `20261026T135000Z`) through `InfoBarPipeline.process_bar(bar)` with **no trade input**; `bar_only_pipeline` refuses enabled signals | `scripts/optimize_orderflow.py:267-305, 378-388, 612-616` |
| A2 | `process_bar(bar, trade: TradeBarUpdate | None)`; enabled signals require `trade`, and `trade.fp_bucket_ticks == round(footprint_bucket*100)` | `orderflow_system/pipeline.py:148-173` |
| A3 | `TradeBarUpdate = (stats: TradeBarStats, footprint_levels: 3 arrays, finalised_days: list[(day, arr, arr)], fp_bucket_ticks)` | `orderflow_system/analytics/trade_accumulator.py:112-116` |
| A4 | Full-history trade replay of 4.42 B trades (all six lanes in one pass) takes 8 063–8 804 s replay wall (~2.3 h), peak RSS ≈ 2.0–2.06 GiB; 1 573 finalised UTC days | `output/phase3_detector_plan/phase3_fullhistory/*.json` |
| A5 | Bars per type: dollar_imb 1 998 532, dollar_run 2 464 981, tick_imb 2 339 617, tick_run 2 429 670, volume_imb 1 996 838, volume_run 2 464 960 — identical counts in v0 sidecars and paired lanes | sidecars, lane reports |
| A6 | v0 info-bars and the paired dataset both cover **2019-09-10 → 2023-12-31 only**. The raw trade archive extends to 2026-11-04. There are no 2024–2025 bars, so `validate_oos.py` cannot evaluate any candidate today | sidecar `data_range`; `ls futures_trades` |
| A7 | Per-trial replay with live trade decoding is infeasible (2.3 h × 300 trials). Detectors themselves are cheap; decoding dominates | A4 |
| A8 | Objective: two-tier day-weighted Sharpe over 5 contiguous equal-bar folds of the first 80 % of bars, 2 % warm-up blackout per fold, continuous-equity accounting, trades attributed by exit bar; `NopPruner`; `TPESampler(seed)` | `optimize_orderflow.py:307-603, 676-704` |
| A9 | Costs: fee 5 bps + slippage 1 bps per side (CLI defaults); risk 1 %/trade, min stop 5 bps, capital $5 000 | `optimize_orderflow.py:307-310, 1278-1279` |
| A10 | Prior result: best trials of the main studies have **no gross edge** (gross +0.07…+0.09 R, 0.6–1.6 se; costs 0.19–0.30 R/trade) | `scripts/search_spaces/FINDINGS.md` §1 |
| A11 | Phase 3 diagnostic: A2 at untuned defaults raises absorption signals 5–12× (e.g. dollar_imb 31 137 → 199 913); absorption is the only entry trigger | Phase 3 full-history reports |
| A12 | Stored studies: `output/optimize/studies.db` 4 studies / 913 COMPLETE; `main/studies.db` 2 / 400; `smoke/studies.db` 1 / 5. No `holdout::` attrs were listed | read-only sqlite |
| A13 | `deflated_gate` (RiskLabAI) is diagnostic only; RiskLabAI also ships `backtest.probability_of_backtest_overfitting` (CSCV PBO) — installed, unused | `optimize_orderflow.py:998`; `pkgutil` listing |
| A14 | Host: arm64, 14 cores, 48 GiB RAM; ≈475 GiB free | Phase 3 report `hardware`, driver log |
| A15 | `docs/` is git-ignored; this plan and its log are not committed | `.gitignore:26` |
| A16 | Paired `trade_manifest.json` lists 1 576 files; exactly one has 2024 timestamps (`date=2024-01-01`, min_time 1704067200006, 2 197 331 rows, 20.3 MB); full-run verification hashes all listed files | manifest; `trade_stream.py:217-220`; `trade_alignment.py:287-288` |
| A17 | `campaign_trials` aggregates one DB; `diagnose_trades.decompose` `se_r` is computed from net R | `optimize_orderflow.py:760-778`; `diagnose_trades.py:150-158` |
| A18 | RiskLabAI PBO with a single candidate column classifies every split as overfit (rank 1/2) | `probability_of_backtest_overfitting.py:69-77` (Codex) |

## 3. Decisions (recommendation = default; owner may override)

| ID | Decision | Recommendation | Why / cost of guessing wrong |
|---|---|---|---|
| P4-D1 | How trials get trade inputs | **Trade-feature cache (TFC)**: one full-history pass per dataset writes each bar's exact `TradeBarUpdate` to memory-mappable arrays; trials replay from it | Only feasible option (A7). Wrong choice = 700 h per study |
| P4-D2 | Bar source for trade arms | Keep the optimizer's v0 `CachedBars`; TFC stores `seq`, `timestamp_ms` per bar and is refused unless equal to the bar cache arrays | Same bars for baseline and arms; fail-closed alignment |
| P4-D3 | Experiment structure | Staged and pre-registered: Stage A fixed-default ablation (no Optuna) → Stage B bar-only baseline studies → Stage C feature-only studies on frozen B0 params (§10) | Fewer tuned dims per feature, clean attribution; joint tuning is optional and owner-gated |
| P4-D4 | Bar types for Optuna | `fixed_tick_imbalance`, `fixed_volume_imbalance` (same as prior main studies); Stage A covers all six | Bounds multiplicity; Stage A can justify a change |
| P4-D5 | Warm start | **Off** for every Phase 4 study (`--no-warm-start` enforced for trade modes) | Equal cold budgets across arms; warm seeds would favour the bar-only arm |
| P4-D6 | Pruning | Keep `NopPruner`; record `"pruner": "NopPruner"` and sampler in identity | One replay yields all folds; fold-wise pruning favours early regimes |
| P4-D7 | CV geometry | Keep existing chronological scheme unchanged; add per-trial daily-return storage for CSCV | Comparable with known objective; no new leakage surface |
| P4-D8 | Costs | Taker both sides: fee 5 bps, slippage 1 bps (optimisation); post-hoc stress grid fee {4,5} × slippage {0,1,2,3} + gross; funding not modelled (reported) | Binance USDT-M VIP0 taker 0.05 % (verify before Phase 5) |
| P4-D9 | Where OOS stands | Trade-signal OOS deferred to Phase 5; needs 2024–2025 v0 bars **and** paired dataset built with frozen 2019–2023 calibration (other repo; separate approval) | Policy: 2024–2025 used once, after freezing |
| P4-D10 | Study storage | New DB `output/optimize/phase4/studies.db` + `results.json` + bar cache `output/optimize/phase4/cache/`; `--phase4` write allowlist checked before any write (§4.3) | Protects existing studies (§12) |
| P4-D11 | D6 legacy fingerprints | Separate read-only evidence track (§11); **no list change**; recommend deferring the mapping decision until Phase 4 code is merged | Phase 4 changes `STRATEGY_FP` again; a mapping to `d592fcf9` would be stale on merge |

Owner decisions required before build: O1 (P4-D3/D4 arms and bar types), O2 (compute budget, set after benchmark gate FH4), O3 (whether to schedule the 2024–2025 dataset build — not part of Phase 4), O4 (D6 policy, after §11 evidence).

## 4. Architecture

### 4.1 Trade-feature cache (new: `scripts/trade_feature_cache.py`)
- **Build**: one run of the existing paired replay (`scripts/replay_trades.py` lane machinery, flags off) over the full development range, all six lanes in one pass, capturing for each bar the `TradeBarUpdate` the pipeline would receive. No new trade decoding or alignment logic: reuse `trade_stream` / `trade_alignment` / `TradeAccumulator` exactly.
- **Development-only reads (P4-R1)**: the paired `trade_manifest.json` lists one margin-day file with 2024 timestamps (`date=2024-01-01/…683eef76….parquet`, min_time 1704067200006, 20.3 MB). Today's full-run verification hashes every manifest file (`trade_stream.py:217-220`, `trade_alignment.py:287-288`) and so reads that file's bytes. The TFC build therefore runs in a **policy-bounded mode**: files whose `min_time ≥ 2024-01-01T00:00Z` (from the authenticated manifest) are excluded from both queries and hash sweeps; every file that can contribute an in-range row is still hashed; the trade-count check becomes "in-range rows == `hours.parquet` sum for in-range hours" (`hours` is authenticated by the root hash); the report lists excluded files as `excluded_by_policy` (not authenticated). Tests: (a) **synthetic full-stream regression (P4-IR-04)** — a fake paired dataset with ≤ 4 development days plus a manifest entry whose `min_time` is in 2024, run as a full stream (`start_ordinal == 1`, `end_ordinal is None`) so both the query selection and the terminal all-manifest hash sweep execute; an open-interception harness (patched `builtins.open`/`io.open`/`os.open`, hashing helper, DuckDB file list) proves the 2024 path is never opened or passed to DuckDB, and a mutation that removes the policy filter makes the test fail; (b) the bounded real check of §19.1 as complementary evidence. Default (`unbounded`) full-run behaviour stays unchanged for Phase 2/3 reproducibility runs, which must be launched explicitly.
- **Layout** per `(dataset_id, bar_type)` under `output/optimize/trade_cache/<dataset_id>/<bar_type>-<key>/`:
  - `seq.npy`, `timestamp_ms.npy` (int64, n_bars)
  - `stats.npy` (n_bars × 13 int64 — the `'<13q'` packing of `TradeBarStats`)
  - footprint CSR: `fp_offsets.npy` (int64, n_bars+1) and the three footprint arrays concatenated with their **original dtypes** (`fp_0.npy`, `fp_1.npy`, `fp_2.npy`)
  - finalised days: `day_bar_index.npy`, `day_id.npy`, `day_offsets.npy`, `day_0.npy`, `day_1.npy`
  - `manifest.json`: dataset_id, root_hash (must equal `paired_datasets.json`), bar_type, n_bars, first/last seq, `fp_bucket_ticks`, `pf_bucket_ticks`, lane `trade_digest` (recomputed during build and from the cache on load), sha256 of every array file, builder source fp (sha256 of `trade_feature_cache.py`, `trade_accumulator.py`, `trade_stream.py`, `trade_alignment.py`), git sha + dirty flag, `source_state` (below), `mode` (`bounded_verification` | `production`), created_utc. Written last (atomic staged publish; directory renamed into place).
  - Provenance (P4-IR-02): a **production** (full-history) build requires a clean tree. A **bounded verification** build (≤ 4 days) may run from an uncommitted tree: it records `source_state = sha256(git diff --binary BASE + sorted (path, sha256) of untracked non-ignored files)` with `BASE`, recomputes it after the run and fails if it changed. Optimizer studies refuse any cache whose `mode` is not `production`, so bounded caches can only feed tests, the B-FH gate and the benchmark.
- **Read** (`CachedTradeUpdates`): `np.load(mmap_mode="r")`, verify manifest + file hashes on open, verify `seq`/`timestamp_ms` equal the `CachedBars` arrays, yield `TradeBarUpdate` per bar index (views, no copies) with `fp_bucket_ticks` set. Any mismatch → `ValueError` before replay.
- Writes only under `output/`; refuses any path under `/Volumes/` and any protected path (§12).

### 4.2 Cached replay
- `run_backtest(bars, config, price_proximity_pct, bar_type, prefix_ends=(), trade_updates=None, trade_config=None)`.
  - `trade_updates is None` → **current code path byte-for-byte** (`bar_only_pipeline`), so bar-only studies and `validate_oos` stay unchanged.
  - otherwise build `InfoBarPipeline(config, bar_type, trade_config=trade_config)` and call `process_bar(bar, trade_updates[i])`. A raw-signal digest (Phase 3 `RawSignalDigest`) is recorded alongside `replay_digest` in trial attrs.
- `evaluate_fold`, `evaluate_holdout`, `diagnose_trades` replay paths accept the same optional trade source.

### 4.3 Optimizer integration (`scripts/optimize_orderflow.py`)
- New CLI: `--trade-signals MODE` (shared `parse_trade_signals` spelling; default `off`), `--trade-cache DIR`, `--freeze-params FILE` (Stage C: JSON of bar-dimension values, taken from a stored B0 trial by study/trial id and verified by its stored `replay_digest` before use).
- Search space: existing `SEARCH_SPACE_BOUNDS` keys unchanged; new trade dimensions in `TRADE_SEARCH_SPACE_BOUNDS` (prefix `ts_`, §6). Space per arm = active feature dims (+ bar dims unless frozen). Dimensions that are provably inert in an arm are removed from that arm (builder must show inertness with a test, e.g. a bar threshold the arm bypasses).
- Study identity adds: `trade_signals` (existing helper; full `TradeSignalConfig` incl. tuned values is per-trial, mode + fixed fields per study), `trade_cache` (manifest sha256 + builder fp + dataset_id/root_hash), `frozen_params` (values + source study/trial/digest) when used, `pruner`, `sampler`, `phase: "phase4"`.
- Refusals (each a red-first test): trade mode without `--trade-cache`; cache for another dataset/bar type/bucket; cache `seq`/time mismatch; `time_1m` in **any** Phase 4 mode run, flags off included, refused before trade-root discovery or cache construction (P4-IR-01: minute staging builds whole months with a next-day margin and would hand 2024-01-01 parquet paths to DuckDB for a 2023-12-31 end); warm start with trade mode; end date after 2023-12-31 (error for trade modes and for any `--phase4` run; warning kept for legacy bar-only use); `--evaluate-holdout` in Phase 4 runs (replaced by the single-look command below).
- **Phase 4 mode activation (P4-R7)**: Phase 4 controls are active when `--phase4` is given **or implied**; it is implied (and recorded) by any of `--trade-signals` ≠ `off`, `--trade-cache`, `--freeze-params`, `--holdout-look`, `--campaign-manifest`, or a `--study-db`/`--output`/`--cache-dir` that resolves under `output/optimize/phase4/`. There is no way to request a trade mode without the allowlist. Legacy (non-Phase-4) runs additionally pre-read the target study's identity with `sqlite3 mode=ro` before `open_study` and refuse any study tagged `phase: "phase4"`. Tests: trade mode without `--phase4` and with default `--study-db`/`--output` is refused before any open; protected-set hashes and directory snapshots unchanged.
- **Write allowlist (P4-R3)**: in Phase 4 mode (every campaign command, B0 included) every resolved write destination — `--study-db`, `--output`, `--cache-dir`, `--trade-cache`, lock files, temp/staging files, DuckDB spill dir, diagnostic and report outputs — must lie under `output/optimize/phase4/`, `output/optimize/trade_cache/`, `output/phase4/` or `output/duckdb_tmp/`, and must not resolve (after `realpath`, symlinks included) into the protected set (§12). Checked in `main()` **before any `mkdir`/open**, including before `build_cache`. Phase 4 bar caches live in `output/optimize/phase4/cache/`; `output/optimize/cache/` is read-only for Phase 4. Test: each destination pointed at `output/optimize/main/cache` (and via a symlink) is refused with no filesystem change (directory snapshot before/after).
- **Deterministic candidates (P4-IR-03)**: P2 (and any other configuration evaluated without its own study) is frozen as an immutable artifact `output/phase4/candidates/<sha256>.json` (content-addressed; refused if a file of that name exists with other bytes) binding: arm/mode, full effective `TradeSignalConfig`, merged params, TFC identity (manifest sha256, builder fp), search-segment `replay_digest` and raw-signal digest, and the source B0 study identity hash, trial number, params sha256 and stored `replay_digest`; creation verifies the source B0 study has COMPLETE == budget and no RUNNING/WAITING trials, and replays the candidate to confirm its digests. `--holdout-look candidate:<sha256>` admits it under the same reservation protocol, keyed by the artifact hash. Tests: P2 admission succeeds without creating any P2 study; repeat returns the stored result; a different candidate for the P2 arm is refused; tampered artifact refused.
- **Single look per frozen candidate (P4-R2)**: pseudo-holdout evaluation is a separate command `--holdout-look STUDY:TRIAL` (no dynamic best-trial choice). The campaign manifest pre-registers, per arm, the candidate rule ("best feasible trial of the completed study with exactly N trials"). Before any holdout bar is replayed, under an exclusive lock on `output/phase4/holdout/exposures.json`, a reservation is persisted with: study identity hash, trial number, merged-params sha256, stored search-segment `replay_digest`, study COMPLETE count == pre-registered budget. Afterwards the result is persisted against that reservation. Re-invocation: same candidate with result → returns the stored result, no replay; same candidate reserved without result (crash) → may complete the identical deterministic replay, logged `completed_after_interruption`; any other candidate for that arm, or a study whose trial count changed → refused. The legacy `evaluate_holdout` path is unchanged for bar-only legacy use and refused under `--phase4`.
- `bar_only_pipeline` stays the guard for all bar-only paths and for `validate_oos`.
- Per-trial attrs add `daily_returns_search` (compact float32 base64 or side file `trials/<study>/<n>.npy` + sha256 attr) for CSCV, and `signal_counts` per signal type.

### 4.4 Unchanged on purpose
`validate_oos.py` (stays bar-only; trade studies refused — existing test), `COMPATIBLE_LEGACY_STRATEGY_FPS`, bar construction, detector logic and defaults, `footprint_bucket=1.0`, `profile_bucket=10.0`, `DeltaEngine` (tick-rule delta separate from aggressor delta).

## 5. Chronological cross-validation (goal 4)
- Window: 2019-09-10 → 2023-12-31 (the full development set with data). Search segment = first 80 % of bars; 5 contiguous equal-bar folds; each fold replays from bar 0 (continuous account) and scores exits in `[fold_start + 2 % N, fold_end]`. Unchanged from A8.
- Pseudo-holdout = last 20 % of bars (≈ Feb–Dec 2023 depending on type) after a 2 % blackout; **one look per pre-registered candidate**, enforced by the `--holdout-look` reservation protocol (§4.3), never by dynamic best-trial selection.
- Purge is a no-op (rule-based, nothing fitted); leakage protection = causal replay + untouched tail + multiplicity-aware statistics. Holdout reported as **conditional**: bar thresholds were calibrated on full 2019–2023 history.
- Trade features are causal by construction: the bar's own footprint is complete at bar close; P2 uses only finalised UTC days (Phase 3 invariant). FH3 re-proves this from cache.
- Reported per study: fold calendar bounds, per-fold Sharpe/trades, worst fold, fold dispersion.

## 6. Parameter search spaces (goal 4)
Bar dims: existing `SEARCH_SPACE_BOUNDS` with the existing per-type overrides in `scripts/search_spaces/*.json` (B0 only; frozen in Stage C).

Trade dims (defaults at or near the centre; bounds are search hygiene, not edge claims):

| dim | maps to | bounds | scale | arm |
|---|---|---|---|---|
| `ts_level_zone_frac` | `level_zone_frac` (0.25) | 0.10–0.50 | linear | A2 |
| `ts_min_level_share` | `min_level_share` (0.05) | 0.02–0.20 | log | A2 |
| `ts_min_side_ratio` | `min_side_ratio` (0.65) | 0.55–0.90 | linear | A2 |
| `ts_min_reject_bps` | `min_reject_bps` (2.0) | 0.5–8.0 | log | A2 |
| `ts_max_through_bps` | `max_through_bps` (1.0) | 0.0–4.0 | linear | A2 |
| `ts_contrarian_ratio` | `contrarian_ratio` (2.5) | 1.5–5.0 | log | E2 |
| `ts_contrarian_zone_frac` | `contrarian_zone_frac` (0.30) | 0.10–0.50 | linear | E2 |
| `ts_contrarian_min_level_share` | `contrarian_min_level_share` (0.02) | 0.005–0.10 | log | E2 |

Modes are fixed per study, never searched. P2 has no threshold: the P2 arm is a single deterministic evaluation on frozen B0 params (no Optuna study). Bounds validated against `TradeSignalConfig.__post_init__` ranges (test). `search_space_audit.py` learns the `ts_` dims.

## 7. Pruning policy
`NopPruner` (explicit, in identity). No custom early abort: a trial that fails a floor is scored in the infeasible tier with its reason, never recorded as a normal value. Rationale: one replay yields all folds; intermediate pruning would rank on early regimes and complicate determinism; trial cost is bounded by the TFC (FH4).

## 8. Trading-cost assumptions
- Optimisation costs: taker fee 5 bps/side + slippage 1 bps/side (identity-bound; same for every arm).
- Post-hoc stress (top-3 trials of each study, digest-verified replays): fee {4, 5} × slippage {0, 1, 2, 3} bps and a zero-cost gross line; report net R and break-even slippage.
- Funding: not in the data; report share of trades held across 00/08/16 UTC and their notional; limitation.
- Stop fills use the ledger's existing fill model (unchanged); flagged as optimistic for fast moves; stress grid covers partially.

## 9. Overfitting controls
1. Pre-registration: campaign manifest (arms, bar types, seeds, budgets, floors, gates, costs) written and hashed before the first trial; later changes create a new campaign id and are logged.
2. Equal budgets: every Stage B study 200 trials, every Stage C study 100 trials, seed 42, cold. **Exact budget (P4-R8)**: in Phase 4 mode `--n-jobs` must be 1 (one worker per study: no completion-order overshoot, sequential seeding reproducible); parallelism is across studies (≤ 6 concurrent study processes, §15). The loop admits a new trial only while COMPLETE < budget; FAIL trials do not count and are reported. `--holdout-look` additionally refuses a study with any RUNNING/WAITING trial or COMPLETE ≠ budget. Test: budget reached exactly; `--n-jobs 2` refused in Phase 4 mode.
3. Trade floors from the existing per-trade-sd derivation: `--min-total-trades 105 --min-trades-per-fold 11` (as in the main studies).
4. CSCV probability of backtest overfitting (RiskLabAI) per study from stored per-trial daily returns, S = 16 blocks; diagnostic + gate input (G2).
5. Deflated Sharpe at campaign multiplicity (P4-R6): `campaign_trials` reads one DB today, so T4 adds a cross-database aggregator over a pre-registered list of read-only snapshots — `output/optimize/studies.db`, `output/optimize/main/studies.db`, `output/optimize/smoke/studies.db`, `output/optimize/phase4/studies.db` — each recorded with its sha256. COMPLETE trials are deduplicated by (study_name, trial number, params sha256, `replay_digest`) because `main/` holds copies of two `studies.db` study names; trial count and `raw_daily_sharpe` dispersion are computed over the deduplicated union; the report lists per-DB and union counts. Test: union > Phase 4 DB alone; duplicates counted once. Diagnostic only.
6. Gross/net decomposition in R (existing `diagnose_trades.decompose` fields), plus the pre-registered gross-edge statistics of §10 (gross R uses `gross_pnl / dollar_risk`; `decompose`'s `se_r` is a net-R statistic and is not used for gross gates).
7. Signal-frequency multiple vs B0 at the chosen params (Phase 3 lesson: A2 5–12× at defaults).
8. Parameter stability: top-10 dispersion, boundary audit, one-at-a-time neighbourhood replays of the best trial. **Domain-aware perturbation (P4-R9)**: step = ±10 % of the dimension's search-range width (additive, so zero-valued and bound-pinned values work); for `float_log` dims the step is ±10 % of the log-range width; `int` dims step by max(1, round(10 % × width)) and stay int; each neighbour is clipped to the search bounds ∩ the config validity domain (`TradeSignalConfig.__post_init__`, `load_space` rules); a neighbour that clips back to the base value is recorded `unavailable_at_bound`; every replay records the effective change. Tests (T4): bound-pinned float, zero value (`ts_max_through_bps=0`), `ts_min_side_ratio` at 0.55, int `lookback_bars` at both bounds.
9. Single-look pseudo-holdout; 2024–2026 untouched (policy guards stay; trade modes hard-fail after 2023-12-31).

## 10. Baseline and ablation experiments (goal 5) — pre-registered, none run in Phase 4 design

| Stage | Run | Bar types | Budget | Output |
|---|---|---|---|---|
| A | Fixed-default ablation from TFC: B0 (flags off), A2, E2-bonus, E2-required, P2, ALL (A2+E2-bonus+P2) at instrument defaults, full search segment | all 6 | 36 replays | signal counts, trades, gross/net R ± se, fold Sharpe; diagnostic only |
| B | B0 Optuna (bar dims, flags off, from v0 cache — no TFC needed) | tick_imb, volume_imb | 2 × 200 | baseline best params + report |
| C | Feature-only Optuna on frozen B0 best params: A2 (5 dims), E2-bonus (3), E2-required (3) | tick_imb, volume_imb | 6 × 100 | per-arm report |
| C | P2 on frozen B0 best (single evaluation) | tick_imb, volume_imb | 2 replays | report |
| C′ (optional, owner-gated) | ALL arm, or joint bar+trade tuning for an arm that passes G2 | as approved | as approved | report |

Gates:
- **G-A** (after Stage A): none blocking; publish counts and gross R (lesson: report frequency before enabling).
- **G-B** (after Stage B): if B0 best top-3 gross E_r ≤ 0, still run Stage C (the question is whether features add gross edge), but record "baseline has no edge".
- **Gross-edge statistics (pre-registered, P4-R5)**, computed on the search segment only:
  - Per trade: `gross_R = gross_pnl / dollar_risk` (trades with `dollar_risk ≤ 0` excluded and counted). Arm statistic = ratio estimator `Σ gross_R / n_trades` over trades whose **exit** falls in the segment.
  - Uncertainty: moving-block bootstrap over UTC calendar days of the search segment, block length 7 days, 2 000 resamples, seed 20261105. Each resample draws one day-block sequence and applies it **jointly** to the arm and to B0, recomputing both ratio estimators from the trades exiting on the drawn days; this preserves time dependence and arm/B0 co-movement without pairing trades (A2/E2 change entries and exits, so trades are not matched).
  - A resample where either side has 0 trades is invalid; if > 1 % are invalid the gate is `indeterminate` (= fail).
  - Candidate = the pre-registered single best feasible trial (not a top-3 mean). Top-3 are reported as robustness: all three must have point gross E_r > 0.
  - Insufficient sample: candidate below the floors (105 total, 11 per fold) → fail.
- **G2** (Optuna arm → pseudo-holdout look), all required: bootstrap 2.5th percentile of the arm's gross E_r > 0; bootstrap 2.5th percentile of (arm − B0 best) gross E_r > 0; all top-3 point gross E_r > 0; CSCV PBO < 0.5 for the arm's study; signal-frequency multiple ≤ 3× B0 best; floors met.
- **G2-P2** (deterministic P2 candidate, P4-R4): P2 has one candidate (frozen B0 best params + `framing_profile_source=trades`), so PBO and top-3 do not apply and are reported `not_applicable`; its selection risk is inherited from B0 (B0's PBO and campaign multiplicity are reported beside it). Required: both bootstrap conditions above; frequency multiple ≤ 3×; floors met. P2 gets its own exposure record and single look under the same reservation rule.
- **G3** (arm → recommend Phase 5 OOS): pseudo-holdout net E_r > 0, n_trades ≥ 11, not blown; deflated Sharpe reported. Recommendation only — OOS needs the 2024–2025 datasets (P4-D9) and owner approval.
- If no arm passes G2: Phase 4 concludes "no evidence that A2/E2/P2 add edge"; no further tuning (rule: stop tuning when gross expectancy ≤ 0).

## 11. D6 — historical strategy fingerprint compatibility (goal 3, separate, read-only)

Recon findings (recomputed `source_fingerprints()` at each first-parent master commit via `git show`; no checkout):

| fp | introduced by | studies |
|---|---|---|
| `0ab6e2dd…` | `66d2119` PR #8 (optuna scripts) | `studies.db` study 1 (53 trials, `s9ceb6b40`) |
| `f6820adb…` | `43f7c75` PR #9 (edge diagnosis: two-tier objective, continuous-equity accounting, narrower `price_proximity_pct`, new dims, aggregator change) | `studies.db` 2–4 (860 trials) + `main/studies.db` 1–2 (400 trials) |
| `6d4cf8fc…` | `0dc37ba` PR #12 (trade pipeline: additive `trade` argument, accumulator, feed changes; STRATEGY section unchanged) | none |
| `d592fcf9…` | `390e29c` PR #13 (opt-in A2/E2/P2; STRATEGY section only gains `bar_only_pipeline`) | none (current) |

Assessment:
- `0ab6e2dd` → current is **not** a compatibility candidate: the objective, accounting and search space changed in PR #9, so stored values are not comparable even where replay digests match.
- `f6820adb` → current: changes are PR #12/#13 code paths gated by `trade is not None` / `signals.any_enabled`, plus the wrapper. Evidence so far: 80/80 sampled + 6/6 earlier stored digests reproduced (development data only).
- No current use: validation needs 2024–2025 bars, which do not exist (A6).

Read-only evidence task (T8, no list change): (a) replay **every** COMPLETE trial of the five `f6820adb` studies under current code and compare stored `replay_digest` (≈1 260 replays; ≤ 6 read-only processes, §15 RSS limits); (b) static diff report of all fingerprinted files `43f7c75..HEAD`, classifying each hunk as flags-off-reachable or not; (c) memo for the owner with options: keep list empty and re-tune (default), add a reviewed `f6820adb` entry after Phase 4 merges (keyed to the then-current fp), or redesign the fingerprint (architecture change — not recommended now). The owner decides; this plan never edits the list.

## 12. Preservation
Protected set (hash before and after every build/run; driver fails closed):
- the 46 files of `output/phase2_trade_pipeline/run_scripts/protected_hash.sh` (v0 info-bars, `output/optimize/studies.db`, `output/optimize/results.json`, `paired_datasets.json`, dataset certificate, local and mounted paired dataset files);
- plus `output/optimize/main/**`, `output/optimize/smoke/**`, `output/certificates/**`, `output/diagnose/**`.
Rules: `/Volumes/Market-Data` read-only; no 2024+ file opened (dev-window guards + policy-bounded TFC build, §4.1); existing study DBs opened only `mode=ro`; `output/optimize/cache/` read-only; Phase 4 writes only `output/optimize/phase4/`, `output/optimize/trade_cache/`, `output/phase4/`, `output/duckdb_tmp/` (allowlist, §4.3).

## 13. Implementation tasks (Codex builds; TDD: each test fails before its fix, shown in the build report)

| Task | Content | Key tests (red first) |
|---|---|---|
| T0 | Baseline capture: protected hashes, test count, fp values, Phase 2/3 reference digests per lane copied into `tests/fixtures/phase4_reference_digests.json` (values only) | fixture loads; values equal source reports |
| T1 | `scripts/trade_feature_cache.py` writer + reader; policy-bounded trade stream (§4.1) | open-interception: no `min_time ≥ 2024` file opened by query or hash sweep, every contributing file hashed, in-range count check; synthetic round-trip (empty bar, zero levels, multi-day finalisation, int dtypes preserved); manifest/file-hash tamper → refuse; seq/time mismatch → refuse; dirty-tree **production** build refused, bounded verification build allowed with `source_state` pinned before/after (changed source → fail); optimizer refuses non-production caches; time_1m refused in every Phase 4 run (P4-IR-01) with a flags-off 2023-12-31 case asserting refusal before any trade-root access; writes outside `output/` or under `/Volumes` refused; staged publish crash leaves no visible cache |
| T2 | Cached replay in `run_backtest` / folds / holdout / `diagnose_trades` | `trade_updates=None` byte-identical digests vs master on fixture + real bars; cached vs live trade pipeline equal `replay_digest` and raw-signal digest on 2021-05-19, all six types, off and all-on |
| T3 | Optimizer CLI, spaces, identity, Phase 4 mode activation, refusals, write allowlist, exact budget, `--holdout-look` reservation (§4.3, §9), per-trial daily returns + signal counts | one test per refusal; every implied-activation trigger; legacy run refuses a `phase4`-tagged study via read-only pre-read; `--n-jobs 2` refused in Phase 4 mode and budget reached exactly; allowlist refusal for every write destination incl. `--cache-dir` and symlinks, with unchanged directory snapshot; holdout reservation: second candidate refused, changed trial count refused, crash-between-reserve-and-result completes same candidate only, concurrent attempts serialised (sleep-widened race + lock-no-op mutation); identity contains new fields; P2 arm has no Optuna study; inert-dim removal test; bounds within `TradeSignalConfig` validity; `--trade-signals off` identity/name unchanged except fp |
| T4 | Overfitting report `scripts/phase4_report.py`: CSCV PBO, cross-DB DSR multiplicity, gross-R block bootstrap (§10), cost stress grid, frequency multiples, stability, gross/net R | PBO on synthetic matrices (known 0/1 cases) and refusal of single-column input; cross-DB dedupe; bootstrap reproducible by seed, joint resampling (arm−B0 diff of identical inputs is exactly 0), invalid-resample rule; stress grid matches hand computation; legacy DBs opened `mode=ro` only, writes only under `output/phase4/` |
| T5 | Campaign manifest + host-written driver (`output/phase4/run_scripts/`), reviewed by Claude before use; preflight: mount, free disk ≥ 100 GiB, no conflicting process, protected hashes, source-state pin | driver dry-run; manifest hash refuses edits |
| T6 | `search_space_audit.py`, `walk_forward_selection.py` accept `ts_` dims | existing tests + new dims |
| T7 | Docs: `USAGE.md` Phase 4 section. (`scripts/CLAUDE.md` / root `CLAUDE.md` updates are proposals only — need owner approval.) | — |
| T8 | D6 read-only evidence (§11) | probe refuses write opens; digest comparison report |

## 14. Acceptance gates (in order; each blocks the next)
- **U** unit/integration: full suite green (≥ 1107 + new), red-first evidence, mutation checks for every refusal.
- **FH1** TFC build, full development history, six types: per-lane `trade_digest` == Phase 2/3 references; manifest hashes verified on reopen.
- **FH2** cached replay flags-off: per-lane `replay_digest` == Phase 3 off-run == optimizer `run_backtest` on v0 cache with default config.
- **FH3** cached replay all-on defaults: `replay_digest` and `raw_signal_digest` == Phase 3 all-on run 1.
- **FH4** benchmark: 3 trials per arm on `fixed_tick_imbalance`; per-trial wall, per-worker peak RSS (wait4), page-cache footprint; owner approves campaign budget (O2).
- **R** Codex review of plan (this loop) and Claude inspection of build; Codex re-inspection of fixes (labelled same-provider when applicable).
- Experiments (§10) run only after U, FH1–FH4, inspection and explicit owner authorisation.

## 15. Resource limits
- Peak RSS ≤ 4 GiB per process (wait4, same gate as Phase 2/3); ≤ 6 concurrent study processes, one worker each (P4-R8); sum of worker peaks ≤ 24 GiB.
- TFC disk ≤ 40 GiB total (measured in FH1; abort if exceeded); DuckDB spill bounded by existing settings (1 GB memory limit, `output/duckdb_tmp`), spill files 0 at end.
- Build wall ≤ 4 h; per-trial wall ≤ 3× a bar-only trial (FH4), else redesign before experiments.
- Free disk ≥ 100 GiB at preflight; `caffeinate`/keep-awake during long runs.

## 16. Rollback
- Code: all Phase 4 behaviour sits behind `--trade-signals` (default `off`) and the TFC path; bar-only optimizer and `validate_oos` stay on current code paths. Rollback = revert the Phase 4 merge commit.
- Data: Phase 4 artifacts are derived and rebuildable (`output/optimize/phase4/`, `output/optimize/trade_cache/`, `output/phase4/`); delete to roll back. Protected set is never written, so nothing to restore; hash comparison proves it.
- `STRATEGY_FP` will change on merge (expected); existing studies are already blocked for OOS by D6; no new incompatibility class.

## 17. Verification commands (builder and host run independently)
```bash
source .venv_orderflow/bin/activate
pytest -q
zsh output/phase2_trade_pipeline/run_scripts/protected_hash.sh /tmp/p4_before.sha256   # extended list in T5 driver
python scripts/trade_feature_cache.py build --dataset-id 20261103T011635Z-d3a7b9603b37 --bar-type all   # FH1
python scripts/trade_feature_cache.py verify --bar-type all --against tests/fixtures/phase4_reference_digests.json   # FH1-FH3
python scripts/optimize_orderflow.py --phase4 --bar-type fixed_tick_imbalance --trade-signals absorption_levels \
  --trade-cache output/optimize/trade_cache --cache-dir output/optimize/phase4/cache \
  --study-db output/optimize/phase4/studies.db --output output/optimize/phase4/results.json --no-warm-start --dry-run
```
Exact flag spellings may change in build; the gates do not.

## 18. Risks and limitations
- Frozen-B0 Stage C can miss interactions (a feature useful only at other bar params); C′ exists for that, with multiplicity counted.
- Pseudo-holdout is conditional on full-history bar calibration; OOS for trade arms not possible until 2024–2025 datasets exist.
- A2 at defaults floods signals; tuning may only "fix" frequency, not create edge — G2 requires gross-edge improvement, not score.
- CSCV with 5 contiguous folds of a single replay has dependent blocks; PBO is indicative.
- Funding costs and stop-fill optimism are unmodelled.
- TFC equivalence is proven on the development history only; any trade-pipeline code change invalidates the cache (builder fp in manifest + study identity).

## 19. Build-phase scope (owner approval 2026-11-05, rev 5)

Owner approved the build of rev 3 (sha256 `fef3d620…`) with defaults O1–O4 and these additional requirements. This section **overrides** earlier sections where they conflict; every superseded passage is listed below.

Owner requirements (verbatim intent):
1. Enforce the 2019–2023 data boundary **before opening files**, with a regression test proving no 2024+ file is accessed.
2. Preserve legacy bar-only optimizer behaviour and existing study databases.
3. Validate cache correctness against the certified Phase 2/3 digests.
4. Keep the Phase 4 study database and output writes isolated under the approved allowlist.
5. Implement the benchmark and resource-budget reporting, but **request owner approval before full-history cache generation or Optuna experiments**.
6. Do not build 2024–2025 datasets, change fingerprint compatibility, merge, or deploy.

### 19.1 What this build does
- Implements T0–T7 code and tests, and T8 as code only (the probe); TDD red-first evidence per task.
- Runs **bounded** real-data checks only (≤ 4 UTC days per run, development window):
  - **B-FH (bounded cache gate)**: build a TFC for 2021-05-19 (heaviest day, all six types) and verify against the certified Phase 3 final-code bounded reports in `output/phase3_detector_plan/trade_replay/` (main checkout, read-only): flags-off report `…T092344357413Z.json` (per-lane `trade_digest`, `replay_digest`, `bars`) and all-on report `…T092420976414Z.json` (A2 + E2-bonus + P2: `trade_digest`, `replay_digest`, `raw_signal_digest`, signal counts). Reference values are copied into `tests/fixtures/phase4_reference_digests.json` with the source file sha256.
  - Live-pipeline vs cached-replay equality (same code) on 2019-09-11 and 2023-10-16, all six types, flags off and all-on.
  - **Boundary regression (owner req. 1)**: primary proof is the synthetic full-stream test of §4.1 (P4-IR-04), which exercises the terminal hash sweep that bounded windows skip; complementary real check: a bounded TFC build for 2023-12-30 → 2023-12-31 under an open-interception harness that records every path opened by Python (`builtins.open`, `os.open`, `io.open`), by hashing helpers and by DuckDB (the explicit file list passed to each query). Asserts: no path whose manifest `min_time ≥ 2024-01-01T00:00Z` (i.e. `date=2024-01-01/…683eef76…`) is opened or passed to DuckDB; every file contributing an in-range row is hashed. The filter is applied when the manifest is read, before any file path reaches a reader. A synthetic test covers the same rule with a fake manifest. Marked `real_data`.
  - Bar-only parity: `run_backtest(..., trade_updates=None)` replay digests equal master's for the same params on a fixture and on a bounded real window; existing `tests/test_optimize_orderflow.py`, `test_warm_start.py`, `test_validate_oos.py` unchanged and passing.
- Implements `scripts/phase4_benchmark.py` (FH4, owner req. 5): on pre-drawn development days (2019-09-11, 2021-05-19, 2023-10-16), measures TFC build wall and bytes per trade/bar, cached-replay wall per arm vs bar-only replay on identical bars, per-process peak RSS (`resource.getrusage` / wait4 of child processes), and extrapolates — using authenticated per-day trade counts from `hours.parquet` and full-segment bar counts from the existing bar-cache manifests — the full-history TFC build wall/disk/RSS, per-trial wall per arm on the 80 % search segment, and campaign totals for Stage A/B/C budgets at ≤ 6 concurrent study processes. Outputs `output/phase4/benchmark/benchmark.json` + `REPORT.md` with every extrapolation formula and its inputs. It never creates an Optuna study or storage, never writes outside the allowlist, and is refused for any date after 2023-12-31.
- Host and Claude inspect; Codex fixes.

### 19.2 What this build does NOT do (needs a later owner approval)
- Full-history TFC generation and the full-history gates FH1–FH3 (§14); the TFC build CLI refuses a range longer than 4 days unless `--full-history --owner-approved <token>` is given, and the build must not run it.
- Any Optuna study or trial, Stage A/B/C experiment, `--holdout-look`, or T8 full replay execution.
- 2024–2025 datasets, `COMPATIBLE_LEGACY_STRATEGY_FPS`, commits, push, merge, deployment. Changes stay uncommitted in the worktree `.claude/worktrees/phase4-optuna-trade-signals` (branch `feat/phase4-optuna-trade-signals`, base `390e29c`).

### 19.3 Superseded passages
- §1 items 1, 3, 4 and §14 FH1–FH4: in this build only B-FH, the bounded live/cache equality checks and the bounded FH4 benchmark run; FH1–FH3 (full history) and the campaign follow a later approval.
- §13 T8: code only; not executed beyond its unit tests.
- §17 the `build --bar-type all` / full `verify` commands are not run in this build.

### 19.4 Proof commands for this build
```bash
/Users/bobet/Documents/Code-Repository/Trading/orderflow-analysis-pro/.venv_orderflow/bin/python -m pytest -q -p no:cacheprovider
/Users/bobet/Documents/Code-Repository/Trading/orderflow-analysis-pro/.venv_orderflow/bin/python -m pytest -q -p no:cacheprovider -m real_data tests/test_phase4_real.py
/Users/bobet/Documents/Code-Repository/Trading/orderflow-analysis-pro/.venv_orderflow/bin/python scripts/phase4_benchmark.py --days 2019-09-11,2021-05-19,2023-10-16 --out output/phase4/benchmark
zsh /Users/bobet/Documents/Code-Repository/Trading/orderflow-analysis-pro/output/phase2_trade_pipeline/run_scripts/protected_hash.sh <out>   # before and after; must be identical to the Phase 3 after-snapshot
```
Run with cwd = worktree root so `pytest pythonpath=["."]` imports the worktree code (the venv's editable install points at the main checkout); the build report prints `orderflow_system.__file__` to prove it.

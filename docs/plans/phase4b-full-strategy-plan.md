# Phase 4b: full-strategy joint optimization (rev 10: strict tail inequality, P4B-18)

Branch `feat/phase4b-full-strategy`, created from `95bde47`, the reviewed and committed Phase 4 head.
Status: **plan only.** Nothing is implemented, no Optuna has run, and campaign `b54076c6…` is unchanged.

## 1. Goal

Find out whether the **complete** Orderflow Analysis Pro strategy is **net-profitable** under the registered costs, on BTCUSDT `fixed_tick_imbalance` and `fixed_volume_imbalance` bars. The complete strategy means:
- real trade-level volume footprint and volume profile;
- footprint absorption as the entry trigger;
- initiative, exhaustion and stacked imbalance read from the trade footprint.

The test is one pre-registered joint Optuna study per bar type. It uses the same 2019–2023 search segment, folds, costs, floors and seed as Phase 4 Stage B, and Stage B's best trial is the comparator.

### Acceptance criteria (observable)

1. **AC1: defaults unchanged.** Every Stage A replay digest (36) is byte-identical to the stored value when each of the six Stage A configurations is replayed with its **original** feature flags (off, A2, E2-bonus, E2-required, P2, ALL) and trade inputs. Every field introduced in 4b is held at its backward-compatible default. Every Stage B digest (best, top-3 and median per study, bar-only) is also byte-identical when re-replayed on the search segment (rev 4, P4B-12).
2. **AC2: absorption replaces M1.** With `absorption_source="trade"`, the bar delta-versus-close heuristic (M1) never emits, and legacy M2 never runs. Every ABSORPTION signal has `details.type == "trade_level_absorption"`.
3. **AC3: initiative reads the trade footprint.** With `initiative_imbalance != "off"`, initiative reads imbalance from the **trade** footprint, never the bar footprint. With `"required"`, initiative emits only when the directional stacked run is ≥ k. An execution-level test shows that the footprint condition changes break-even/trail actions and ledger exits.
4. **AC4: imbalance and stacked counts are exact.** Stacked-imbalance and imbalance counts match a hand-computed oracle on synthetic footprints, including:
   - diagonal geometry;
   - bucket aggregation;
   - a minimum-volume filter;
   - empty, one-sided and edge levels.
5. **AC5: Phase 4b is isolated.** Phase 4b registers in its own namespace and DB:
   - manifest `output/phase4b/campaign.json`;
   - studies `output/optimize/phase4b/studies.db`;
   - approvals `~/.orderflow-approvals/phase4b/`.
   Campaign `b54076c6…`, its seal, its checkpoints and `output/optimize/phase4/studies.db` stay byte-identical. The existing Phase 4 tests and `verify_campaign_artifacts` still pass for b54076c6.
6. **AC6: the report evaluates the gates mechanically.** The report computes the §7 net-profitability gates (G4b-N) and refuses a pseudo-holdout look unless they all pass. Both refusal and pass are tested on synthetic inputs.
7. **AC7: preservation and proofs.** Protected files (46), legacy DBs (3), the certified trade cache `COMPLETE.json` and the frozen builder files (4) are unchanged. The full pytest suite and the real-data tests pass.

### Non-goals

- Changing fees, slippage, folds, the embargo, floors, the seed or `holdout_fraction`.
- Rebuilding the trade cache, or changing the frozen builder files.
- Running Phase 4 Stage C, which is a separate owner decision (§10, D8).
- Any 2024–2026 data access. Phase 5 out-of-sample work needs separate owner approval after G3.
- Live trading or alerts; order book or sweep.

## 2. Evidence: existing versus new functionality

Source: worktree `95bde47` and upstream `0206ef7` (see `output/phase4/investigation/STRATEGY-ARCHITECTURE-FINDINGS.md`).

| Component | Exists at 95bde47 | Gap for "complete strategy" | Phase 4b change |
|---|---|---|---|
| Trade footprint ($1 buckets, aggressor bid/ask) | Built by `TradeAccumulator` and cached per bar (`fp_0/1/2`, `trade_feature_cache.py:33`); materialised lazily by `_CachedFootprint.levels` | None. The cache already holds every level, so **no rebuild** is needed | Reuse |
| Trade profile ($10, POC/VAH/VAL, integer-exact) | `TradeView.compute_profile`; `framing_profile_source="trades"` (P2) | Off by default | Fixed **on** in the 4b arm |
| Footprint absorption (A2) | `_check_trade_absorption` (`absorption.py:169-203`), 5 `ts_*` dims | Runs **only after** M1 fails (`absorption.py:105-112`), so M1 still drives most entries; A2 adds 13–33× signals | **New** `absorption_source` = `bar` (default) / `trade`. `trade` skips M1 and M2 |
| Exhaustion contrarian (E2) | `_trade_contrarian`, modes off/bonus/required, 3 `ts_*` dims | None | Reuse; mode fixed ex ante (D3) |
| Initiative one-sided imbalance prints | `initiative.py:100-105`, but `pipeline._detect` always passes the **bar** footprint (no levels), so it is inert | No trade input; horizontal ratio 3.0 hard-coded; a lone single-sided $1 print counts as an imbalance (`footprint.py:70-73`) | **New:** a trade footprint for initiative, plus a geometry, ratio, minimum-volume filter and bucket aggregation |
| Stacked imbalance | `FootprintEngine.count_consecutive_imbalances` (cross-bar, `footprint.py:197`); **no production caller** (test only) | Unused; it counts across bars, not stacked levels within a bar | **New:** within-bar stacked run of ≥ k adjacent same-direction imbalanced levels |
| Entry gating | `_check_new_entry`: ABSORPTION only, near a qualified profile level, composite score = 0.30·absorption strength + 0.25·level strength + 0.20·bias confidence | Initiative does not add to the entry score (aggregator docstring says 20 % but the code does not do it) | Unchanged (D4). Initiative stays as the break-even/trail manager, as in the upstream execution model |
| Exits | Initiative → break-even/trail; exhaustion/divergence ≥ 70 → exit; SL/TP from VAL/VAH, else percentage defaults | — | Unchanged; they read the improved signals |

### Why profitability is hard here (Stage A/B evidence, same costs)

- **Cost per trade.** Cost is about 12 bps round trip: 5 bps fee per side plus 1 bps slippage per side. In R that is roughly 12 bps / stop. At default stops it costs about 0.40 R per trade (Stage A).
- **Stage B optima.** Cost was 0.15–0.27 R per trade, against a gross of +0.09 to +0.18 R.
- **What a net-positive variant needs:** gross > cost per trade, so either a higher gross edge per trade or wider stops with fewer trades.

The plan does not touch costs. Stop width (`default_stop_pct`, `target_rr`) and the frequency-controlling dimensions stay in the joint search.

## 3. Design

### 3.1 Code changes (Codex builds; TDD, tests written first and failing)

Every new behaviour is opt-in behind `TradeSignalConfig`. The defaults reproduce 95bde47 exactly (AC1).

#### C1 — `TradeSignalConfig` (`config/settings.py`): new validated fields

| Field | Values | Default | Rule |
|---|---|---|---|
| `absorption_source` | `bar`, `trade` | `bar` | `trade` requires `absorption_levels=True` (the A2 parameters) |
| `initiative_imbalance` | `off`, `bonus`, `required` | `off` | — |
| `imbalance_ratio` | float | 3.0 | > 1 |
| `imbalance_min_level_share` | float | 0.01 | (0, 1]; the pair's volume as a share of bar volume, which filters lone prints |
| `imbalance_bucket_mult` | int | 1 | in {1, 2, 5, 10}; aggregates $1 cache buckets into $k levels |
| `stacked_min_levels` | int | 0 | 0 = off (diagnostic `bonus` only); 1–4 for `required` (`required` refuses 0) |

Validation mirrors the existing `__post_init__`: the type is checked, bool is rejected for numerics, and values must be finite.

`any_enabled` (`settings.py:230-233`) also returns true when `initiative_imbalance != "off"` or `absorption_source == "trade"` (rev 3, P4B-10). An initiative-only configuration therefore:
- passes through the missing-trade guard, raising `ValueError` rather than `AttributeError`;
- serialises as enabled in `trade_signal_identity`;
- is subject to `validate_trade_signal_range` (development range only);
- is refused for certificates.

Tests cover each of these for initiative-only configurations.

#### C2 — pure footprint functions (`analytics/footprint.py`, new functions only)

- **`aggregate_levels(levels, mult)`** groups integer price ticks by `floor(tick / (100·mult))`. It sums bid and ask, and it is exact on cached integers (cents and lots), with no float summation order dependence.
- **`diagonal_imbalances(levels, ratio, min_share, total)`** uses the industry-standard diagonal comparison. It compares the ask (aggressive buy) at level *i* with the bid (aggressive sell) at level *i − 1*: buy if `ask_i ≥ ratio·bid_{i−1}`, sell if `bid_{i−1} ≥ ratio·ask_i`.
  - A pair qualifies only if `(ask_i + bid_{i−1}) / total ≥ min_share`.
  - A zero opposite side counts only when that filter passes.
  - It returns ordered `(level, side)` pairs.
  - The upstream horizontal method is unchanged and stays in use by the legacy paths (D2).
- **`stacked_runs(imbalances, min_levels)`** returns the longest run per side of *adjacent* aggregated levels with the same side.

#### C3 — initiative (`patterns/initiative.py`)

- `check_bar(..., trade_footprint=None, trade_bucket_ticks=None)`.
- When `initiative_imbalance != "off"`, a trade footprint is required (`ValueError` otherwise, like A2). The directional diagonal imbalance count and stacked run come from the aggregated trade footprint.
- `required` (the only Phase 4b mode, rev 2, P4B-01): no emit unless the longest stacked run in the delta direction is ≥ `stacked_min_levels` (k ≥ 1; k = 1 means "at least one imbalanced level").
- Why this mode:
  - Initiative's only execution consumers are break-even and trail (`aggregator.py:291-363`). Both ignore strength, and the ledger ignores management scores.
  - A strength-only `bonus` would therefore change digests but never trades.
  - In `required` mode the footprint condition decides whether break-even and trail actions fire, so it is economically active.
- `bonus` stays available (strength `+10·count`) for diagnostics only. It is never used in a 4b arm, and `effective_trade_config` refuses it for `phase4b`.
- New details: `imbalance_source="trade"`, `imbalance_levels`, `stacked_run`.
- With `"off"` the code path is byte-identical to today.
- **Execution-level test.** On a synthetic bar sequence with an open position, changing only the trade footprint, from no directional imbalance to a qualifying stack, changes the emitted `break_even`/`trail` actions and the resulting ledger exit. This proves the dimensions are economically active.

#### C4 — pipeline (`pipeline._detect`)

Pass `trade_args` to initiative when `initiative_imbalance != "off"`. Extend the bucket-width integrity check at `process_bar` to this flag.

#### C5 — absorption (`patterns/absorption.py`)

- When `absorption_source == "trade"`, `check_bar` returns `_check_trade_absorption(...)` directly: no M1 and no M2.
- `absorption_min_delta_pct` is then inert. It is removed from the 4b **sampled** dimensions. `build_instrument_config` still reads it unconditionally (`optimize_orderflow.py:354`), so C6 materialises it (P4B-04).

#### C6 — search space and config mapping (`scripts/phase4_studies.py`)

- New bounds `FULL_STRATEGY_SPACE` = the bar dimensions minus `absorption_min_delta_pct` (11), plus A2 (5), plus E2 (3), plus initiative imbalance:
  - `ts_imbalance_ratio` (2–6, log);
  - `ts_imbalance_min_level_share` (0.002–0.05, log);
  - `ts_imbalance_bucket_mult` (categorical {1, 2, 5, 10});
  - `ts_stacked_min_levels` (int 1–4; 1 = any single imbalance, ≥ 2 = stacked).
- That is 23 dimensions.
- **Sampled versus effective parameters (P4B-04).** `full_strategy_params(sampled)` returns the complete effective dict:
  - the sampled dimensions;
  - `absorption_min_delta_pct` = the registered inert value (the instrument default, `base.absorption.min_delta_pct`), recorded in the study identity as `inert_params`.
  - **Every 4b full-strategy builder call** goes through this function: the 4b objective, S1 full-strategy replays, 4b candidate report replays, neighbour replays and the 4b holdout replay.
  - Test: each of those five call sites, given only sampled dimensions, builds without `KeyError`, and the inert value appears in the replay identity.
  - **Not applied to comparators or legacy configurations (rev 3, P4B-08; rev 4, P4B-12).** Each of these replays with its **exact stored parameters and its stored feature flags**:
    - The Stage B comparators and historical `off` configurations replay bar-only, with trade signals disabled as stored. Tick trial 187 has `absorption_min_delta_pct` 0.2525…, volume trial 124 has 0.5423…, and M1 is active in both.
    - The Stage A feature configurations (A2, E2-bonus, E2-required, P2, ALL) replay with their original flags and trade inputs. Every 4b-only field (`absorption_source`, `initiative_imbalance`, `imbalance_*`, `stacked_min_levels`) is held at its default.
  - Three type-separated entry points: `replay_comparator(frozen_artifact)` (bar-only), `replay_legacy(stageA_config)` (original flags) and `replay_full_strategy(sampled)`. Misuse is a `TypeError`.
  - Report integration test: the comparator parameters and digest are unchanged, while the 4b candidate receives the inert constant.
- `effective_trade_config` maps the fixed structure (D1–D3) together with the `ts_*` values.
- `suggest_params` gains categorical support.
- No warm start: 4b studies run with `--no-warm-start`, so no Stage B parameters are enqueued.
- **Neighbour perturbation (P4B-07).** A **new** `neighbours_rel` is used only by the 4b report; the Phase 4 `neighbours` is unchanged. The convention, pre-registered:
  - Numeric dimensions: base × 0.9 and base × 1.1, i.e. ±10 % of the **value**, not of the span.
  - For a base of 0, use ± 10 % of the span instead.
  - Ints: round, then force a step of at least ±1.
  - Clip to bounds. A neighbour equal to the base after clipping is `unavailable` and excluded.
  - Categoricals: the adjacent categories in declared order.
  - The gate needs ≥ 50 % of the 2·d neighbours available, else it fails.
  - A hand-computed oracle covers float, log, int, zero-base, clipped and categorical cases.

#### C7 — campaign namespace (`scripts/phase4_campaign.py`, `phase4_approvals.py`)

- Registration is parameterised by `phase` (`phase4` or `phase4b`). For `phase4` it is byte-identical: the same arms, manifest bytes and hash for an unchanged input (regression test with b54076c6 inputs).
- `phase4b` arms: `BTCUSDT:<bt>:full`, kind `study`, budget N = 300 (D6, owner-accepted). Each arm carries:
  - `comparator`: the Stage B frozen artifact path and sha256 (`output/phase4/frozen/fixed_*_imbalance.json`);
  - the Stage B study name;
  - the campaign hash `b54076c6…`.
- Stages for 4b:
  - `S1`: deterministic replays and a benchmark;
  - `S2t` and `S2v`: one study each, so each has its own approval;
  - `S3`: report, then a single gated look per arm.
- Approval root `~/.orderflow-approvals/phase4b/`, reusing the single-use, argv-bound and write-ahead machinery unchanged.
- Snapshots: the 3 legacy DBs, plus `output/optimize/phase4/studies.db` (read-only, sha recorded at registration; Stage B is complete and Stage C is not running), plus the 4b DB by path.
- **Phase-aware filesystem policy and writers (P4B-05).** A single `PhaseLayout(phase)` value object supplies every path. The writers below take a layout, not a literal; for `phase4` the layout returns exactly today's paths.

  | Item | phase4 (today) | phase4b |
  |---|---|---|
  | Write roots | `output/phase4`, `output/optimize/phase4`, `output/optimize/trade_cache`, `output/duckdb_tmp` | `output/phase4b`, `output/optimize/phase4b`, `output/duckdb_tmp` |
  | Studies DB | `phase4_db_path` (Phase 4 DBs only) | `output/optimize/phase4b/studies.db` only |
  | Driver logs and checkpoints | `phase4_campaign.py:387,408,497` | `output/phase4b/…` |
  | Report snapshots | `phase4_report.py:125` | `output/phase4b/reports/snapshots/` |
  | Execution and driver locks | — | under the 4b checkpoint directory |
  | Exposure file | — | under `output/phase4b/` |

  - **Read-only for phase4b:** the trade cache, `output/optimize/trade_cache`, `output/phase4/**` and the phase4 DB. `checked_write` for phase4b refuses them; they are added to that layout's protected list.
  - **Tests:**
    - an integration run (synthetic data, temporary ROOT) from register → S1 → S2t (2 trials) → S3 writes only under the 4b roots;
    - a hash snapshot of a fake `output/phase4/` tree is unchanged;
    - the phase4 `checked_write` and `phase4_db_path` behaviour is identical (existing tests unchanged).
- **Runtime ceilings without mutating a sealed campaign (P4B-06).** There are two registrations:
  1. **`phase4b-s1` calibration manifest.** S1 only, with fixed pre-registered ceilings: trial-equivalent 600 s, stage 3 h. It produces `output/phase4b/s1/benchmark.json`, sealed by its sha.
  2. **`phase4b` campaign manifest** (S2t, S2v, S3), registered **after** S1. It binds the S1 benchmark sha and derives its ceilings by a formula pre-registered here and coded in `derive_ceilings(benchmark)`:
     - `max_trial_seconds = ceil(1.3 × max per-trial-equivalent seconds measured for that bar type)`;
     - `max_study_seconds = ceil(1.15 × mean per-trial-equivalent seconds × N)`.
     - Registration refuses an S1 benchmark that does not verify, or limits that differ from the formula.
  - Each manifest gets its own approvals; nothing sealed is edited.
  - Test: the formula on a fixture benchmark; refusal of tampered benchmark bytes; refusal of hand-supplied limits that disagree.

#### C8 — report (`scripts/phase4_report.py`)

- **Net bootstrap.** The existing moving-block bootstrap is applied to **net** R as well as gross, jointly with the Stage B comparator, using the same blocks, resamples and seed.
- **New gates, all required for G4b-N (§7):**
  - fold consistency;
  - a slippage-stress point;
  - DSR at campaign multiplicity (the union now includes the Phase 4 DB and the 4b DB).
- Gross-edge gates are kept and reported.

#### C8b — holdout allocation across campaigns (P4B-03)

- **A shared, campaign-independent allocation record:** `~/.orderflow-approvals/holdout-allocation/<dataset_id>__BTCUSDT__<bar_type>__<holdout_start_seq>-<end_seq>.json`.
  - Mode 0600, created with `O_EXCL` and content-addressed.
  - It binds the allocated campaign hash and phase.
  - It lives outside both campaign output trees.
- **The 4b look path checks it before opening any holdout byte**, on every path, under the existing look lock: S3, recovery after an interrupted look, and execution from any checkout. The 4b look refuses unless:
  - an allocation record exists for this bar type and holdout interval;
  - it names this 4b campaign hash;
  - and the **canonical** Phase 4 exposure evidence shows no look for that bar type (rev 3, P4B-09). This is resolved independently of the executing checkout:
    - The 4b registration binds `phase4_origin`: the absolute root of the checkout that owns b54076c6 (today `…/.claude/worktrees/phase4-optuna-trade-signals`), the manifest path, and sha256 `b54076c6…`.
    - The binding also records the exposure-file path that `holdout_look` writes under that root (`phase4_studies.py:460`).
    - At look time the check re-hashes the origin manifest and requires `b54076c6…`, which proves it is the right source. It then reads the exposure file there.
    - The result is one of three states: `verified_absent` (manifest verified, exposure file absent or without this bar type's arms); `present` (refuse); `unavailable` (origin missing, unreadable or the hash differs: refuse).
    - The allocation record also stores the origin binding.
- **Allocation is installed only by the host after the owner's explicit decision (D7)**, like the other approval records.
- **Accepted residual.** Phase 4 code at `95bde47` cannot read the allocation record, and editing it would invalidate campaign b54076c6's source binding. So the Phase 4 direction is enforced **procedurally**: if D7 gives the look to 4b, no Stage C approval (with or without a look) is drafted or installed while a 4b allocation exists. This is recorded in the review log and the owner decision record.
- **Tests:**
  - two checkout roots with a reserved or completed exposure present only in the origin root: the 4b look refuses from the other root; a missing or tampered origin gives `unavailable` and a refusal;
  - refusal with no record, with another campaign's record, and when a Phase 4 exposure exists;
  - refusal on the recovery path;
  - a record created twice raises `FileExistsError`.

#### C9 — S1 runner

A thin driver mode replays fixed configurations on the search segment only (see §5.1 for the read boundary). It records:
- gross R, cost R and net R;
- trades and signal counts;
- runtime and peak RSS per replay.

Its output feeds the S2 ceilings, but it is **not** a gate input.

### 3.2 Fixed strategy structure (ex ante, not tuned)

The structure fixed below is the recommendation; the owner confirms it in §10.

- `absorption_levels=True`, `absorption_source="trade"`: owner-stated.
- `framing_profile_source="trades"`: the real profile is core.
- `exhaustion_contrarian`: `required` or `bonus` (D3).
- `initiative_imbalance`: `required` (rev 2, P4B-01). `bonus` cannot change trades.
- Stacked imbalance (D3): a required initiative confirmation with tuned k ∈ 1–4, where k = 1 means a single imbalance. Alternatively, initiative imbalance is excluded altogether (`off`).

**4b S1 defaults**, pre-registered and used only for the S1 "full strategy at defaults" replays:
- `stacked_min_levels` = 3, `imbalance_ratio` = 3.0, `imbalance_min_level_share` = 0.01, `imbalance_bucket_mult` = 1;
- the A2 and E2 values are the existing `TradeSignalConfig` defaults;
- the bar parameters are the instrument defaults, with `absorption_min_delta_pct` inert.

Fixing the structure ex ante keeps one arm per bar type and avoids a categorical search over strategy definitions, which would inflate multiplicity.

## 4. Experimental protocol

All runs are sequential, one process at a time. The existing safeguards are unchanged: strict RSS and private memory ≤ 4 GiB, memory preflight, heartbeat watchdog, runtime ceilings, write-ahead markers, and an external `timeout` on the process group.

| Stage | What | Runs | Est. runtime | Approval |
|---|---|---|---|---|
| P0 | Build (Codex), host proofs, fresh Claude inspection, local commit | — | — | Owner: build |
| S1 | Deterministic replays on the search segment, per bar type: full strategy at defaults; leave-one-out (initiative imbalance `off`, k = 1 instead of default k = 3, E2 `off`, P2→bar_vwap); full strategy at Stage B best bar params. Benchmark 3 trial-equivalents | 2 × 6 replays | ≈ 1–1.5 h | Owner: S1 (under the `phase4b-s1` manifest, C7); checkpoint, then the `phase4b` campaign is registered with the derived ceilings |
| S2t | Joint TPE study, tick imbalance, budget N = 300, seed 42 | 300 | N × ≈ 220–260 s (to be measured in S1) | Owner: per study |
| S2v | Same, volume imbalance | N | same | Owner: per study |
| S3 | Phase 4b report; G4b-N per arm; **single** pseudo-holdout look only for passing arms; G3 | ≤ 2 looks | < 1 h | Owner: S3; checkpoint |

Fixed settings:
- Objective: unchanged, `two_tier_day_weighted_v2`, the day-weighted mean CV Sharpe **net** of 5 + 1 bps.
- Folds: 5 chronological, 2 % warm-up.
- Floors: 11 trades per fold, 105 total.
- `holdout_fraction`: 0.2 (the pseudo-holdout, still unlooked).
- Comparator: the Stage B best per bar type, tick trial 187 and volume trial 124, replayed digest-verified.

## 5. Overfitting controls

1. **Search segment only during S1 and S2: integrity reads versus strategy exposure (P4B-02).**
   - **Integrity reads (allowed).** The frozen `CachedTradeUpdates` constructor hashes every cache file and computes `trade_digest()` over all bars (`trade_feature_cache.py:355-371,424-434`). `validate_bars` needs the full-length bar array (406-410). These reads produce only hashes and equality booleans, which are compared to registered constants. No value reaches the strategy, objective, report or logs.
   - **Strategy exposure (bounded).** A new `SearchPrefix` wrapper, in `phase4_studies.py` and **outside** the frozen builder files, does three things:
     - it opens the full `CachedBars` and `CachedTradeUpdates` for validation only;
     - it exposes to replay only `bars[:n_search]` and a `TradeUpdatesPrefix` whose `__getitem__`/iteration raise `HoldoutAccessError` for an index ≥ n_search;
     - it is the only object handed to `run_backtest`, the S1 replays and report replays before S3.
   - **Tests:**
     - through the complete cache-opening path on a synthetic cache, a spy shows no `__getitem__` at index ≥ n_search during an objective call, an S1 replay and a report replay;
     - a deliberate holdout access raises;
     - validation still detects a tampered holdout byte, via the hash mismatch.
   - Pre-S3 artifacts never contain holdout-derived values.
2. **One arm per bar type, ex-ante structure, fixed budget.** There is no adaptive re-running or space editing after S1. Any change requires a new pre-registration, and its trials count toward multiplicity.
3. **CSCV PBO** with 16 blocks: required < 0.5.
4. **Deflated Sharpe at full campaign multiplicity.** The no-skill spread is dependence-robust and computed per candidate, with one Sharpe definition throughout. Exact definition in §7.2 (rev 8). Required: DSR ≥ 0.95 (D5). The rev 5 calculation is a diagnostic only.
5. **Block bootstrap**, 7-day blocks, 2 000 resamples, seed 20261105, on gross **and net** R, jointly with the comparator.
6. **Fold consistency:** at least 3 of 5 folds with net Sharpe > 0 (D5).
7. **Parameter stability:** with the `neighbours_rel` convention (C6: ±10 % of the value, clipped, unavailable excluded, ≥ 50 % available), the median net E_r across the best trial's neighbours is > 0. Reported, and gated (D5).
8. **Cost stress:** net E_r > 0 at fee 5 bps / slippage 2 bps, using the re-simulation procedure of §7.1 item 9. The fixed-quantity grid is reported as a diagnostic only.
9. **Frequency multiple** against Stage B best: reported. It does not gate, because net gates already price frequency.
10. **Single pseudo-holdout look per arm.** It is recorded through the existing exposure and reservation mechanism, **and** it requires the cross-campaign allocation record (C8b). The 2024–2025 out-of-sample data is untouched and the 2026 test set is never touched.

11. **Pseudo-holdout provenance (§5.11, host-verified 2026-11-07, read-only).** The pseudo-holdout is the last 20 % of 2019-09-10 → 2023-12-31 bars, after a 2 % embargo.
   - **The boundary is identical everywhere.** Every existing BTCUSDT study has `holdout_fraction` 0.2, embargo 0.02 and the same `n_bars_search`: 1 871 694 for tick and 1 597 471 for volume, in both the legacy DBs and the Phase 4 DB.
   - **No look is recorded anywhere.**
     - There are zero `holdout::` study attributes and zero `holdout*` trial attributes in all 4 DBs.
     - `results.json` shows `holdout_history` = {} for every study.
     - There is no Phase 4 exposure file.
     - Stage A replayed `bars[:n_search]` only (`phase4_campaign.py:231-236`), and the Stage B analysis used `CachedBars(stop=n_search)`.
     - The Phase 4 speed and memory benchmarks (`benchmark.json`, `w1-fullhistory-speed.json`) contain runtimes and signal counts but no P&L or trade outcome.
   - **Residual caveats, not model selection:**
     - The info-bar construction thresholds were calibrated on the full 2019–2023 history, so the look is reported as *conditional* (the existing `evaluate_holdout` caveat).
     - The full-history benchmarks computed signal counts over holdout days, which are frequency statistics only.
     - Exploration that left no artifact cannot be proven absent.

## 6. Comparator handling

- **Stage B best** (frozen JSON, sha recorded) is replayed with digest verification. Its trades provide the joint-bootstrap baseline.
- **Stage A and Stage B outputs** are read-only inputs. No file under `output/phase4/` is modified, and a test asserts their hashes before and after S1–S3.

## 7. Gates

- **G-S1:** diagnostic. The owner checkpoint reports gross R, cost R, net R, trades, signals, runtime and RSS. Nothing is blocking, and S1 results do **not** change the S2 space or budget.
- **G4b-N** (arm → pseudo-holdout look). This is an intersection-union test: every item is required, and failing any one blocks the look. An "indeterminate" item (not computable) counts as a **fail**.

### 7.1 Exact gate definitions (rev 5)

All definitions use only the search segment.

**Optimization objective versus gate statistic (rev 7, clarification only).** Optuna ranks trials by the unchanged objective `two_tier_day_weighted_v2`: the day-weighted mean of the net fold Sharpe ratios. Expectancy is **not** the objective. E_r below (the mean per-trade R) is used only after optimization, by G4b-N items 1–4, 8 and 9 and by G3. Every use computes it with the same function on the same scored-trade definition, at the same costs (5/1 bps, except item 9's 5/2 bps). Selection by Sharpe and gating by E_r are distinct statistics by design; neither is substituted for the other.

Notation:
- **Candidate** = the best feasible trial of the arm's study: highest `value`, ties broken by lowest `number`. It is replayed through `replay_full_strategy`, digest-verified against the stored `replay_digest`.
- **Comparator** = the Stage B best for the same bar type, replayed through `replay_comparator`, digest-verified.
- **Scored trades** = `diagnose.scored_only(simulate_trades_risk_based(ledger, fee_bps=5, slippage_bps=1), search_bars, folds)`. A trade counts in the fold where it exits; warm-up bars are excluded.
- **Per-trade R** = `pnl / dollar_risk`, using `net_pnl` for net and `gross_pnl` for gross. Only trades with `dollar_risk > 0` count; excluded trades are counted and reported, as `_gross_days` does today.
- **E_r** = the **equal-weight mean of per-trade R** (rev 6, P4B-13). This is the existing semantics everywhere: `_gross_days`/`gross_bootstrap` (`phase4_report.py:178-215`), `diagnose.decompose`, `cost_grid` and G3. The plan uses this single estimator for every gate item (1, 2, 3, 4, 8, 9) and for G3.
- New net helpers mirror the gross ones exactly. The Phase 4 helpers are unchanged.
- **Unequal-risk oracle test:** two trades with risks 1 and 10 and P&Ls +2 and −10. E_r = +0.5 (mean R), and the risk-weighted ratio (−8/11) is **not** produced. The same pair repeated daily makes the bootstrap pass under mean R.

| # | Statistic and computation | Pass rule | Test (synthetic) |
|---|---|---|---|
| 1 | **Net bootstrap.** Moving-block bootstrap over the UTC calendar days of the search segment: 7-day blocks, 2 000 resamples, seed 20261105, using the existing `gross_bootstrap` resampling with net instead of gross pnl. Each resample computes E_r = Σ(per-trade R of trades exiting on the drawn days) / (their count).

**Invalid-resample rule (existing):** a resample with zero candidate trades or zero comparator trades is invalid. More than 1 % invalid makes the item indeterminate, which counts as a fail. | 2.5th percentile of the net E_r draws > 0 | all trades +1 R passes; all −1 R fails; reproducible by seed |
| 2 | **Net bootstrap of the difference.** The same day-block draw is applied **jointly** to candidate and comparator trades, and the statistic is net E_r(candidate) − net E_r(comparator). | 2.5th percentile > 0 | identical inputs give exactly 0 on every draw → fail |
| 3 | **Gross bootstrap.** The existing `gross_bootstrap` arm CI. | 2.5th percentile > 0 | existing tests |
| 4 | **Top-3.** The three highest-`value` feasible trials (same ordering) are each replayed and digest-verified. Point net E_r is taken over each trial's scored trades. | all 3 > 0; fewer than 3 feasible trials → fail | one negative → fail |
| 5 | **PBO.** `cscv_pbo(daily_matrix(all feasible trials of the study), blocks=16)`. | < 0.5; fewer than 2 feasible trials or a ValueError → fail | known 0/1 matrices |
| 6 | **DSR and tail check.** Defined in §7.2. | DSR ≥ 0.95 **and** SR̂ **>** c_tail (strict) | §7.2 tests |
| 7 | **Folds.** The candidate's stored `fold_results[i]["sharpe_ratio"]` (net, annualised). A penalized or non-finite fold counts as ≤ 0. | ≥ 3 of 5 strictly > 0 | 2/5 fails; 3/5 passes; a penalized fold is not counted |
| 8 | **Neighbours.** Each `neighbours_rel` neighbour (C6) is replayed and its net E_r computed. A neighbour with zero scored trades counts as net E_r = 0. Unavailable neighbours are excluded. | ≥ 50 % of the 2·d neighbours available **and** median net E_r over the available ones > 0 | hand oracle |
| 9 | **Slippage stress: re-simulation** (rev 6, P4B-14). The candidate's complete search-segment ledger is re-simulated with `simulate_trades_risk_based(ledger, fee_bps=5, slippage_bps=2)`, so equity, quantities and dollar risk follow the stressed costs. Then `scored_only` is applied and E_r is the mean per-trade R. The report's gate row uses this exact procedure. The existing fixed-quantity `cost_grid` is still reported, labelled "diagnostic: fixed-quantity repricing", and is never a gate input. | net E_r > 0 | multi-trade compounding oracle: the re-simulated quantities differ from the fixed-quantity ones, and the gate follows the re-simulation |
| 10 | **Floors.** The candidate's `feasible` is True: 11 trades per fold, 105 in total. | True | existing |

### 7.2 Deflated Sharpe and multiple-testing accounting (rev 8: dependence-robust null; owner-approved R1–R4)

**Why rev 8.**
- The rev 5 definition compared a pooled daily SR̂ with a dispersion σ taken from fold-mean `raw_daily_sharpe` across heterogeneous, mostly losing trials. That is two different Sharpe definitions.
- Under no skill it implied a winner about 11 σ₀ above zero, i.e. near-zero power.
- The owner approved R1–R4:
  - one Sharpe definition throughout;
  - the full multiple-testing count;
  - the 0.95 threshold;
  - the rev 5 calculation kept as a diagnostic only.
- Host checks on stored Stage B search returns showed that a plain 1/√(T−1) spread is **not** valid. Returns are serially dependent (the most active trials have ρ₁…₇ ≈ 0.2), and a HAC-studentized normal threshold still left an excess of about 4 × 10⁻⁴ per trial (`output/phase4b/evidence/p4b_null_check*.json`).
- Rev 8 therefore takes the no-skill spread from the candidate's own returns, using a dependence-preserving bootstrap.

**Inputs (one Sharpe definition throughout).**
- **Returns:** `r = search_returns(full_candidate, search_bars, folds)` (`phase4_report.py:56`): daily scored net P&L ÷ equity at the day's open, over every UTC day of the search segment, with 0 on days without a scored exit. `T = len(r)`.
- **SR̂** = `mean(r) / std(r, ddof=0)`, per day and not annualised. The no-skill spread uses this same SR̂ estimator.

**No-skill spread σ_null.** It is computed from the candidate's own returns, pre-registered:
- **Bootstrap scheme:** a demeaned moving-block bootstrap over calendar days. d = r − mean(r).
  - Each resample concatenates ⌈T/b⌉ blocks, each starting at a uniform random day (0 … T−b), and truncates to T.
  - SR_b = mean/std(ddof = 0) of the resample, or 0 if the resample has zero std.
- **Block lengths** b ∈ {7, 30, 90} days, with B = 2 000 resamples each, seed 20261107 (one `numpy.random.default_rng` stream, blocks drawn in the order 7, 30, 90).
- **σ_null** = max( 1/√(T−1), sd_7, sd_30, sd_90 ), where sd_b is the standard deviation of the 2 000 SR_b draws.
  - The floor 1/√(T−1) is the first-order iid std of SR̂ at SR = 0 (Lo 2002), where the skew and kurtosis terms vanish.
  - Taking the maximum over block lengths is deliberately conservative for long-memory or regime dependence.

**Multiplicity N (unchanged, the full family).**
- N = `n_symbol_all_studies` from `campaign_union`: COMPLETE BTCUSDT trials, deduplicated by (study_name, number, params fingerprint, replay_digest).
- Snapshots: the 3 legacy DBs (`9a5d8e7d…`, `42219fa8…`, `eaf1e59c…`), the Phase 4 DB at its sha when 4b is registered (today `49c3d149…`), and the 4b DB via a backup-API snapshot at report time. Each sha is verified before and after reading.
- All bar types, strategies, phases and infeasible trials count. There is no clustering and no effective-N reduction.
- Today N = 1 313; after 2 × 300 4b trials, N = 1 913. If Phase 4 Stage C is ever run before the 4b report, its trials enter N. N is recomputed at report time and never hard-coded.

**Benchmark and DSR.**
- **e_N** = `expected_max_sharpe_ratio(N, 0.0, 1.0)`, the expected maximum of N independent standard normals (RiskLabAI). At N = 1 913, e_N = 3.435.
- **SR\*** = σ_null · e_N.
- **DSR** = Φ( (SR̂ − SR\*) / σ_null ).
- **DSR condition:** DSR ≥ 0.95, which is equivalent to SR̂ ≥ σ_null·(e_N + 1.645) = 5.08·σ_null at N = 1 913.
- **Bootstrap tail condition (rev 9, P4B-15/16; rev 10, P4B-18).** SR̂ **>** c_tail, a **strict** inequality, where c_tail = max over b ∈ {7, 30, 90} of the empirical (1 − 0.05/N) quantile of SR_b. SR_b uses the same demeaned moving-block scheme as σ_null, with **1 000 000** draws per block, a separate `default_rng(20261108)` stream, blocks in the order 7, 30, 90, and quantile method `numpy` **"inverted_cdf"**.
  - The quantile c_b is then the smallest draw value v with F̂_b(v) ≥ 1 − α/N. That makes the empirical exceedance fraction #{SR_b > c_b} / 10⁶ ≤ α/N exactly, even with ties.
  - The strict ">" means a candidate sitting on an atom at c_b is never admitted (rev 10, P4B-18).
  - Using "inverted_cdf" instead of "linear" removes the interpolation slack at the same boundary; it is part of the same fix.
- **Gate item 6 passes only if both conditions hold:** SR̂ ≥ σ_null·(e_N + 1.645) (the DSR condition, DSR ≥ 0.95, equality allowed) **and** SR̂ > c_tail (strict).
  - The DSR condition is the owner-approved D5 test.
  - The tail condition guards candidates whose null tails are heavier than normal, where a σ-scaled normal threshold is not calibrated (see the verification below).
- The comparison is made by 4b code. The library's `passes` flag, which means > 0.5, is never used.
- **Why σ_null replaces the moment formula.** PSR's iid moment formula is replaced by σ_null because σ_null already embeds the skew, kurtosis, zero days and serial dependence of the returns under the null. At SR̂ of about 0.15 per day, the moment formula's SR²/2 adjustment is under 1.2 %.

**Indeterminate, which counts as a fail:**
- T < 3;
- `std(r) = 0`;
- r contains a non-finite value;
- σ_null or c_tail is non-finite;
- N < 2;
- any snapshot sha mismatch.

**Implied threshold.**
- The threshold is candidate-specific: SR̂ ≥ 5.08·σ_null **and** SR̂ > c_tail.
- For the Stage B best trials, the DSR condition binds: 5.08·σ_null = 0.162 (tick) and 0.139 (volume) per day, against c_tail = 0.115 and 0.094 (`p4b_null_v2.json` §C).
- At the iid floor, with T = 1 328, that is 0.139 per day, about **2.67 annualised**.
- For the median feasible Stage B dependence profile (σ_null/σ₀ = 1.07) it is about **2.84 annualised**.
- For strongly dependent candidates it is higher, up to about 10 annualised at the observed maximum ratio of 3.8.
- For reference, the Stage B best was 0.82 annualised (tick) and 0.61 (volume).

**Error-rate verification (rev 9; replaces the rev 8 text, which was withdrawn per P4B-15/16/17).**
All checks are host-run and read-only on feasible Stage B trials with stored search returns. Evidence: `output/phase4b/evidence/p4b_null_v2.json` and the script beside it.

- **(A) Marginal size of the complete gate statistic, with σ_null re-estimated on every null sample (P4B-16).**
  - Each outer null sample is a demeaned block bootstrap of the trial (outer blocks of 30 and 90 days, 4 000 samples each). σ_null is recomputed from that sample with the registered inner scheme (300 draws per block, a reduced count for compute), and z = SR/σ_null.
  - Results, as exceedance compared with the normal tail at z = 2.33 / 3.09 / 3.72 (normal 1e-2 / 1e-3 / 1e-4):

    | Trial | Exceedance | Compared with normal |
    |---|---|---|
    | Best and median, tick and volume | 0 – 5.5e-3 / 0 – 7.5e-4 / 0 – 2.5e-4 | at or below normal, except tick-best at outer block 30, z = 3.72: 2.5e-4 |
    | Trial 9 (most active and strongly persistent), tick and volume | up to 3.4e-2 / 1.3e-2 / 7.8e-3 | **30–78× heavier than normal** at z = 3.72 |

  - **Conclusion:** the σ-scaled normal threshold alone is **not** calibrated for strongly dependent candidates. That is why the bootstrap tail condition was added.
- **(B) Joint fixed-scale approximation (P4B-17 fixed).**
  - All 274 feasible trials start on 2019-09-10 (asserted), and the series are aligned on the leading 1 328 UTC days.
  - The same outer draw is applied to all trials; σ_null is held fixed per trial. This is an **approximation**, labelled as such.
  - Fraction of 20 000 draws with max_n z_n ≥ the threshold:
    - outer block 30: 2e-4 at the N = 274 threshold (4.51), and 0 at the N = 1 913 threshold (5.08);
    - outer block 90: 2.8e-3 and 1.15e-3.
  - **These figures describe the evaluated 274-trial family only.** They are **not** extrapolated to N = 1 913: the rev 8 "≲ 0.9 %" was an invalid scaling and is withdrawn.

**Family-wise error claim (the only one made).**
- For every trial n, the tail condition requires SR̂_n **>** c_tail,n, the trial's own bootstrap (1 − α/N) quantile, with `inverted_cdf`. For each block b, the empirical P(SR_b > c_b) ≤ α/N holds exactly, including with ties and atoms (P4B-18). Taking the maximum over blocks only lowers the exceedance.
- If each trial's bootstrap null approximates its true null distribution, then P(trial n passes | no skill) ≤ α/N.
- The event "the selected candidate passes" is contained in the union of the per-trial pass events. By the union (Bonferroni) inequality, which holds for **any** dependence across trials, **FWER ≤ N · α/N = α = 5 %**.
- The DSR condition only adds strictness.
- **No smaller FWER figure is claimed.**

**Assumptions and limits, stated:**
1. **Bootstrap validity is the load-bearing assumption.** The demeaned moving-block bootstrap with blocks of at most 90 days must approximate each trial's null distribution of SR̂ in the (1 − α/N) tail. Dependence longer than 90 days, or non-stationarity, can break this; trial 9 shows strong persistence.
2. **The tail quantile is at the 1 − 2.6 × 10⁻⁵ level.** With 10⁶ draws per block, about 26 draws lie beyond it. The quantile's Monte Carlo error is not negligible; the fixed seed makes it deterministic, not exact.
3. **(A) used 300 inner draws per block and 4 000 outer samples.** It resolves tails only down to about 2.5e-4, so it validates the **need** for the tail condition, not its exact size.
4. **The Stage B family is a proxy.** 4b trials use another strategy definition. Both conditions are recomputed per candidate.
5. **e_N assumes independent trials.** Positive correlation between trials makes the DSR condition conservative.

**Diagnostics (reported, never gate inputs):**
- (a) The rev 5 calculation, unchanged: `deflated_sharpe_gate`-style PSR with σ = std of `raw_daily_sharpe` over the union, labelled "rev 5 empirical-σ DSR (diagnostic)".
- (b) The iid-floor DSR, σ_null = 1/√(T−1).
- (c) sd_7, sd_30, sd_90, the per-block tail quantiles, and which block and which condition bind.
- (d) Lag-1 to lag-30 autocorrelations of r.

**Tests:**
- σ_null is reproducible by seed;
- it equals the floor for iid normal input with large T, within Monte Carlo error;
- AR(1) input with ρ = 0.3 gives σ_null/σ₀ ≈ √((1+ρ)/(1−ρ)) within tolerance;
- the max over blocks binds as expected;
- e_N matches the closed form;
- DSR is exactly 0.95 at the boundary, and the gate passes there;
- each indeterminate case fails;
- the rev 5 diagnostic matches the library;
- the union computation is checked on a synthetic three-DB set with duplicates;
- a tampered snapshot is refused;
- the gate never reads the library's `passes`;
- c_tail is reproducible by seed;
- c_tail matches a hand-computed `inverted_cdf` quantile on a small fixture;
- **tie/atom regression test (rev 10, P4B-18).** Use a fixture bootstrap distribution with an upper-tail atom: a mass of tied draws at exactly the value that becomes c_tail, with a mass above α/N.
  - (i) A candidate with SR̂ == c_tail is **rejected**, even when the DSR condition passes.
  - (ii) A candidate with SR̂ = the next representable float above c_tail passes the tail condition.
  - (iii) The fixture's empirical #{draws > c_tail} / n ≤ α/N.
  - (iv) Under `linear` the same fixture would place c_tail inside the atom. The test pins `inverted_cdf`;
- the max over blocks binds as expected;
- the gate requires **both** the DSR and tail conditions: a fixture where DSR passes and the tail check fails is refused, and vice versa;
- a non-finite c_tail fails.

**Other multiple-testing accounting:**
- **Within G4b-N:** an intersection-union test is level-α without correction, so no Bonferroni correction is needed across its items.
- **Across the two arms:** each arm is gated independently. The pseudo-holdout is used only for G3 pass/fail and is **never** used to choose between arms or parameters. If both arms pass G3, both are recommended and Phase 5 re-validates each one.
- **Across campaigns:** N includes every prior BTCUSDT trial, as above. The pseudo-holdout itself has had **no** prior look (§5.11).
- **G3** (look → recommend Phase 5 out-of-sample). Pseudo-holdout net E_r > 0, n_trades ≥ 11, account not blown. This is a recommendation only.
- **Failure rule.** If no arm passes G4b-N, the conclusion is "no evidence of net profitability for the full strategy under registered costs". No further tuning happens without a new pre-registration.

## 8. TDD and verification

### Tests first (all must fail before the code)

`tests/test_phase4b_*.py`:
- C1 validation;
- C2 oracles, covering aggregation exactness, diagonal pairs, the min-share filter, zero sides, edges and stacked runs;
- C3 `required` mode, the diagnostic-only `bonus` refused for phase4b, and that the bar footprint is never read when the trade footprint is required;
- the execution-level break-even/trail test (C3);
- the `full_strategy_params` call-site coverage (C6);
- the `neighbours_rel` oracle (C6);
- the C8b allocation refusals;
- the `SearchPrefix` holdout-access spy (§5.1);
- the phase-layout integration run and the `derive_ceilings` tests (C7);
- C4 integrity error on a bucket mismatch;
- C5 M1 and M2 never called (spy) under `absorption_source="trade"`;
- C6 bounds, categorical suggest and neighbours;
- C7 b54076c6 bytes and hash unchanged, the 4b namespace isolated, a refusal for a phase-mismatched approval, and stage single-use across S1 (`phase4b-s1` manifest) and S2t/S2v/S3 (`phase4b` manifest);
- C8 the net bootstrap reproducible by seed, (arm − comparator) of identical inputs exactly 0, each gate's refusal, and holdout-look refusal unless G4b-N passes;
- C9 search-segment-only reads.

### Proof commands (host runs; real data; read-only on protected paths)

```
source .venv_orderflow/bin/activate
pytest -q                                   # expect: all pass (baseline 1444 passed, 1 skipped + new)
pytest -q -m real_data                      # expect: real-data suite passes
python scripts/phase4b_regression.py --stageA output/phase4/ablation --stageB output/phase4/frozen \
       --check-digests                      # AC1: 36 + 8 digests byte-identical (search segment)
python scripts/phase4b_regression.py --protected   # AC5/AC7: hashes of protected files, 3 legacy DBs,
                                            # phase4 DB, campaign b54076c6 + seal, cache COMPLETE, 4 builder files
```

- **Old/new comparison:** the same inputs are run through 95bde47 and the new HEAD with default config. The digests must be equal.
- **Real-output proof:** on one week of the search segment from the certified cache, `absorption_source="trade"` produces zero M1 signals. Every emitted ABSORPTION equals the output of the unchanged `_check_trade_absorption` helper evaluated **independently on every eligible bar**: signal for signal, the same direction, level and strength (rev 3, P4B-11). Previously emitted A2 signals are not the oracle, because the old path returns M1 first. A synthetic overlap case covers it: open 10009, close 10005, low 10000, high 10010, +80 % delta, and a $10010 level with ask/bid 90/10. The old path emits M1; the new path emits the A2 SELL that the independent oracle predicts.

### Inspection

A fresh Claude CLI inspection is run against pre-build commit `95bde47`. Codex adversarial re-review covers the C7/C8 approval and gate code.

## 9. Risks

- **Dimensions vs budget.** 23 dimensions with N = 200 is thin for TPE. Under-search lowers the best and over-search raises PBO, and DSR penalises the extra trials. The budget is D6.
- **Pseudo-holdout is shared with Phase 4 Stage C.** Whichever looks first consumes independence. Allocation is D7.
- **Runtime.** A2 heavy days ran 4.7–5.6× bar-only, and aggregated imbalance adds per-bar level work. S1 measures this, and the S2 ceilings follow the pre-registered `derive_ceilings` formula (C7). A ceiling breach stops the stage durably.
- **Bucket aggregation performance.** It runs per bar, per trial. It is memoised per (bar, mult) only inside one replay, with no cross-trial cache, so there is no new persistent artifact.
- **DSR threshold is still demanding (rev 8).** It requires SR̂ ≥ 5.08·σ_null on the search segment: at least 2.67 annualised, about 2.84 at a typical dependence profile, and higher for strongly autocorrelated candidates.
  - The tail check adds a further, candidate-specific hurdle for strongly dependent return series.
  - The only family-wise claim is FWER ≤ 5 % by the union bound. It rests on bootstrap validity in the extreme tail (§7.2 assumptions).
  - Any further change is a new pre-registration made **before** S2, never after results are seen.
- **Structural cost.** Even a real edge may not clear ≈ 12 bps round trip. The plan reports break-even slippage and fee so the owner can judge execution assumptions separately; it never relaxes them.
- **The `count_consecutive_imbalances` (cross-bar) semantics are unused and kept.** Phase 4b defines stacked as within-bar, the standard meaning. The old function is not wired, to avoid two meanings in production.

## 10. Owner decisions

On 2026-11-07 the owner **provisionally accepted D2–D8 as recommended**, including 300 trials per study and prioritising Phase 4b over Phase 4 Stage C. Implementation, Optuna and holdout access still need explicit authorization.

| # | Decision | Recommendation | Why it matters / cost of guessing wrong |
|---|---|---|---|
| D1 | Absorption source | `trade`, replacing M1. **Owner-stated, resolved** | — |
| D2 | Imbalance geometry for initiative and stacked | Diagonal, with the $1 → k aggregation tuned | Horizontal at $1 flags single prints as imbalances; picking wrongly mislabels the core signal |
| D3 | Fixed modes: E2, initiative imbalance, stacked role | E2 `required`, which exits only if the contrarian is present, so it is economically active. Initiative imbalance `required`, with the stacked run k ∈ 1–4 tuned. `bonus` is excluded because it cannot change trades (P4B-01) | Fixing these ex ante keeps one arm per bar type; tuning modes as categoricals adds multiplicity |
| D4 | Initiative in the entry score? | No. Keep entry = trade absorption at a qualified trade-profile level; initiative manages break-even/trail | This matches the upstream execution model; adding it changes the strategy definition |
| D5 | Net-gate strictness | All G4b-N items required, including net lower CI > 0 and DSR ≥ 0.95. **Rev 8:** the DSR is defined with a dependence-robust, same-basis no-skill spread (owner approved R1–R4 on 2026-11-07) | Strict gates protect capital but may reject a marginal real edge |
| D6 | Budget per study | **300 trials (accepted)**, ≈ 19–22 h each, ≈ 40 h total sequential | 23 dimensions; DSR multiplicity grows with N |
| D7 | Pseudo-holdout allocation | Phase 4b gets the single look; Phase 4 Stage C becomes attribution-only with no look | Two campaigns looking at one holdout erodes its independence |
| D8 | Phase 4 Stage C | Defer or skip; it remains registered and unmodified | Saves about 15 h and avoids holdout competition |

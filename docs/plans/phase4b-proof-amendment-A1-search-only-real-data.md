# Phase 4b proof-procedure amendment A1: search-only real-data proof

**Status:** OWNER-APPROVED, without independent Codex review. On 2026-11-08 the owner chose to skip Codex review and recorded that choice through exception A2. **No independent reviewer has checked it.**

**Amends:** plan rev 13 (`docs/plans/phase4b-full-strategy-plan-rev13.md`, sha256 `75b22452c0b168f43fd91704a7aee23a4cb1c84c637f4a8f775b87be13857b7f`), §8 "Proof commands", two lines:
- `pytest -q` (the U gate) becomes `pytest -q -m "not real_data"`. With the volume mounted, a plain `pytest -q` would also run the holdout-reading real-data tests (interim review, Critical 1).
- `pytest -q -m real_data` becomes the guarded command in §6.

The rev 10–13 plan bytes are unchanged. The gate values (U, AC1, AC7, R) and their binding are unchanged, except that the unit and real_data artifacts are now **content-checked** (§7). That only tightens the gate. The R rules for S1 are changed only by exception A2.

**Owner decisions (2026-11-08):** "Choose Option 1: search-only real-data proof… Do not silently change or weaken the approved gate." Then: "Implement A1 test-first, enforce search-only real-data access, preserve G1–G3 coverage-gap documentation."

## 1. Problem

The `real_data` artifact required by `verify_phase4b_gates` comes from §8's `pytest -q -m real_data`. That suite reads bars and trades on dates inside the Phase 4b pseudo-holdout. It would expose holdout-period market data before the single gated look (D7).

## 2. Horizon

| Bar type | `n_bars_search` | Last search bar close (UTC) |
|---|---|---|
| `fixed_tick_imbalance` | 1 871 694 | 2023-05-09T19:14:04.024Z |
| `fixed_volume_imbalance` | 1 597 471 | 2023-04-29T03:32:55.751Z |

- The values were read from the Stage B caches with `CachedBars(stop=n_bars_search)`, so only search bars were read.
- **H = 2023-04-29T03:32:55.751Z**, the earlier of the two. It applies to every bar type and every read.
- The enforced limit is the preceding midnight: a reader's exclusive end must be at or before **2023-04-29T00:00Z**.
- The constants are in `scripts/phase4b_proof.py`.

## 3. Inventory (static audit of `tests/` at 95cca15)

Only three files carry `@pytest.mark.real_data` (20 tests). The unmarked tests that mention real-data paths use synthetic fixtures or path-refusal checks.

| # | Test | Read | Disposition |
|---|---|---|---|
| 1 | `test_real_dataset.py::test_pipeline_on_real_bars` ×6 | the 2023-11-19..24 window, **plus** `validate()` → `check_start_independence`, which ignores the window and reads **unbounded** up to the first dataset outage (after H) | Deselect; replace with A1-1; gap G1 |
| 2 | `test_real_dataset.py::test_dollar_imbalance_direction_and_real_pattern_exit_regressions` | 2023-01-01..03-16 | Keep |
| 3 | `test_real_dataset.py::test_tick_run_real_distant_absorption_is_not_entered` | 2023-11-19..24 | Deselect; replace with A1-2; gap G2 |
| 4 | `test_optimize_orderflow.py::test_real_prefix_and_notebook_parity` | 2023-06-01..04 | Deselect; replace with A1-3 (equivalent) |
| 5 | `test_optimize_orderflow.py::test_real_folds_match_reference` | the old dollar-imbalance dev cache, sliced at that cache's own 80 % mark, which is not tied to H | Deselect; gap G4 |
| 6 | `test_phase4_real.py::test_certified_production_builder_fingerprint_and_open` ×6 | `CachedTradeUpdates(production=True)`: a full-file sha256 **and decoding of every bar and day for integrity** (`validate_bars(self.bars())`, the `_days` index, `trade_digest()` over all 2019–2023 bars, which includes the pseudo-holdout). Nothing is output or used for selection. | Keep, disclosed. This is exactly the integrity validation each stage performs when it opens the production cache (rev 12 F3, §5.1), so excluding it would not reduce S1's own exposure. |
| 7 | `test_phase4_real.py::test_bounded_real_cache_and_live[2021-05-19-2021-05-20]` | before H | Keep |
| 8 | `test_phase4_real.py::test_bounded_real_cache_and_live[2019-09-11-2019-09-12]` | the start edge | Keep |
| 9 | `test_phase4_real.py::test_bounded_real_cache_and_live[2023-10-16-2023-10-17]` | after H | Deselect; replace with A1-4 (equivalent) |
| 10 | `test_phase4_real.py::test_bounded_real_cache_and_live[2023-12-30-2024-01-01]` | the end-of-history edge (exclusive end, so no 2024 bytes) | Deselect; gap G3 |

## 4. Replacements

The replacements are in `tests/test_phase4b_real_search_only.py`. No existing test is edited.

| ID | Replaces | Window | Assertions |
|---|---|---|---|
| A1-1 | #1 | 2023-03-19..24, all 6 types | `validate()`'s checks except start independence: ledger valid, completed, no rejected rows, deterministic, no look-ahead prefix, no fabricated levels, `bars > 500`. They are composed from `validate_infobars._replay`, so the unbounded start-independence read is never made. |
| A1-2 | #3 | tick run 2023-03-19..24 | entries exist, and each is within 0.2 % of its qualified level |
| A1-3 | #4 | dollar imbalance 2023-03-01..04 | `assert_prefix_parity` (unchanged helper) |
| A1-4 | #9 | 2023-04-17..18 | `test_bounded_real_cache_and_live(start, end)`, called unchanged |

## 5. Coverage gaps

- **G1. Start independence across a real outage, and the real `data_gap` path.**
  - None of the 5 missing tick hours, in any bar type, falls before H.
  - Remaining synthetic coverage: `tests/test_pipeline.py` gap tests, `tests/test_infobar_feed.py` and `tests/test_initiative.py::test_post_gap_initiative_compares_against_prior_known_rates_only`.
- **G2. The reviewed tick-run bar `seq 4304211644` (Nov 2023).**
  - Remaining: the proximity invariant on a search window, and `tests/test_trade_lifecycle.py` (entry at the qualified level).
- **G3. The real end-of-history edge (2023-12-30..31).**
  - Remaining: `excluded_by_policy == 1` and the forbidden-open interception, asserted on the real manifest in #7, #8 and A1-4; and the synthetic full-stream regression P4-IR-04 in `tests/test_phase4_cache.py`.
- **G4. Real-bar fold-reference equivalence.**
  - Remaining: the synthetic `assert_same_folds(build_chronological_folds, reference_purgedkfold_folds)` cases in `tests/test_optimize_orderflow.py`.

All four checks can still run after the single look.

## 6. Proof command (replaces only §8's `pytest -q -m real_data`)

The command is generated by `scripts/phase4b_proof.real_data_selection()`:

```bash
pytest -q -p tests.phase4b_search_guard -m real_data \
  --deselect tests/test_real_dataset.py::test_pipeline_on_real_bars \
  --deselect tests/test_real_dataset.py::test_tick_run_real_distant_absorption_is_not_entered \
  --deselect tests/test_optimize_orderflow.py::test_real_prefix_and_notebook_parity \
  --deselect tests/test_optimize_orderflow.py::test_real_folds_match_reference \
  --deselect "tests/test_phase4_real.py::test_bounded_real_cache_and_live[2023-10-16-2023-10-17]" \
  --deselect "tests/test_phase4_real.py::test_bounded_real_cache_and_live[2023-12-30-2024-01-01]"
```

It collects 18 tests: 9 retained (#2, #6 ×6, #7, #8) and 9 replacements.

## 7. Enforcement

- **Runtime plugin `tests/phase4b_search_guard.py`.** It is loaded by the command, so it applies to all 18 tests.
  - It replaces `InfoBarReader` in every loaded module with a subclass that refuses an unbounded reader, or one ending after 2023-04-29T00:00Z, **before** construction.
  - It refuses a bounded trade-cache build whose end is after that date.
  - **Not intercepted**, disclosed:
    - (a) #6's integrity decoding of the production cache; see the table.
    - (b) `PairedDataset.open` (#7, #8, A1-4) hashes every listed artifact and loads the **hours index, hourly trade counts through 2023-12-31**, for its count check (`paired_dataset.py`, `trade_stream.py`). These are aggregate counts, not prices, and are not used for selection.
    - (c) Trade-file queries inside an admitted bounded build. They are bounded by the guarded reader window. Their own forbidden-file interception cuts off at 2024-01-01, not at H.
    - (d) Admitted readers decode whole parquet row groups that straddle `end_ms`. Bounds are enforced per row, not per byte. The latest exclusive end is 2023-04-18 (A1-4). **Not verified:** whether any row group straddling an admitted end reaches H. The row-group spans were not inspected.
- **Accidental runs are prevented.**
  - `tests/conftest.py` skips the six deselected holdout-reading nodes unless `PHASE4B_ALLOW_HOLDOUT_TESTS=1` is set (only after the single look).
  - The replacement module skips itself unless the guard plugin is loaded.
- **Guards** in `tests/test_phase4b_proof_guards.py`. They are non-real and run in the normal suite.
  - G-static: every replacement window ends at or before 2023-04-29.
  - G-collect:
    - every deselected node exists, and none remains in the selection;
    - the selection collects exactly 18 nodes;
    - no collected test's body or parameters state a date after 2023-04-29.
  - Plugin tests:
    - unbounded and late readers and builds are refused;
    - the limit is midnight 2023-04-29 UTC;
    - the selection loads the plugin.
  - Mutation check: moving A1-4 to 2023-04-30 makes two guards fail.
- **Artifacts, enforced at the gate.** `scripts/phase4b_proof.write_artifact` runs the commands and writes JSON artifacts.
  - `verify_phase4b_gates` now checks their content on **both** R paths:
    - the `unit` argv equals `proof_argv('unit')`, ignoring the interpreter path, with 0 failed and 0 errors;
    - the `real_data` argv ends with `real_data_selection()`;
    - exactly 18 unique collected nodes, none of them deselected;
    - H matches;
    - this document's sha256 matches the pinned path;
    - passed = 18 with 0 failed, 0 skipped and 0 errors.
  - This only tightens the approved gate.
- **Writes.** #7, #8 and A1-4 write bounded caches under the checkout's `output/phase4/real/<start>` (as #7 and #8 already did).

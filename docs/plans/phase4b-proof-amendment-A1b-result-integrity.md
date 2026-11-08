# Phase 4b proof-procedure amendment A1b: proof-result integrity at the gate

**Status:** owner decision, 2026-11-08. Implemented test-first; reviewed only by the interim Superpowers (Claude) review. **No independent reviewer has checked it.**

**Owner instruction:** "Before installing approval records, fix the remaining proof-gate issue using focused TDD: require successful test-process exit and confirmation that every collected test passed. Reject missing or malformed results. Preserve A1, A1a, A2, frozen plans, campaign artifacts and holdout protections. Document any required amendment."

**Amends:** A1 §7 (`58b0598d…`) and A1a (`docs/plans/phase4b-proof-amendment-A1a-unit-gate.md`, `a9f04a3f…`), for the `unit` and `real_data` artifact checks only.
- The bytes of A1, A1a, A2 and plans rev 10–13 are unchanged.
- The approved gates are unchanged: U, AC1, AC7 and R, their binding, and the A2 exception.

## Problem

The interim review of A1a found a gap. `scripts/phase4b_proof._counts` sets any count to 0 when its word is missing from the last line of output, and the gate checked neither the exit code nor that the passed count matched the collected count.

So an aborted or unparsable run, such as a collection error, an interruption or a crash before the summary line, could produce an all-zero artifact that satisfied every count check.

## Change (`scripts/phase4_approvals._a1b_results`; runs for both artifacts and on both R paths)

The checks run in this order, before the existing A1 and A1a checks:

1. **Well-formed results.**
   - `returncode`, `passed`, `failed`, `errors` and `skipped` must all be JSON integers. A value that is missing, `null`, a boolean, a float or a string is refused as malformed.
   - The collected count must also be an integer: `collected_count` for `unit`, and the length of the `collected` list for `real_data`.
2. **Successful test-process exit:** `returncode == 0`.
3. **Every collected test passed:**
   - `passed == collected > 0`;
   - `failed == errors == skipped == 0`.
4. **The summary agrees.** The recorded pytest `summary` must be a string containing "`<collected> passed`". This catches counts that came from defaults rather than from pytest's own result line.

The existing checks still apply after these:
- the argv;
- the A1a host flag;
- the exact 18 nodes for real_data;
- the horizon;
- the A1 hash.

## Effect

- **Tightening only.** Nothing previously refused is accepted.
- **Strict by design, and fails closed.** Two legitimate kinds of run would be refused:
  - a suite containing an `xfail` or `xpass` test, because it is collected but not counted as passed;
  - a run whose collected list disagrees with its own result.

  Neither is ever wrongly accepted.
- **Current proofs still pass.** Artifacts written by `write_artifact` from `686ff4b` onward record all of these fields and pass: unit 1797/1797 with exit 0; real_data 18/18 with exit 0.

## Tests

`tests/test_phase4b_s1_exception.py` refuses each of these:
- a non-zero exit, or an exit code that is null, boolean or missing;
- fewer passed than collected;
- zero collected;
- a non-integer `collected_count`, `passed`, `skipped` or `failed`;
- a summary that is interrupted, null, missing or mismatched;
- a missing `passed` count;
- a failed unit exit on the A2 S1 exception path.

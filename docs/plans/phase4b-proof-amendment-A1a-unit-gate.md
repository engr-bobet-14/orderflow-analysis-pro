# Phase 4b proof-procedure amendment A1a: unit-proof gate enforcement

**Status:** owner decision, 2026-11-08. Implemented test-first; reviewed only by the interim Superpowers (Claude) review. **No independent reviewer has checked it.**

**Owner instruction:** "Before authorizing S1, fix the remaining unit-proof gate enforcement gap using TDD. 1. Require the gate itself to verify zero skipped unit tests and the mandatory host-input proof flag. 2. Document the narrow A1 amendment without modifying frozen rev 10–13 plans."

**Amends:** amendment A1 (`docs/plans/phase4b-proof-amendment-A1-search-only-real-data.md`, sha256 `58b0598d…`), §7 "Artifacts, enforced at the gate", the `unit` bullet only.
- The A1 bytes are unchanged, and the real_data artifact still pins them.
- The bytes of plans rev 10–13 are unchanged.
- The approved gates are unchanged: U, AC1, AC7 and R, their binding, and the A2 exception.

## Change

| | A1 (before) | A1a (now) |
|---|---|---|
| `unit` argv | equals `proof_argv('unit')`, ignoring the interpreter | unchanged |
| `unit` counts | `failed == 0` and `errors == 0` | also requires **`skipped == 0`** (a missing count is refused) |
| `unit` host flag | not checked | requires **`require_host` to be `true`** (JSON boolean; anything else, or a missing value, is refused) |

`scripts/phase4b_proof.write_artifact` (commit `686ff4b`) runs both proof commands with:
- `PHASE4B_REQUIRE_HOST=1`;
- `PHASE4B_ALLOW_HOLDOUT_TESTS` removed.

It records `require_host: true` in each artifact. With the flag set, a host-input test (for example the real-manifest S1 preflight regression in `tests/test_phase4b_protected_preflight.py`) **fails** rather than skips when its host inputs are missing.

## Effect

- **Before A1a**, only the proof writer enforced the requirement. A unit artifact that recorded skips, or that had no host flag, still passed `scripts/phase4_approvals.verify_proof_artifacts`.
- **Now** the gate refuses it on both R paths (`pass` and the A2 S1 exception). That includes every unit artifact written before `686ff4b`.
- **This only tightens the gate.**
  - Nothing that was previously refused becomes accepted.
  - The real_data checks are unchanged; they already require 0 skipped.

## Tests

`tests/test_phase4b_s1_exception.py`:
- `test_a1_gates_check_proof_artifact_content`: `skipped` of 1 or `null`, and `require_host` of `false`, `"true"` or `null`, are each refused.
- `test_a1a_unit_artifact_without_skip_count_or_host_flag_is_refused`: an artifact missing either key is refused.

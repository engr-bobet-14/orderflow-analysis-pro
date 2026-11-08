# Phase 4b exception A2: S1 calibration on the interim review (owner exception to gate R)

**Status:** owner decision, 2026-11-08. **Not an independent review.** This document defines the exception and is its policy text; its sha256 is bound into every exception record.

**Owner instruction:** "I want to skip Codex review to save time. Prepare a narrowly scoped, auditable exception allowing S1 calibration using the existing Superpowers interim review. Do not falsely mark independent inspection R as passed… Do not weaken other gates, alter frozen plan bytes, access holdouts, or modify the sealed Phase 4 campaign… Keep Optuna blocked pending a separate decision."

**Amends:** plan rev 13 (sha256 `75b22452…`), §8 "Inspection" and gate R, **for the `phase4b-s1` manifest (stage S1) only.** The rev 10–13 plan bytes are unchanged.

## 1. What changes

`scripts/phase4_approvals.verify_phase4b_gates` accepts exactly one alternative to `R = "pass"`:

| | `R = "pass"` (unchanged) | `R = "owner_exception_interim_review"` (new) |
|---|---|---|
| Allowed manifests and stages | `phase4b-s1`, `phase4b` | **`phase4b-s1` manifest and stage `S1` only**: `verify_phase4b_gates(layout, campaign, stage)` refuses it for any other stage or for `stage=None` |
| Required artifact roles (exactly) | unit, digests, protected, real_data, **inspection** | unit, digests, protected, real_data, **interim_review**, **owner_exception** |
| U, AC1, AC7 | must be `pass` | must be `pass` (unchanged) |
| Plan, campaign, source-state binding | unchanged | unchanged |

- **R is never recorded as `"pass"` under this exception.** A gates record with `R = "pass"` cannot be backed by the interim review: its role set would lack `inspection` and fail.
- Any other R value is refused.
- The `phase4b` manifest (S2t, S2v, S3) refuses the exception value. **Optuna (S2) and the look (S3) therefore still need an independent inspection with `R = "pass"`.** This document does not unblock them.

## 2. The owner exception record

- **Installed** by `install_inspection_exception(layout, record)`, host-only and after the owner's explicit decision. It goes under `<approval root>/exceptions/<sha256>.json`: content-addressed, mode 0600, created with `O_EXCL`.
- **Verification refuses** a record or exceptions directory with any group/other permission bits, and a `created_utc` that is not an ISO UTC timestamp.
- **Fields (exactly):**
  - `kind` = `inspection_exception`;
  - `phase` = `phase4b-s1`;
  - `stage` = `S1`;
  - `campaign_sha256` (the S1 manifest);
  - `plan_sha256` (rev 13);
  - `source_state` (exact code state);
  - `amendment_sha256` (this file);
  - `interim_review_sha256` (the `interim_review` artifact);
  - `owner_statement`;
  - `created_utc`.
- **Verification** (`verify_inspection_exception`) runs on every S1 admission and recovery.
  - The record must be private, uid-owned, canonical, content-addressed and inside the exception root.
  - Every binding must match the current state: the S1 manifest, rev 13, `source_state()`, this document's current bytes, and the `interim_review` artifact's hash.
- **Any code change** (a different `source_state`) invalidates the record, so a new record and a new owner decision are required.

## 3. The interim-review artifact

The interim review (Superpowers, fresh Claude subagents, Claude-built code) is recorded as an artifact that lists:
- the review rounds and their verdicts (fix round 3: No → With fixes → Yes; pre-inspection: With fixes → Yes; zombie-sample fix: Yes; A1/A2 changes: see the review log);
- each finding and its disposition;
- the explicit statement "interim; not an independent cross-provider inspection".

## 4. Scope and effects

- **Allowed:** S1 calibration only. That means deterministic replays and the benchmark on the search segment, with the registered S1 ceilings.
- **Not allowed:** S2t/S2v (Optuna) and S3 (the look); recovery of those stages; any holdout access.
- **Audit trail.** Every stage record, and the campaign-log line written at stage start, carries `inspection_basis` = {`R`, `owner_exception_sha256`, `interim_review_sha256`}. This is copied at admission, and recoveries carry it through their admitted marker.
  - So the record of S1 having run under this exception survives even if `gates-<sha>.json` is later replaced, for example by an `R = "pass"` record after an independent inspection.
  - The interim-review artifact also embeds copies of A1 and A2, because `docs/` is gitignored until they are committed.
- **Downstream.** The S1 benchmark feeds the S2 ceilings, but S2 registration and admission need the `phase4b` gates record with `R = "pass"`.
  - If an independent inspection later requires code changes, the S1 benchmark no longer matches the runtime fingerprint (F2), so S1 must be rerun.
  - If the inspection passes without changes, the S1 run record still shows it ran under this exception, through its gates sha (R value).

## 5. Risks accepted by the owner

- S1 runs on code that only Claude has built and reviewed, so there is no cross-provider check of the rev 13 lifecycle, the exception code or A1.
- A defect that the independent inspection would have caught could waste the S1 compute (up to 4 h) or produce wrong runtime ceilings. That costs time, not holdout exposure, because S1 reads only the search segment.
- A1 (search-only real-data proof) is likewise owner-approved without independent review.

## 6. Tests

`tests/test_phase4b_s1_exception.py` (31 tests) covers the following, plus the interim-review fixes (explicit stage S1, artifact content checks, the inspection basis in the stage record and log, a private exceptions directory, UTC validation):
- acceptance for S1, with R recorded as not `pass`;
- refusal for the `phase4b` manifest;
- the `R = "pass"` path unchanged;
- `pass` cannot be backed by the interim review;
- any other R value is refused;
- each binding mismatch;
- privacy, `O_EXCL`, the exception root, and changes to the policy document;
- U, AC1 and AC7 are still required.

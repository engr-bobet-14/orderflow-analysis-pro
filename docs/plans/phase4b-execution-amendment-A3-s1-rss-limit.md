# Phase 4b execution amendment A3: S1-only 12 GiB RSS limit

**Status:** owner decision, 2026-11-08. Implemented test-first; reviewed only by the interim Superpowers (Claude) review. **No independent reviewer has checked it.**

**Owner instruction:** "I have 48 GB RAM and authorize increasing the Phase 4b S1 memory limit. Instead of immediately refactoring the two-arm memory handling: investigate whether the RSS spike is bounded cache residency or an actual memory leak; propose a 12 GiB RSS limit for S1, subject to host-memory and concurrent-process checks; verify the new limit is compatible with the approved execution policy and document any required amendment; keep all other runtime limits, integrity gates and holdout protections unchanged."

**Amends:** plan rev 13 (`75b22452…`), §"Execution". Rev 13 states: "All runs are sequential, one process at a time. The existing safeguards are unchanged: strict RSS and private memory ≤ 4 GiB…".
- The rule changes **for stage S1 only**: the RSS limit becomes 12 GiB.
- Private memory stays ≤ 4 GiB for S1. RSS and private memory both stay ≤ 4 GiB for S2t, S2v, S3 and all Phase 4 code paths.
- The bytes of plans rev 10–13, A1, A1a, A1b and A2 are unchanged.

## 1. Evidence: bounded cache residency, not a leak

**S1 run `42d5c2da`** (manifest `98e29396…`, consumed):
- Stopped by `strict_memory_stop`: peak RSS 4.15 GiB against the 4 GiB limit; peak private (`phys_footprint`) 0.46 GiB.
- Worker RSS held at about 2.4–2.6 GiB through all 10 tick-arm replays (184–246 s each). It rose only after the last tick replay, while the volume arm was opening.
- The watchdog's last private sample (0.40 GiB) was below its peak (0.46 GiB). Private memory was not growing.

**Read-only probe** (`output/phase4b/evidence/s1-run-42d5c2da/memory-probe.json`). It replicates S1's `open_prefix` for both arms in one process, with no replays, through the same production cache open as AC1:

| Step | RSS GiB | vmmap mapped-file resident | Private GiB |
|---|---|---|---|
| baseline | 0.11 | 80 K | 0.07 |
| tick arm open | 2.31 | 2.2 G | 0.12 |
| volume arm open, tick alive | 4.17 (lifetime peak 4.38) | 4.0 G | 0.11 |
| tick released (`del` + `gc`) | 2.02 | 1.9 G | 0.11 |
| both released | 0.16 | 80 K | 0.11 |

- The production trade caches are 2.31 GB (tick) and 1.99 GB (volume). Each becomes fully resident through the integrity decode.
- That residency is **file-backed**: vmmap "mapped file". It is released in full on close, and private memory does not grow.
- The S1 spike is therefore the two arms' caches resident together. It is bounded by their sizes, and it is not a leak.

**Limitations:**
- No per-replay private series was persisted, so the no-leak conclusion for the replays rests on the RSS plateau and the falling end-of-run private sample.
- S2's 300-trial in-process leak behaviour is not assessed; A3 does not cover S2.

## 2. Configuration

| | Before | A3 |
|---|---|---|
| S1 RSS limit (stage-level `file_timing` memory watchdog, per-replay watchdog, arm-open point check, driver post-exit check) | 4 GiB | **12 GiB** (`phase4b_campaign.S1_RSS_LIMIT_BYTES`) |
| S1 private limit | 4 GiB | 4 GiB, unchanged |
| S2t, S2v, S3 RSS and private | 4 GiB | unchanged |
| Phase 4 (`phase4_campaign`, `phase4_benchmark`, `optimize_orderflow` callers without a stage limit) | 4 GiB | unchanged (default argument) |
| S1 host admission (driver, before any fresh **or recovery** approval is consumed) | none beyond the worker preflight's 8 GiB available | **≥ 20 GiB available** (12 GiB limit + the unchanged 8 GiB margin), via `vm_stat` free + inactive + speculative |
| Worker preflight, arm-open memory plan (24 GiB total, 8 GiB available) | unchanged | unchanged |
| Concurrent-process check (`assert_no_conflicts`, single stage lock, sequential arms) | unchanged | unchanged |
| Runtime limits (1800 s per replay, 14 400 s per stage), timeout classification, integrity gates, protected hashes, holdout guards, approvals machinery | unchanged | unchanged |

**Sizing:** the S1 upper bound is about 5.0 GiB:
- both trade caches resident, 4.0 GiB;
- the bar caches, 0.42 GiB;
- the replay working set, 0.46 GiB private;
- the interpreter, about 0.1 GiB.

The observed peaks were 4.15 GiB (S1) and 4.38 GiB (probe). 12 GiB leaves about 2.4× headroom over the bound and is 25 % of host RAM (48 GiB).

- A real leak would still be stopped by the unchanged 4 GiB private limit.
- A bounded excess beyond 12 GiB would still be stopped by the RSS limit.

**Host check, 2026-11-08:** 22.0 GiB available; macOS reports 89 % free.

## 3. Mechanism

- `phase4_resources.memory_bounds`, `enforce_memory` and `enforce_current` take an `rss_limit` defaulting to `PRIVATE_BUDGET`. `memory_plan` and `require_memory_plan` take a `min_available` defaulting to `MIN_AVAILABLE`.
- `optimize_orderflow.phase4_trial_timing` reads the optional `args.phase4_rss_limit_bytes`; without it, the limit is 4 GiB.
- `phase4b_campaign`:
  - `stage_rss_limit(stage)` returns 12 GiB only for `'S1'`;
  - `stage_min_available(stage)` returns 20 GiB only for `'S1'`;
  - `enforce_stage_memory` is the driver's post-exit check;
  - `file_timing(..., rss_limit=stage_rss_limit(stage))` covers the stage-level memory watchdog and the replay guards. The time-only `stage_watchdog` is unchanged.
  - `open_prefix(..., stage)`: only S1 passes a stage, and the S2 and S3 call sites are unchanged. The point check `enforce_current` now runs at the start of `open_prefix`, not just before `SearchPrefix.open`.
    - This applies to S2 and S3 too, at their unchanged 4 GiB.
    - The effect is negligible: it checks the lifetime peak, and only light reads come before the open.
- The S1 driver checks `require_memory_plan(..., stage_min_available('S1'))` after the stage locks are taken and **before** either the recovery branch or `require_phase4b_approval(consume=True)`. A host short of memory is refused without spending a fresh or recovery approval, and no `.admitted.json` is written. This was interim-review finding I1.
  - A recovery that only needs the post-flight check also requires 20 GiB. That is stricter than needed, but safe.
  - The host check runs once, at admission. On a busy host (22.0 GiB measured, against the 20 GiB gate), expect refusals rather than mid-run stops.
- The limit is a code constant. The manifest binds it through `source_state`, and every approval, gates record and exception record binds `source_state`.

## 4. Tests

`tests/test_phase4b_s1_memory_limit.py` (14 tests) covers:
- the defaults are unchanged;
- 12 GiB admits the observed S1 residency, refuses 12 GiB + 1, and still refuses private > 4 GiB and an unavailable private measurement;
- the limit is S1-only (S2t, S2v, S3 and None keep 4 GiB);
- `min_available` works as a parameter;
- the live watchdog uses the stage limit (4.38 GiB passes at 12 GiB and is fatal at the default);
- S1 replay guards and arm opens use 12 GiB;
- the arm-open point check comes first;
- every production `file_timing` call passes `stage_rss_limit(stage)`;
- the post-exit check uses the stage limit;
- the driver refuses S1 with 19 GiB available **before** consuming the approval, with the admission error text, and admits it with 21 GiB;
- the same holds on the recovery path: no recovery admission with 19 GiB available;
- S2t still has no driver-side memory admission: it is admitted with 1 GiB available, and its worker preflight keeps the 8 GiB check.
- The S1 recovery refusal writes no `.admitted.json`.

One existing mock was widened, with no assertion changed: `test_i4_driver_memory_breach_after_rc0_is_strict_memory_stop`'s `breach(measurement)` became `breach(measurement, *a)`, because `enforce_memory` now receives the stage limit.

## Interim review

The interim Superpowers review (Claude, not independent) returned **With fixes**:
- **I1:** the recovery path skipped the 20 GiB admission. Fixed test-first, with red evidence in `red-a3/red-i1.txt`.
- **Minor findings fixed:** the stricter AST check, the admission error text, the S2t case, and the exact `is None` fallback for `phase4_rss_limit_bytes`.
- **M1 and M4:** documented above.
- **Round 2:** Ready: Yes. Minors m1 and m2 (test hardening) were fixed; m3 and m4 are documented.

## 5. Re-registration requirements (nothing here is done yet)

The consumed S1 manifest `98e29396…`, its approvals and its records are **not** altered. A new S1 attempt needs, in order:

1. **Owner authorization** to commit A3 (code, tests, this doc) to PR #15.
2. **Proofs re-run on the A3 commit**, because the source fingerprint changes:
   - U (full suite);
   - guarded real_data;
   - AC7;
   - **AC1, re-run.** A3 changes `optimize_orderflow.py`, which AC1's replay path imports.
   - `optimize_orderflow.py` is also part of `phase4_source_fp`, so any *new* Phase 4 or 4b study identity hash differs.
     - This is harmless: Phase 4 is sealed at its own worktree (`95bde47`), and S2 is not registered.
     - The consumed S1 records keep their old binding.
3. **A new S1 registration** from the same registration input (only `source_state` changes), after the 98e29396 manifest is archived as superseded and consumed. The new manifest gets its own checkpoint namespace, so the consumed attempt's records stay untouched.
4. **A new interim-review record and A2 owner exception** bound to the new manifest and `source_state`, plus a new gates record. This needs owner approval for step 5 again.
5. **New single-use driver and worker approvals.** This needs explicit owner authorization.
6. **Owner authorization to run S1**, with at least 20 GiB available at admission.

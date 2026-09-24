# Research review record: data processing and attacker study

Updated 2026-09-21. This page replaces the executor-only draft of 2026-09-18.

## Rounds

| Run | Date | Reviewer | Scope | Outcome | Trace |
|---|---|---|---|---|---|
| 01 | 2026-09-18 | Codex gpt-5.6-sol, ultra | Extraction, JSD metric, benchmark | Failed (usage limit) | `.aris/traces/research-review/2026-09-18_run01` |
| 02 | 2026-09-18 | Codex gpt-5.6-luna, xhigh | Sampling and smoothing | Failed (usage limit) | `.aris/traces/research-review/2026-09-18_run02` |
| 03 r1 | 2026-09-21 | Codex gpt-5.6-sol, **high** (owner's choice) | Sampling and smoothing | Done | `.aris/traces/research-review/2026-09-21_run03/round1.md` |
| 03 r2 | 2026-09-21 | same thread | Attacker baseline and comparison analysis; D2 follow-up | Done | `.aris/traces/research-review/2026-09-21_run03/round2.md` |
| 03 r3 | 2026-09-21 | same thread | Check of the mending plan for part 1 (reference implementation + tests) | Done: one real bug and several weak points found, all fixed | `.aris/traces/research-review/2026-09-21_run03/round3.md` |
| 03 r4 | 2026-09-21 | same thread | Closing round on the fixes | NOT RUN (usage limit, resets 15:28); brief ready in `review/RESEARCH_REVIEW_ROUND_4.md` | `.aris/traces/research-review/2026-09-21_run03/round4.md` |

Codex thread: `01a0c314-85c9-7a30-918b-db863338b2fd`.

## What was agreed

**Sampling and smoothing**

- **The central defect is confirmed.** The 100-point arrays are resampled by row index, so they carry no valid time step.
- **TSGM noise (D2)** was upgraded to correct after the TSGM 0.1.0 source was read. The one remaining caveat is that the repo does not pin the TSGM version.
- **The executor's draft was corrected on four points:**
  - the S4 "475 m" figure measured shifts along the track, not damage to the route's shape;
  - the gap-split threshold goes from 120 s to 60 s;
  - generator arrays (time grid) and scenario routes (arc-length grid) need separate chains;
  - the residual reading was backwards.
- The reviewer's parameter values were adopted.
- **Four new defects:** NaN points bridging gaps, a global-mean projection, v3 straightening the whole route, and a probe that re-implements code rather than running it.
- **Deliverable:** `review/SAMPLING_SMOOTHING_FEEDBACK.md`.

**Attacker study**

- "Not defensible as it stands" is **confirmed**.
- **New and verified:** PPO zeroes the return at mid-episode buffer boundaries.
- **Withdrawn:** the truncation defect. `unresolved` is a legitimate terminal failure.
- **Softened:**
  - "RL adds nothing" becomes "the benchmark mostly saturates";
  - "the scripted baselines are the wrong kind" becomes "they are sanity references only";
  - "scorecard leakage" becomes "likely, but unverifiable, because the scorecard is not reproducible".
- **The plan is reduced to:**
  - the two existing reactive controllers as systems under test;
  - random search, CEM/CMA-ES and PPO, all on the same fixed encounters with the same total budget;
  - held-out parameter intervals and 10 seeds;
  - collision and domain violation reported separately;
  - counterfactual replay as a secondary analysis with three labels.
- **Dropped from the minimum paper:** the new velocity-obstacle controller, adaptive stress testing, a learned controller under test, and the Fossen transfer.
- **Deliverable:** `review/STUDY_PLAN_PEER_REVIEW.md`.

**Earlier findings (2026-09-18, executor, not externally reviewed)**

- `Evaluation/jsd_matrix.ipynb` compares each trace with a Normal distribution fitted to that same trace, so it cannot rank generators.
- `review/jsd_probe.py`: a random walk scores better than a clean trace.

The external rounds did not re-check this. Treat it as executor evidence.

**Mending plan for part 1, checked (rounds 3 and 4)**

- The agreed chain was built (`data_prep/ais_prep.py`) and tested on synthetic voyages with a known truth (`data_prep/check_out/results.md`).
- Old chain: one array point stands for 13 to 218 s, speeds read from the array are 0.34 to 1.60 of the truth. Mended: 20 s per point, speeds within 2 %, position error 3.6 / 7.7 / 12.7 m.
- Seven plan items changed: split before any estimate, gap rule per voyage, distance-form outlier rule with runs and cuts, bounded choice of the smoothing strength, COG as an opt-in with three tests, route as a C2 spline with a table, yaw limit and scale from the scenario standard.
- Round 3 found a real bug (the route acceptance read knot curvature, the played curve peaked up to 3 times higher) and leakage, guard and privacy weaknesses; all fixed and re-tested. One reviewer proposal, long hidden blocks for tuning, was tested and rejected with evidence.
- Round 4 (closing) ran on 2026-09-24 in a new gpt-5.6-sol session at high effort (the old thread had expired): round-3 fixes confirmed, long-block evidence accepted, `prepare` and `main_attack_ppo_enc.py --trace` cleared; two pre-split leaks in `prepare`, a one-sided displacement test, a missing `--trace_scale` and the v2 / v3 route path were found and fixed or documented the same day (`.aris/traces/research-review/2026-09-21_run03/round4.md`).
- **Deliverable:** `review/MENDING_PLAN_PART1.md`.

## Prioritised TODO

1. Fix PPO accounting and the evaluation protocol, and commit the scorecard script. *(Study plan §5, item 2; 0.5–1 day.)*
2. Build the common comparison problem, with random and CEM/CMA-ES attackers. *(1–1.5 days.)*
3. Held-out parameter intervals, then the main runs: 2 controllers + sanity case × 4 methods × 10 seeds. *(About 1 day including compute.)*
4. Owner, on the private data: `python data_prep/ais_prep.py legacy-check` and `prepare` (see `MENDING_PLAN_PART1.md`, section 4); share `report.md` if a second look is wanted.
4b. DONE 2026-09-24: round 4 sent and answered (`round4.md`); the review of part 1 is closed.
4c. DONE 2026-09-24: the route chain is wired into the environments (plan section 4, step 4; `data_prep/check_env_wiring.py`).
5. Secondary: counterfactual replay.

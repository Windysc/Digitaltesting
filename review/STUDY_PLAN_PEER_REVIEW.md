# Study plan for peer review (agreed after the external review)

Date: 2026-09-21.

The executor's first plan (2026-09-18) was reviewed by Codex gpt-5.6-sol at high effort. The thread is `01a0c314-85c9-7a30-918b-db863338b2fd` and the trace is `.aris/traces/research-review/2026-09-21_run03/round2.md`. This version keeps what the reviewer confirmed, drops what it called overstated, and uses its smaller package.

## 1. Verdict on the current study

**It is not defensible as it stands.** The code works; the problems are these:

| # | Problem | Evidence | Status |
|---|---|---|---|
| 1 | **No collision-avoidance controller is under test.** The default target is on a fixed track and ignores the attacker. An "attack success" therefore cannot show a failure of any avoidance logic. | `scenario_targets.py:270` | Confirmed |
| 2 | **The benchmark mostly saturates.** A scripted intercept succeeds 1.0 in 6 of the 8 families. It gets only 0.25 in the parallel lanes, because it always accelerates at maximum. So a learned policy does add something there. The honest claim is "mostly saturated", not "RL adds nothing". | `runs/enc_check_full/baseline_summary.md` | Confirmed, as corrected |
| 3 | **The comparisons lack power.** Dense vs sparse (1667 ± 236 vs 1583 ± 236 episodes) rests on 3 seeds and differs by less than one evaluation interval (250). | `runs/enc_full/matrix_summary.csv` | Confirmed |
| 4 | **Full vs turn-only control is confounded.** The two differ in actions (9 vs 3), observations (25 vs 20), capability, and episode length (25 vs 35–39 decisions). This must be reported as two capability settings, compared on environment transitions. | `attack_scenarios.py:174,251,502` | Confirmed |
| 5 | **Selection and test are not separated.** Checkpoints are selected on seed + 1000, and `viz_tool.build_env` also uses seed + 1000. The scorecard script and the 1152 raw records do not exist, so the scorecard cannot be reproduced and leakage is likely but unverifiable. | `main_attack_ppo_enc.py:151`, `viz_tool.py:121` | Confirmed, as corrected |
| 6 | **The PPO rollout boundary is mishandled.** Updates run every 4000 transitions, which can fall mid-episode, and `PPO.update` starts the return at zero. The unfinished episode is treated as if it ended. About one partial episode per buffer is affected. | `main_attack_ppo_enc.py:73,89,306`; `PPO.py:132-137` | **New (reviewer); verified** |
| 7 | **Evaluation policy mismatch.** Training evaluation samples actions, while the scorecard used argmax. | `main_attack_ppo_enc.py:460,468` | Confirmed |
| 8 | **"Success" means severity ≥ 2**, i.e. held twice inside a ship domain; collision is a separate case. The two must be reported separately. | `main_attack_ppo_enc.py:536`, `encounter_standard.py:147` | New |
| 9 | **The 1152 scorecard episodes** are 6 checkpoints × 8 scenarios × 24 episodes. They are not 1152 independent tests of one policy. | notes | New |
| 10 | **Provenance.** The old `enc_base` args.json files do not record `control`, and today's default is full control, so re-running the old command would not reproduce turn-only. | `runs/enc_base/*/args.json` | New |
| – | Truncation at `unresolved` | This is an explicit terminal failure with a penalty, so a zero bootstrap is **correct**. Document it; no change needed. | Withdrawn |

## 2. Framing

> **An RL adversary as a scenario generator for stress-testing simplified reactive ship collision-avoidance controllers.**

- **Systems under test.** The existing `manual` and `autonomous` reactive presets (`scenario_targets.AUTOMATION`), described plainly as simplified controllers that make one starboard alteration, with latency and cooldown. No general claims about MASS safety.
- **Sanity case.** The fixed-track target.
- **Out of the minimum paper:**
  - a new velocity-obstacle controller;
  - a learned controller under test;
  - adaptive stress testing;
  - transfer to the Fossen model.

  These can come later.

## 3. Methods compared

| Method | Role |
|---|---|
| hold / intercept / pursuit | sanity references (kept) |
| uniform random search | lower bound |
| CEM or CMA-ES over a 3-segment turn/speed manoeuvre | optimisation baseline |
| PPO adversary (fixed, §5 item 2) | feedback adversary under study |

**Fairness:**
- All methods attack the **same fixed set of initial encounters**, under the same attacker limits and the same 6 s decision timing.
- The budget counts **all** simulator calls, including PPO's training rollouts.
- Report both the discovery cost (the whole budget) and the deployment cost per new encounter. PPO's cost is amortised across encounters; the optimisers pay per encounter.

## 4. Outcomes and statistics

- **Collision and domain violation, reported separately.**
- Failure probability against the number of simulator calls (the budget curve), with the area under it and a confidence interval.
- The worst continuous margin reached: domain margin and DCPA.
- Coverage of **predefined bins**: encounter family × DCPA band × SUT response (evaded or not).
- Time to first failure, treating runs that never fail as censored.
- A **paired bootstrap** over seeds and the common held-out encounters. Use Wilson or Clopper–Pearson intervals for rates. rliable (IQM) is optional.
- **Held-out test set.** Hold out parameter *intervals* (course difference, DCPA, meeting time, speed ratio), not just new random seeds. The ranges can be set from your private AIS encounters, locally.
- **10 independent seeds** per controller and setting.
- Separate seeds for checkpoint selection, validation and the final test. Report the final test deterministically (argmax).

## 5. Minimum work package, in order

| # | Work | Effort (laptop CPU) |
|---|---|---|
| 1 | **Freeze the claim:** the framing in §2, with the fixed target as a sanity case only. | – |
| 2 | **Fix PPO and evaluation:** bootstrap the value at non-terminal buffer ends (or switch to GAE); deterministic reported evaluation; separate selection/validation/test seeds; count transitions and 1 s sub-steps; commit the scorecard/evaluation script with per-episode outputs; record `control` and every environment parameter in args.json. | 0.5–1 day |
| 3 | **One common comparison problem:** a fixed set of initial encounters; a random-search and CEM/CMA-ES attacker over a 3-segment manoeuvre on a shared `evaluate()` interface. | 1–1.5 days |
| 4 | **Held-out parameter-interval test set.** | 0.5 day |
| 5 | **Main runs:** 2 controllers (manual, autonomous) plus the fixed-target sanity case; 4 methods; 10 seeds. PPO takes about 4 min per 3000-episode run and the optimisers minutes. About 3–4 h wall time. | 0.5 day of compute + analysis |
| 6 | **Report** with the outcomes in §4. | 1 day |
| 7 | **Secondary: counterfactual replay.** Store full state snapshots and action sequences; replay the attacker's actions open-loop from t_warning *and* from the scenario start; search a small library of SUT manoeuvre sequences (constant course changes, one course/speed change). Label each failure **preventable / not demonstrated preventable / initially infeasible**. Do not relabel the second group as attacker-caused. | 1–2 days |

## 6. Claims each outcome allows

| Outcome | Claim |
|---|---|
| PPO finds more failures per total budget than CEM and random, on the held-out intervals | An RL adversary is an efficient failure generator for simplified reactive avoidance controllers. |
| PPO costs more to train but less per new encounter | Amortisation: worthwhile when many encounters must be tested. |
| CEM ≥ PPO | A benchmark result. The contribution is the testing framework: collision standard, lifecycle, exact-DCPA scenarios, and the comparison protocol. |
| No failures against `autonomous` within the limits | That controller is robust in the tested envelope. Widen the envelope (speed ratio, latency) before claiming more. |

## 7. References (status from the review)

| Reference | Status |
|---|---|
| Corso et al. 2021, JAIR, *A Survey of Algorithms for Black-Box Safety Validation of Cyber-Physical Systems* | Verified by the reviewer |
| Koren et al. 2018, *Adaptive Stress Testing for Autonomous Vehicles* | Verified. AST searches for *likely* failures using disturbance likelihood. |
| Agarwal et al. 2021, *Deep RL at the Edge of the Statistical Precipice* (rliable) | Verified |
| Porres et al. 2020, scenario-based testing of a ship collision-avoidance system | Probably real. Do not cite it for GP/BO until read. |
| "Torben et al." | Dropped (unverifiable as cited) |

## 8. What carries over unchanged

- the collision standard;
- the encounter lifecycle;
- the exact-DCPA scenario builder;
- full-control limits;
- the reactive target presets;
- the visualisation;
- the enc_base/enc_full runs, as the fixed-target sanity result.

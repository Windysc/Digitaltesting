# Review brief, round 2: is the attacker baseline and its comparison analysis fit for peer review?

Date: 2026-09-21. Same thread as round 1. Verify everything against the files; my notes are not evidence.

## What the owner asked

"Is the algorithm used in the current baseline and comparable analysis applicable and feasible under peer review? If not, reconstruct a better plan for the studies."

## Executor's answer to judge

`C:\Users\ASUS\Desktop\PPO_scenario_generate\review\STUDY_PLAN_PEER_REVIEW.md`

Its verdict is "not as it stands", for three reasons:
- there is no system under test;
- the comparisons lack power and are confounded;
- the baselines are the wrong kind.

It proposes reframing the work as falsification testing of a MASS collision-avoidance system under test (SUT). PPO would be compared with random search, Bayesian optimisation or CMA-ES, and adaptive stress testing, all at an equal simulation budget. Attacks would be constrained to plausible behaviour, with a replay step that attributes each failure. Results would follow the rliable statistics protocol with 10 seeds.

## Artifacts to check the claims against

All paths are in `C:\Users\ASUS\Desktop\PPO_scenario_generate\`:

| What | Where |
|---|---|
| PPO implementation | `PPO.py` (update at about lines 129–160: Monte-Carlo returns, terminal handling, minibatch) |
| Training and evaluation loop | `main_attack_ppo_enc.py` (env seed line 135, eval env seed + 1000 at line 151, eval loop about lines 389–420, stochastic `select_action`) |
| Environment and scripted baselines | `attack_scenarios.py` (`EncounterAttackEnv`, `steer_action`, `intercept_action`, `pursuit_action`, `BASELINES`) |
| Reactive target presets | `scenario_targets.py` (`AUTOMATION`, `TargetShip`) |
| Collision standard | `encounter_standard.py` |
| Scorecard environment rebuild | `viz_tool.py`, `build_env` (seed + 1000) |
| Results | `runs/enc_full/matrix_summary.csv`, `runs/enc_full/matrix_runs.csv`, `runs/enc_base/`, `runs/enc_check_full/baseline_summary.md` |
| Notes | `C:\Users\ASUS\Desktop\MASS_TESTING_ENV_REBUILD.md` (full-control and training sections) |

## Questions

1. Is each diagnosis in §2 of the plan correct? In particular:
   - checkpoint selection on the scorecard seed stream;
   - truncation treated as a terminal state;
   - the episodes-vs-steps confound between full and turn-only control;
   - the claim that the intercept baseline makes the RL result uninformative.

   Is anything overstated, or missed?
2. Is the reframing to SUT falsification the right one? Or is there a lighter framing that would pass review with less work, for example "RL adversary as a scenario generator" evaluated only against a reactive target?
3. Are the proposed comparison methods and metrics the ones reviewers in this area expect?
   - For each literature reference in the plan (Corso et al. 2021, JAIR; Koren et al. 2018; Porres et al. 2020; Torben et al.; Agarwal et al. 2021, rliable), say whether it exists and is described correctly, or mark it as unverifiable.
4. Is the attribution rule in §3.4 sound? The rule: a failure counts as valid only if the SUT had a feasible avoiding manoeuvre at t_warning. Is it practical to compute?
5. Give the minimum study package that makes the work defensible, as a prioritised list. The budget is a laptop CPU, one 3000-episode run in about 4 min.

Keep proposals in scope. No hashing and no infrastructure. Say plainly what is correct.

## Follow-up on round 1 (sampling)

I accept these round-1 points:
- the S4 metric was wrong;
- the 60 s gap split;
- separate chains for generator arrays and for scenario routes;
- the residual reading was backwards;
- the extra private checks you listed.

I will revise SAMPLING_SMOOTHING_FEEDBACK.md accordingly.

**D2 (TSGM noise size).** I read the source of tsgm 0.1.0, the latest release on PyPI (sdist). `GaussianNoise.generate` computes `sigma = variance**0.5` and adds `np.random.normal(mean, sigma, (n_samples, T, F))` to `X[seeds_idx]`. So σ = 0.01° i.i.d. per point and feature holds for the current release. The owner's installed version is still unknown. Does this settle D2 for you, subject to that version caveat?

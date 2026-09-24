# research-review run 03, round 2: attacker baseline and comparison analysis

- Date: 2026-09-21
- Call: codex-reply on thread 01a0c314-85c9-7a30-918b-db863338b2fd (gpt-5.6-sol, effort high)
- Brief: `review/RESEARCH_REVIEW_ROUND_2.md`, which points to `review/STUDY_PLAN_PEER_REVIEW.md`
- Full reply: result of MCP task k1ij936k3

## Reviewer verdict

"Not defensible as it stands" is correct, but several of the reasons were overstated or misidentified.

### Checkpoint leakage
Conditionally correct. `viz_tool.build_env` and the training evaluation both use seed + 1000. The scorecard script and the 1152 raw episode records are missing, so the leak is likely but cannot be verified.

There is also a mismatch in the reported policy: evaluation during training samples actions, while the scorecard used argmax.

### Truncation
Overstated. `unresolved` is an explicit terminal outcome at the safety horizon and carries a penalty, so bootstrapping it with zero is correct. It only needs documenting.

**Missed defect:** PPO updates every 4000 transitions, which can fall in the middle of an episode, and `PPO.update` starts computing returns from zero. So non-terminal rollout boundaries are wrongly treated as zero-return.

The executor verified this in `main_attack_ppo_enc.py:73,89,306` and `PPO.py:132-137`. About one partial episode in each 4000-transition buffer is affected.

### Full control vs turn-only
Correct. The two are confounded by action count, observation size, capability, and transitions per episode. Compare them on environment transitions and substeps, and describe them as different capability settings.

**Provenance:** the old `enc_base` args.json files do not record `control`, and the defaults have since changed to full control. Rerunning the notes' command without `--control turn` would not reproduce the turn-only study.

### Intercept baseline
Partly correct but overstated. Intercept scores 1.0 in 6 families but only 0.25 in the parallel lanes, because it always accelerates at maximum. The honest conclusion is "the benchmark mostly saturates", not "RL adds nothing".

### Also noted
- State "no collision-avoidance controller under test" rather than "no SUT".
- Keep the scripted baselines as sanity checks.
- "Success" is severity 2 by default, so report domain violations and collisions separately.
- The 1152 figure is 6 checkpoints × 8 scenarios × 24 episodes; these are not independent tests.

### Framing
Take the lighter option: an RL adversary as a scenario generator for stress-testing *simplified* reactive controllers, using the existing `manual` and `autonomous` presets. The fixed target stays as a sanity case.

Dropped from the minimum paper:
- the new velocity-obstacle controller;
- the learned SUT;
- adaptive stress testing (AST);
- the Fossen transfer check.

### Methods
Scripted baselines, random search, CEM or CMA-ES over a 3-segment manoeuvre, and PPO.

Fairness requirement: all methods must search the same space (fixed common initial scenarios), and the budget must include PPO's training rollouts. Report both discovery cost and deployment cost.

### Metrics
- Predefined bins rather than clustering.
- Time to first failure treated as censored.
- Area under the budget curve with confidence intervals.
- Paired or hierarchical bootstrap.
- Wilson or Clopper–Pearson intervals.
- rliable is optional.

### References

| Reference | Status |
|---|---|
| Corso et al. 2021, JAIR | Exists; described correctly |
| Koren et al. 2018, AST for autonomous vehicles | Exists; AST searches for *likely* failures using disturbance likelihood |
| Agarwal et al. 2021, rliable | Exists; described correctly |
| Porres et al. 2020 | Appears to be real, but its use for GP/BO is unverified; do not cite it for that |
| "Torben et al." | Unverifiable as cited; drop it |

### Attribution rule
Too strong as a binary rule. Use three labels:
- preventable;
- not demonstrated preventable;
- initially infeasible.

Implementation: replay the attacker's actions open-loop, restore the full state at t_warning, and search a small library of SUT manoeuvre sequences. Test from both t_warning and the scenario start. Episode logs must store full state snapshots and action sequences. Treat this as a secondary analysis.

### Minimum package
1. Freeze the claim.
2. Fix PPO accounting and evaluation.
3. One common comparison problem.
4. A held-out parameter-interval test set.
5. 10 seeds.
6. The right outcomes, reported with bootstrap intervals.
7. Counterfactual replay as secondary.

### D2 follow-up
Settled for tsgm 0.1.0, and upgraded to **correct**. The only remaining caveat is provenance: tsgm is not pinned.

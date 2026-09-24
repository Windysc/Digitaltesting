# research-review run 03, round 4 (closing round): RUN 2026-09-24

- Date: 2026-09-24 (the brief was written on 2026-09-21 and could not be sent then: usage limit)
- Call: the original thread 01a0c314-85c9-7a30-918b-db863338b2fd no longer existed on the Codex server
  ("Session not found"), so a NEW session was opened: codex gpt-5.6-sol, model_reasoning_effort high,
  sandbox read-only, cwd PPO_scenario_generate, with the round 1 to 3 traces and the plan named as context.
  Thread id of the new session: 01a0d398-ba78-7f00-a072-ecf885eed404
- Brief: `review/RESEARCH_REVIEW_ROUND_4.md` (round-3 fixes, the long-block disagreement, and the addendum on
  step 4 of the plan, which was applied on 2026-09-24 before the call)
- Full reply: result of MCP task k2x3pz2j6

## Reviewer verdict (closing)

1. Round-3 fixes: the main defects are fixed. The curvature path is sound (T7: table 0.184 / 0.495 / 0.643
   deg/s against playback 0.185 / 0.495 / 0.643). Placement uses the actual start tangent. Split-first holds
   for q, the COG diagnostics and the augmentation limits, with two gaps (below).
2. Long-block holdout: the evidence is accepted; `check_long_block_holdout.py` implements the proposal
   correctly, its failure is consistent across the report regimes, the bounded short-block rule is supported
   by every truth criterion and by the untouched T12 fleet. No further tuning experiment required; an active
   bound on the private report stays a review trigger.
3. Step-4 wiring: correct in the two fixed-track environments (no remaining box-5 or segment-heading path for
   their default fixed targets); not project-wide (v2 / v3 keep their legacy route path).
4. Go / no-go: nothing stops `prepare` with the default positions-only configuration, the report review, or
   `main_attack_ppo_enc.py --trace`. Before claiming equal coverage for `main_attack_ppo_scen.py` at a
   reduced scale or for v2 / v3, expose the scale and wire or document their route path. Fix the pre-split
   SOG-unit fallback before using a declared receiver COG on the private data.

## Findings and what was done (same day)

| # | Reviewer finding | Rank given | Action |
|---|---|---|---|
| 1 | `ais_prep.prepare`: the fallback SOG unit (majority vote) was taken from all voyages before the split | changes results | Split moved before the projection; the vote uses training voyages only, falls back to all voyages with an explicit note; the report prints the vote source and the majority (section 4) |
| 2 | `prepare`: `train_segments or all_segments` silently uses validation / test data when the training split leaves no segment | edge case | Explicit: console warning and `Estimated on:` in report section 3 (`q_segments_source`) |
| 3 | `--use_sogcog on` bypasses the three COG gates | optional override | Kept for the tests (`check_forced_aiding.py`); console warning and a capitalised note in report section 4 when it forces aiding past failed tests |
| 4 | v2 / v3 still call the legacy `route_from_trace` (point-count smoothing, v3 CLI default `--smooth_sigma 2`); they got the speed / deviation fixes only | changes training geometry | Documented explicitly (README_v2 / README_v3, plan step 4, rebuild notes, check scope): their targets steer on a polyline by pure pursuit under a yaw-rate limit, so the table playback does not apply; with data_prep arrays run them with `--smooth_sigma 0`; the CLI default stays 2 for the legacy arrays |
| 5 | `main_attack_ppo_scen.py` does not expose `trace_scale`; the checked 0.143 configuration cannot be selected | CLI limitation | `--trace_scale` added (default 1.0), passed through `run_matrix.py` and read back by `viz_tool.py` |
| 6 | `fit_route_to_limit`: the 50 m displacement is one-sided (smoothed to base only) | could change acceptance counts | Two-sided (larger of the two one-sided distances on the tables); `check_env_wiring.py` and `check_mending_plan.py tests` rerun, plan numbers updated where they moved |
| 7 | `check_env_wiring.py`: narrower than "all environments" (no v2 / v3), 8 / 3 routed episodes, yaw taken from `route_eval` instead of the target's own heading | scope | `TargetShip.psi_track` records the heading at every sub-step and the check reads it; W3 runs all 8 scenarios x 3 seeds x 2 attackers per scale, W4 3 families x 3 seeds; the scope statement (v2 / v3 excluded, and why) is in the docstring and the output |

No finding touched the numbers of T0 to T6, T8, T10 to T12 or the private-data reading guide.

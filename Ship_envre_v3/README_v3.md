# Ship_envre_v3 — the encounter-lifecycle definition

Third build of the critical-scenario trainer.  It replaces the "end at a
point / end at a time" thinking of the reference scripts (navigation target
with 200 m / 15 deg tolerance, 1000-step horizon) and of `Ship_envre_v2`
(destination radius, timeout failure, six-step "passed" rule) by a
definition in which no ship ends: both sail open-ended routes and an
episode is one ENCOUNTER, delimited by its own lifecycle.  Nothing in
`Ship_envre_v2` or `Ship_envre` was modified; v3 imports the unchanged v2
modules (vessel dynamics, danger criteria, target model and automation
levels, scenario families, grading, PPO, renderer, scorecard, self-play).
The two SimpleShipAI elements that the original design referenced are
adapted here: the live turtle viewer (`viewer_v3.py`) and the
ShipExperiment recorder (`ship_data_v3.py`).

```
Ship_envre_v3/
  ship_env_v3.py          ShipEncounterEnv: lifecycle phases, outcomes, reward, 19-value observation, PolicyShipV3
  target_ship_v3.py       steady target: course-holding steering, one alteration per emergency + cooldown, route straightening
  runner_v3.py            training / evaluation runner (structure of main_attack_ppo.py), lifecycle metrics
  main_attack_ppo_v3.py   attack task     (counterpart of Desktop/main_attack_ppo.py)
  main_ppo_v3.py          navigation task (counterpart of Desktop/main_ppo.py)
  viewer_v3.py            live viewer adapted from SimpleShipAI/viewer.py (two ships, routes, severity colours)
  ship_data_v3.py         episode recorder adapted from SimpleShipAI/ship_data.py (pickle under _experiments/)
  animate_v3.py           v2 renderer driven by the v3 runner (plain full-scope map + detailed display + panels)
  evaluate_agent_v3.py    v2 scorecard driven by the v3 runner
  selfplay_v3.py          v2 self-play driven by the v3 runner
  checkpoints/, scorecards/, animations/, runs/
```

Requirements: as v2 (numpy, torch, matplotlib; tqdm optional); `viewer_v3.py`
needs a display (tkinter).  Data: the same `(n, 100, 2)` lon/lat trace sets;
synthetic ones are created in `../Ship_envre_v2/data` on first run.

## The definition

Phases of an encounter, computed every step from the danger criteria of
`encounter.py` (DCPA/TCPA, ship domain, severity 0-3):

| phase | name | condition |
|---|---|---|
| 0 | approach | CPA ahead, no warning yet |
| 1 | action | a CPA warning or higher severity is or was active |
| 2 | passing | CPA behind (TCPA < 0) after phase 1 |
| 3 | clear | severity 0, range above `--r_clear` (2 nm) and opening for `--n_clear` consecutive steps (30 = 5 min); a second approach resets the count and continues the encounter |

Outcomes (`step` returns `obs, reward, done, success_flag` with flag 1 / 0 / −1):

| task | success (+1) | failure (−1) | neither (0) |
|---|---|---|---|
| attack | severity >= `--success_severity` held `--hold_steps`, or collision | encounter clears without the event (`clear`); leaving the map | `unresolved` at the safety cap |
| navigate | encounter clears with max severity <= 1 AND the ship is back on its plan: \|XTE\| < `--xte_tol` (0.1 nm) and course within `--course_tol` (10 deg) of the route for `--recover_steps` (18 = 3 min) after clearing | severity 2 or 3; passage abandoned (\|XTE\| > `--abandon_xte` 1 nm for `--abandon_steps` 60 = 10 min); leaving the map | `unresolved` at the safety cap |

The safety cap `--max_ep_len` (1080 steps = 3 h) only stops runaway
episodes; unresolved episodes are listed separately and never counted as
success or failure.  Routes are extended by `--route_extension` (15 nm)
along their final course so neither ship reaches an end; the map is the
region both ships can reach within the cap, so leaving it means wandering
off sideways, never a long regular passage.  The scenario meeting time
(scenarios.py) still decides where the encounter happens; it no longer
decides when the episode ends.

Observation (19 values): range, bearing, own surge / sway / yaw rate,
relative target velocity, course difference, closing rate, DCPA, TCPA,
severity, encounter phase, course error against the local route course,
cross-track error, previous rudder.  The elapsed-time fraction and the
destination features of v2 are gone, so the agent is not clock-aware and
has no point to race to.

Reward: attack as in v2 (closure, DCPA reduction, alignment, first-time
severity bonuses, time and manoeuvre costs, +50 on the event, −20 on
`clear` or leaving the map).  Navigate: track-keeping (cross-track and
course-error penalties, small progress term so the ship keeps sailing),
−1 per step under CPA warning, +0.2 per on-track step after clearing, +50 on
`resolved_on_track`, −30 / −50 on domain violation / collision, −20 on
abandonment or leaving the map.  No timeout penalty.

Metrics added to `evaluation()` and the per-scenario table
(`eval_scenarios_*.txt`): phase times `t_warning`, `t_cpa`, `t_clear`,
`t_on_track`, `resolution_time` (first warning to clear), `recovery_time`
(clear to back on track), `max_xte_after_passing`, `final_phase`,
`unresolved`.  Event grade and score come from `grading.py` unchanged.

## Target motion on the approach (target_ship_v3.py)

Checked 2026-09-14: with the v2 target model on scenario routes shaped like
the generated samples, the target's course changed by 9 deg on average
(up to 25 deg) before its first evasion, and the routes bent up to 1.2 km
off a straight line over 15 km; the approach looked randomised in the
animations.  Cause: the route shape was a rotated generated sample (with its
bend and noise) followed by pure pursuit on it.  v3 therefore uses:

* straight scenario routes by default (`--target_route_shape straight`);
  `data` keeps the generated shape, `auto` straightens a route only when it
  deviates less than `--straighten_tol` (500 m) from its chord, so small
  differences are tolerated and real bends are kept;
* a course-holding target: the desired course is the local route course
  at the ship's own projection plus a cross-track correction capped at
  5 deg, so on a straight route the course is constant to the metre; speed
  is the cruise speed throughout;
* one alteration per emergency: the COLREG starboard turn as before, then
  return to the route course and a cooldown of `--evasion_cooldown` (30
  steps = 5 min) before any new alteration.

Measured after the change (12 episodes over the families, navigation
baseline): approach course change 0.0 deg, route deviation 0 m; with
`--target_route_shape data`: 10.3 deg and 645 m.  Between-episode variation
of the scenario geometry (meeting time, DCPA offset, course difference,
speed) is kept; it is scenario diversity, not motion noise.

## Arrays from the mended sampling chain (2026-09-24)

With arrays from `PPO_scenario_generate/data_prep/ais_prep.py` (one point every
20 s of real time) pass `--trace_dt 20 --smooth_sigma 0`.  The `data` scenario then
takes the smoother's median speed of the window as the target's cruise speed when
`windows_speed_course.npy` lies next to `windows_lonlat.npy`.  `--straighten_tol`
defaults to 50 m since the same date and `route_deviation` measures the whole
route before the extension (was the first 15 km with a 500 m tolerance).  The
route model itself is the v2 one (polyline route, pure-pursuit steering under
`--target_yaw_rate`, `route_from_trace` with the point-count smoothing whenever
`--smooth_sigma` is above 0); the curvature-checked table playback of
`PPO_scenario_generate` is not used here.

## Running

```
python main_attack_ppo_v3.py --scenario mix --num_episodes 3000            # train attacker
python main_ppo_v3.py --scenario mix --num_episodes 3000                   # train navigator
python main_attack_ppo_v3.py --mode eval --ckpt runs/attack_v3/models/PPO_ShipAttackEnv_seed0_best.pth
python main_ppo_v3.py --mode eval --render                                 # scripted baseline in the live viewer
python main_ppo_v3.py --mode eval --record_experiment                      # + ShipExperiment pickle under _experiments/
python animate_v3.py --ckpt <ckpt> --scenario mix --episodes 3 --grid 6
python evaluate_agent_v3.py --task attack --ckpt <ckpt> --n 5 --out scorecards/attack_v3
python selfplay_v3.py --rounds 3 --scenario mix --num_episodes 600
```

All v2 arguments (scenario families, automation levels and overrides, danger
criteria, data files, PPO hyper-parameters) are accepted; the lifecycle
parameters are `--r_clear --n_clear --xte_tol --course_tol --recover_steps
--abandon_xte --abandon_steps --route_extension`.  PPO updates every
`--update_every_steps` (1200) environment steps instead of a multiple of an
episode length, because episodes no longer have a fixed length.

## Checks on the synthetic routes (2026-09-14)

Scripted baselines over the six scenario families (2 episodes each, other
ship `assisted`):

| task | success | unresolved | terminations | mean resolution / recovery |
|---|---|---|---|---|
| attack (pure pursuit) | 1.00 | 0 | domain_violation 12 | – |
| navigate (route following + starboard turn) | 0.67 | 0 | resolved_on_track 8, domain_violation 3, passage_abandoned 1 | 1488 s / 222 s |

With the lifecycle definition the pure-pursuit attacker also reaches the
faster overtaking target (the v2 "passed" rule used to cut that encounter
short), and the navigation baseline's failures are now maritime ones
(domain violation while standing on, a passage abandoned) rather than
timeouts or arrival misses.

Training on the six families, 400 episodes, seed 1, CPU (`--eval_every 100
--num_eval 24`, other ship `assisted`):

| task | scripted baseline | eval success @100 / 200 / 300 / 400 | best checkpoint | per-family @400 |
|---|---|---|---|---|
| attack | 0.88 | 0.46 / 0.21 / 0.25 / 0.25 | `checkpoints/ppo_attack_v3_mix_best.pth` (0.46) | crossing_port 0.75, others 0.00-0.25; 18 of 24 end `clear`, 0 unresolved |
| navigate | 0.88 | 0.00 / 0.12 / 0.17 / 0.71 | `checkpoints/ppo_navigate_v3_mix_best.pth` (0.71) | data 1.00, overtaken 1.00, head_on 0.75, crossing_starboard 0.75, crossing_port 0.50, overtaking 0.25; 17 resolved on track, 5 domain violations, 2 abandoned, 0 unresolved |

Scorecards (`scorecards/`, 2 episodes per cell, 6 families x 4 automation levels):

| checkpoint | success | robustness (min over automation) | hardest family / level | other |
|---|---|---|---|---|
| attack v3 mix | 0.38 | 0.17 | head_on / autonomous | mean event score 53, efficiency 0.71 |
| navigate v3 mix | 0.92 | 0.92 | crossing_port / none | safety score 55, COLREG compliance 0.38 |

Reading.  The definition change is what the navigation agent needed: under
the v2 arrival gate the same 500-episode budget gave 0.04 with most episodes
ending by timeout or off-map, under the lifecycle definition it reaches 0.71
with no unresolved episodes and its remaining failures are domain
violations in crossing and overtaking.  Its recovery times are long (25 min
on average) and its COLREG compliance is low: it resolves encounters but
not by the give-way starboard turn, which is where a compliance term in the
reward or the scripted starboard rule as a prior would go next.  The attack
agent is unstable at this budget (peak 0.46 at episode 100, 0.25 after);
without the "passed" rule it must now beat a target that keeps evading
until the range opens, and it loses most encounters to `clear`.  Both need
thousands of episodes and several seeds; all numbers are for the synthetic
routes.

Re-evaluation under the steady target (straight routes, course-holding,
one alteration per emergency; 24 episodes over the families, the two
checkpoints above were trained on the earlier curved-route target and are
NOT retrained):

| | scripted baseline | trained checkpoint | terminations (checkpoint) |
|---|---|---|---|
| attack | 1.00 | 0.42 | domain_violation 10, clear 14 |
| navigate | 0.79 | 0.62 | resolved_on_track 15, domain_violation 5, passage_abandoned 4 |

The steady target makes the attack easier for the scripted pursuit (every
family now ends in a domain violation) and slightly harder for the
navigation baseline (0.79 vs 0.88), because a target that holds its course
does not open the passage by itself.  Retrain on the steady target before
comparing agents; the checkpoints are kept only as starting points.

# Rebuilt modules for `main_ppo.py` / `main_attack_ppo.py` (2026-09-15)

The two reference scripts on the Desktop (dated 2024-02-08) import four things
that were not on this machine and are not public: `PPO`, and
`MassTestingEnv, ownship, navigation_target, obstacle` from `env_moving_obj`
(navigation) and `env_moving_attack` (attack).  They are rebuilt here from the
contract the scripts impose, next to the scripts, so both run **unmodified**.
Since 2026-09-15 everything lives in `Desktop/PPO_scenario_generate/` (scripts, rebuilt
modules, `viz_tool.py`, `runs/`).

| file | role |
|---|---|
| `PPO.py` | PPO-PyTorch interface (`select_action`, `buffer.rewards/is_terminals`, `update`, `decay_action_std`, `save`, `load`); copy of the tested `Digitaltesting-main/.../Ship_envre_v2/ppo_agent.py`. CPU by default, `set PPO_DEVICE=cuda` for the GPU. |
| `env_moving_obj.py` | world model + `MassTestingEnv` (task `navigate`), `ownship`, `obstacle`, `navigation_target` |
| `env_moving_attack.py` | `MassTestingEnv` subclass with task `attack`; same object classes |
| `gym.py` | shim: `import gym` resolves to `gymnasium` 1.3.0 (legacy `gym` is not installed on Python 3.14) |

Relation to ShipAI: none by code.  ShipAI (`SimpleShipAI-master`, `Ship_envre/Ship_env.py`)
is a continuous rudder/propulsion RK45 hull model on AIS traces; these scripts are a
discrete-action obstacle-scene trainer from a different codebase.  The only shared
element is the gym `Env` pattern.

## World model (assumptions the scripts leave open)

* Coordinates: `x = long` in 0..X_LEN m, `y = lat` in -Y_LEN..Y_LEN m; `cog`/`Direction`
  in degrees, 0 = +long, 90 = +lat, counter-clockwise positive.  This is the only
  convention consistent with every obstacle and destination placement in the scripts.
* Time: `duration` and `decision_interval` are 0.01 s ticks.  Script defaults
  (60000 / 600) give a 600 s episode, one decision every 6 s, at most 100 decisions.
  Motion is integrated in 1 s sub-steps; collisions, map exit and attack contact are
  checked each sub-step.
* Own ship: kinematic with rate limits, speed 0..15 m/s, acceleration +-0.1 m/s^2,
  turn rate +-2 deg/s, treated as a 10 m disc.  `ownship(lat, long, sp, cog, a, rot)`;
  `env.live_ownship.a` is the per-decision acceleration-command history.
* Actions (Discrete 9): index = 3 * accel + turn with accel in (decel, hold, accel) and
  turn in (port, straight, starboard).  Index 4 = do nothing, 7 = accelerate straight.
* Obstacles: rectangles L x W oriented by `Direction`, constant velocity (`sp` m/s along
  `cog`), position computed from time (no state, safe to share between env and eval_env).
  `risk_range` is a penalty zone (dense rewards) and a counter (`steps_in_risk`), not a
  terminal event: the turn destination lies 360 m from `ob_turn_static`, inside its 500 m
  range, so the risk zone cannot be a failure condition.  Touching the hull is a collision.
* Map exit: 100 m margin beyond the map.
* Observation (11 + 8 per object): own position/speed/heading, goal block (destination, or
  nearest target in the attack task), per-object relative position, hull distance,
  bearing, relative velocity, risk flag.  With one obstacle: 19 values.

## Episode rules and rewards

| task | success (flag 1) | failure (flag -1) | flag 0 |
|---|---|---|---|
| navigate (`env_moving_obj`) | within `target_deviation_distance` of the target with heading within `target_deviation_direction` | hull collision, map exit | running / timeout |
| attack (`env_moving_attack`) | within `attack_range` (default one target ship length) of a target hull; targets = `ts_list`, else all moving obstacles | collision with a static hazard, map exit | running / timeout |

`reward_type` (the scripts pass `final_step_reward` / `final_attack_reward`):

* `final_*`: terminal only, +10 success, -10 failure, timeout -10 x (final distance /
  initial distance) (attack: closest approach / initial distance).
* `dense_*`: adds progress toward the goal per decision (m / 100), -0.01 per step and
  -0.05 per step in a risk zone (navigation) or +0.05 (attack), plus the terminal terms.

`evaluation()` returns 22 fixed keys (outcome, steps, final state, distances, risk steps,
path length, command counts, total reward).  `show_scenes()` saves
`<save_dir>/scenes/scene_<task>_ep<N>_<k>.png` and never calls `plt.show()`.

## Running

Run from any directory; `PPO_logs/` and `--save_dir` are created relative to the cwd.

```
cd <work dir>
python C:\Users\ASUS\Desktop\main_ppo.py        --save_dir logs --obst_id line_static --dest_id line
python C:\Users\ASUS\Desktop\main_attack_ppo.py --save_dir logs --obst_id line_move   --dest_id line
```

Known defects in the reference scripts (left untouched):

* `main_ppo.py` default `--obst_id turn2` matches no branch, so `ob_list` is undefined
  and the script raises `NameError`.  Always pass one of `line_static`, `line_move`,
  `turn_static`, `turn_move` (`main_attack_ppo.py`: `line_move`, `line_move2`,
  `line_move_front_turn`, `line_move_back_turn`, `line_move_side_turn`).
* `eval()` assumes the episode finishes within 1000 steps; the environment guarantees
  it at 100 decisions.
* `--gpu`, `--batch_size`, `--lr` and the epsilon arguments are parsed but unused.

## Verification (2026-09-15, global Python 3.14, torch 2.14.0+cu126)

* Unit test (`envtest.py` in the session scratchpad): straight run reaches the line
  destination in 30 decisions; straight run into `ob_line_static` collides at 24; random
  policies give timeouts / map exits with stable evaluation keys; idling in the attack
  scene times out at 375 m closest approach (no trivial success), a scripted intercept
  succeeds at 198 m; PPO update, deepcopy, save/load and the gym shim all pass.
* `main_ppo.py` 60 episodes, `line_static` / `line`: 19 s wall clock, checkpoints, eval
  CSVs and a scene written; eval success 0.2 at episode 30 (early, expected low).
* `main_attack_ppo.py` 60 episodes, `line_move` / `line`, `--render 1`: 14 s, eval
  success 1.0 at episode 60.  That scenario is easy by construction (the target steams
  toward the attacker); `line_move2` and the side/back variants require chasing.

## GPU feasibility (2026-09-15, RTX 4060 Laptop, driver 561 / CUDA 12.6, torch 2.14.0+cu126)

`PPO_DEVICE=cuda` runs the whole chain on the GPU (the agent prints `PPO agent on device: cuda`).
300 episodes, eval every 100 with 20 episodes, seed 1, run from the project folder:

| run | device | scenario | reward | training time | eval success @100/200/300 |
|---|---|---|---|---|---|
| runs/nav_cpu | cpu | line_static / line | final_step_reward | 35 s | 0.00 / 0.00 / 0.00 |
| runs/nav_gpu | cuda | line_static / line | final_step_reward | 70 s | 0.00 / 0.00 / 0.05 |
| runs/attack_gpu | cuda | line_move2 / line (chase) | final_attack_reward | 74 s | 0.00 / 0.00 / 0.00 |
| runs/attack_gpu_dense | cuda | line_move2 / line (chase) | dense_attack_reward | 56 s | 0.00 / 0.45 / 0.65 |

* Feasible on both devices; the GPU is about 2x SLOWER than the CPU for this 64-unit MLP
  (per-step host/device transfers dominate).  Use the CPU unless the network grows.
* The scripts' original sparse rewards (`final_*`) barely learn in 300 episodes; the
  `dense_*` family learns the chase (0.65 success, episode length 88 -> 46 decisions).
  Pass `--reward_type dense_attack_reward` / `dense_step_reward` for short budgets.
* Failure mode of the learned chaser: the target track crosses the map edge near
  (400, -1000), so late intercepts end `off_map` (replay fan shows this).

## Visualisation tool: `viz_tool.py`

```
python viz_tool.py all    runs/attack_gpu_dense --episodes 20      # everything below
python viz_tool.py curves runs/nav_gpu runs/nav_cpu                # overlay eval curves of several runs
python viz_tool.py evals  runs/attack_gpu                          # outcome mix / final positions / closest approach
python viz_tool.py replay runs/attack_gpu_dense --ckpt best --episodes 20 [--deterministic] [--fps 8]
python viz_tool.py report runs/attack_gpu_dense                    # single HTML page (base64-embedded images)
```

Outputs under `<run>/viz/`: `curves.png`, `evals.png`, `replay_fan.png`,
`replay_episode1.gif` (own ship, moving hulls, risk zones, action / reward overlay),
`replay_episodes.csv`, `report.html`.  The scenario is rebuilt from `<run>/args.json`
(obst_id, dest_id, reward_type decide task and scene), the checkpoint from
`<run>/models/MassTestingEnv/` (`best` falls back to `last`).  Replay samples actions like
the scripts' own `eval()`; `--deterministic` uses argmax, which can score lower on a
policy trained this briefly.

## Scaled training (2026-09-15, 3000 episodes per run, CPU, 6 runs in parallel, seed 1)

`runs/scale/<name>/`, eval every 250 episodes with 20 episodes, 3 threads per process;
wall clock 150-193 s per run while sharing 22 cores.  `runs/scale/compare/` holds the
overlay figure, CSV and HTML from `viz_tool.py compare`.

| run | task | scenario | reward | eval success 1.0 first reached | final eval length (decisions) | replay 20 ep: stochastic / argmax |
|---|---|---|---|---|---|---|
| nav_line_dense | navigate | line_static / line | dense_step_reward | 1250 | 41.4 | 19/20 / 20/20 |
| nav_line_sparse | navigate | line_static / line | final_step_reward (original) | 1000 | 42.9 | 20/20 / 20/20 |
| nav_turn_dense | navigate | turn_static / turn | dense_step_reward | 1250 | 45.0 | 18/20 / 20/20 |
| atk_chase_dense | attack | line_move2 / line | dense_attack_reward | 750 | 29.7 | 19/20 / 20/20 |
| atk_chase_sparse | attack | line_move2 / line | final_attack_reward (original) | 1250 | 28.1 | 20/20 / 20/20 |
| atk_side_dense | attack | line_move_side_turn / line | dense_attack_reward | 750 | 30.5 | 20/20 / 20/20 |

Findings:

* Every configuration converges to 1.0 evaluation success and then keeps shortening
  the episode (efficiency keeps improving after success saturates).
* The scripts' ORIGINAL sparse rewards do learn at the original budget: they need
  ~1000-1250 episodes, the dense variants ~750-1250, so the 300-episode verdict was
  a budget effect, not a defect.  Dense rewards start earlier (success visible at 250).
* Learned strategies (replay fans): line navigation swings north around the crossing
  obstacle and enters the gate heading east; turn navigation curves up to the gate and
  arrives heading north, occasionally clipping the obstacle corner under sampling;
  chase attack runs south-west to meet the target track, the rare failure being an
  intercept pushed off the map edge; side attack parks ahead of the crossing track
  (x ~1300-1450) and lets the target run into the 200 m contact range.
* Argmax replays are 20/20 everywhere; the scripts' sampling evaluation loses 1-2
  episodes per 20 on the navigation tasks (collision or timeout).

## Presetting revision 2026-09-15 (owner): speeds and target routes

Problems raised: (1) own-ship and target speeds did not correspond (own ship up to
15 m/s from rest vs targets at 4-9 m/s); (2) the target line was a straight,
infinite, inert constant-velocity track with no connection to the project's route
data.  The 48-run matrix under those presettings was stopped (32 finished, summary
in `runs/scale2`, superseded for the attack task).

Decisions: EQUAL SPEEDS (both ships at one constant cruise speed, own ship starting
at it, turn-only actions); routes from the v2 COLREG scenario generator scaled to
the 2 km arena with optional data shapes; steady course-holding target with one
COLREG alteration per emergency (v3 behaviour); arena and time base unchanged.

New files: `scenario_targets.py` (generator, `TargetShip`, `ScenarioAttackEnv`),
`main_attack_ppo_scen.py` (copy of the attack script with `--scenario`,
`--cruise_speed`, `--trace`, `--automation`, `--turn_rate`, `--speed_control`,
`--attack_range`; identical training loop), `run_matrix.py --tasks scen`, viz_tool
support (routes and target tracks in fans and animations, per-scenario replay
success).  Base env gained per-instance action tables / speed caps and
advance/reset hooks for stateful objects.

Families: head_on (course diff 174-186 deg), crossing_starboard (60-120),
crossing_port (-120..-60); meeting time 120-300 s; CPA offset +-500 m; overtaking /
overtaken excluded by the equal-speed rule.  Data shapes: `--trace <npy>` uses
(n,T,2) lon/lat traces (only `Ship_envre_v2/data/*_synthetic.npy` exist; real
generated traces drop in).  Target presets scaled from v2: none | manual (20 s
latency, alert DCPA 250 m / TCPA 120 s, 30 deg starboard, 60 s cooldown) |
autonomous (0 s, 300 m / 150 s, 45 deg).

Scripted-pursuit check (30 episodes each, seed 11): a lead-pursuit heuristic
succeeds 28/30 at ANY contact range (200 / 60 / 20 m) against every automation
level (closest approach median 16-17 m at 20 m range); pure pursuit succeeds
23/30 at 200 m but only 5-12/30 at 60 / 20 m.  So the target's evasion presets
barely change the outcome in this arena; what the agent must learn is the lead
angle.  `--attack_range` tightens the success criterion when needed.

### Scenario baseline matrix (`runs/scen_base`, 3000 episodes, eval 30 every 250, seeds 1-3)

| scenario | target | reward | first eval at 1.0 (mean+-sd) | final success | final length |
|---|---|---|---|---|---|
| head_on | manual | sparse / dense | 1000+-204 / 750+-204 | 1.00 / 0.96 | 32.6 / 34.5 |
| crossing_starboard | manual | sparse / dense | 917+-118 / 667+-118 | 1.00 / 0.98 | 32.3 / 33.2 |
| crossing_port | manual | sparse / dense | 1083+-118 / 1083+-312 | 1.00 / 1.00 | 26.2 / 27.2 |
| mix | none | sparse / dense | 1417+-118 / 1250+-204 | 1.00 / 0.97 | 30.8 / 31.8 |
| mix | manual | sparse / dense | 1333+-118 / 1583+-236 | 1.00 / 0.99 | 31.3 / 31.5 |
| mix | autonomous | sparse / dense | 1583+-624 / 1917+-825 | 0.99 / 0.97 | 31.8 / 33.1 |

All 36 runs reach 1.0 evaluation success; the mix needs ~1.3-1.9k episodes, single
families 0.7-1.1k; a reactive target delays convergence mildly (none < manual <
autonomous) but does not lower the final success, consistent with the pursuit
check.  Wall clock 836 s for 24 runs + 500 s for 12 (10 parallel).

## Chart-style visualisation (2026-09-15, owner: same style as the previous build)

`chart_viz.py` ports `Ship_envre_v2/animate_scenes.py` to this frame (metres, arena
scale) and works for every environment variant (obstacle scenes and scenario
targets).  Layout as before: fixed-extent FULL-SCOPE map (routes, trails with 30 s
dots, both ships, contact-zone outline) | DETAIL navigation display (sea background,
km graticule, scale bar, north arrow, camera following both ships, true-scale hull
silhouettes, 60 s velocity vectors, fading trails, target COLREG sectors, contact
ellipse and risk ring, CPA prediction with the DCPA segment, data box) | side panels
(range / hull distance / DCPA, TCPA, severity + target evading, own commands).
Severity: 0 clear, 1 risk ring, 2 contact zone, 3 hull contact.

* `viz_tool.py replay <run>` now writes the chart-style `replay_episode1.gif` and
  `replay_episode1_storyboard.png` (4 full-scope panels); the plain animation is gone.
* `viz_tool.py chart <run> --episodes N [--grid M]` writes one GIF + storyboard per
  episode and an M-episode grid GIF under `<run>/viz/chart/` with `chart_summary.json`.
* `report.html` embeds the storyboard above the animation.
Verified on `runs/scen_base/scen_mix_manual_sparse_s1`, `runs/scale/atk_chase_dense`
and `runs/scale/nav_line_dense` (static hazard + destination gate rendering).

## Fixed-track target (2026-09-15)

The attacked ship is not an agent: `TargetShip` in `scenario_targets.py` is scripted. Its reactive
presets (`manual`, `autonomous`) made it alter course when the attacker came close. The default is
now `automation='fixed'` (in `TargetShip`, `ScenarioAttackEnv`, `main_attack_ppo_scen.py --automation`
and `run_matrix.py --automation`): the target plays its generated route back exactly by arc length at
cruise speed. Its position depends on time only, so the track is identical whatever the attacker does,
lies on the route (straight chord or data-shaped trace), never evades, and has heading = segment
direction. `none` is now an alias of `fixed` (identical to the old `none` on straight routes, so the
`scen_mix_none_*` checkpoints stay valid: 9/9 argmax success). `manual` / `autonomous` remain available
only when passed explicitly; runs trained with them keep their setting via args.json.

Verified: all three families, straight and trace-shaped routes, own ship holding course / turning
port / pursuing -> target tracks identical to 0 m, off-route below 1e-12 m, 0 evasions.
Note the generator still aims each route at the own ship's projected position (DCPA offset <= 500 m).

## Collision standard and multiple attack scenarios under the v3 scenario definition (2026-09-15)

### What v2 and v3 define (checked in Desktop/Digitaltesting-main/Digitaltesting-main)

| Item | Ship_envre_v2 | Ship_envre_v3 | previous PPO_scenario_generate frame |
|---|---|---|---|
| danger levels | `encounter.py`: 1 CPA warning (DCPA < d_safe 926 m and 0 <= TCPA < t_safe 900 s), 2 domain violation (other ship inside either ship's ellipse, semi-axes 4 L along / 1.6 L across, L = 244.74 m), 3 collision (centre distance < 1.0 L) | same module, unchanged | 500 m risk ring (penalty only), success = hull distance <= one target length, collision = hull contact (10 m) |
| further indices | CRI = 0.2^((DCPA + v TCPA) / 2000 m); COLREG type from relative bearings (head-on +-6 deg at course difference > 174 deg, overtaking > 112.5 deg abaft the beam, crossing by side) | same | none |
| event grading | `grading.py`: grade = max severity (safe passage / close quarters / domain infringement / collision); score 0-100 = 40 (1 - min DCPA / d_safe) + 25 sev / 3 + 15 CRI + 10 time in domain / 300 s + 10 closing speed / 10 m/s | same | none |
| attack success | severity >= 2 held 2 decisions, or collision; failure: off map, timeout (300 steps), "passed" (CPA behind, range > 2 d_safe for 6 steps) | success as v2; failure: encounter `clear` (severity 0, range > 2 nm and opening for 30 steps = 5 min) or off map; `unresolved` at the 3 h safety cap (flag 0, never counted); no timeout, no destination | success = contact; timeout flag 0 with a partial reward |
| scenario geometry | `scenarios.py`: meeting point T_meet (15-30 min) ahead on the own course, offset laterally by "dcpa" +-0.3 nm; the target starts T_meet earlier on its course through that point; course difference 174-186 / 60-120 / -120..-60 / +-15 deg; target 4-9 m/s, overtaking 0.4-0.65 x own, overtaken 1.4-2.2 x own | same families; straight routes, steady course-holding target, one alteration per emergency | v2 geometry scaled to 120-300 s and +-500 m, equal speeds |
| map / time | bounding box of the routes + 1500 m, 10 s decisions | region reachable within the cap, routes extended 15 nm | 2 x 2 km arena, 600 s |

Findings from the check:
* v2's "dcpa" is the lateral offset of the meeting point, not the DCPA; for a crossing the DCPA is only 0.5-0.87 x the offset.  With +-0.3 nm (556 m) against d_safe 926 m and a domain half-width of 392 m, every v2 encounter starts with a pending CPA warning and most start inside the domain half-width, so an own ship that only holds course already produces the event.  Measured below: v2 sampling, attacker holding course, success 83-100 % in every head-on and crossing scenario.  This is also why the target looked as if it were steering into the attacker.
* d_safe (3.78 L) is smaller than the domain's along-course semi-axis (4 L), so near the bow or stern a domain violation can occur without a CPA warning ever being raised.
* Collision is a centre distance below 1 L whatever the relative heading (no hull geometry).
* v3's clear rule needs an opening range, so two equal-speed ships in parallel lanes never clear; such an encounter ends `unresolved`.

### Port (new files)

* `encounter_standard.py`: the v2 criteria (cpa, domain_margin, severity, CRI, encounter_type plus `parallel`), v2 grading, the v3 `Lifecycle`, and `CollisionStandard(scale)`.  Every length and time is v2 x scale with the speed kept.  `arena` = 35/244.74: L 35 m, W 8.75 m, d_safe 132 m, t_safe 129 s, domain 280 x 112 m, collision 35 m, meeting 129-257 s, clear > 530 m for 43 s, turn rate 1.96 deg/s (turning radius 5 L at 6 m/s), safety cap 1544 s.  `maritime` = 1: the v2 numbers, L 245 m, turn rate 0.28 deg/s.  Any other factor is accepted.
* `attack_scenarios.py`: `EncounterAttackEnv` (task attack, one encounter per episode, v3 outcomes success / clear / off_map / unresolved; severity per decision = worst 1 s sub-step so a collision cannot be stepped over; fixed-track target, equal speeds, turn-only attacker; 20-value observation with DCPA, TCPA, severity, phase, domain margin, CRI and hold count), the scenario catalogue, the DCPA bands and scripted baselines (hold, intercept, pursuit).  Reward `final_attack_reward`: +10 success, -10 clear / off map, unresolved -10 x min(1, closest range / initial range).  `dense_attack_reward` adds range closure, DCPA reduction, bearing alignment, a step cost and first-time bonuses (+1 warning, +3 domain).
* `main_attack_ppo_enc.py`: the reference training loop with `--scenario --dcpa_band --scale --cruise_speed --trace --turn_rate --success_severity --hold_steps --max_encounter_s`.
* `check_attack_scenarios.py`: geometry self-test (both scales, 1600 draws each: the initial DCPA equals the drawn value, side, bow / stern crossing, parallel start), baseline tables, catalogue figures, optional scripted grid GIF.
* `run_matrix.py --tasks enc --enc_scenarios ... --bands ... --scale ...`; `viz_tool.py` builds the env from args.json (`dcpa_band` key).

### The attack scenarios

| scenario | rule | geometry (attacker at 0,0 heading east, both holding course) |
|---|---|---|
| head_on_port / head_on_starboard | Rule 14 | reciprocal courses 174-186 deg, target passes down the attacker's port / starboard side |
| crossing_starboard_bow / _stern | Rule 15, attacker gives way | course difference 60-120 deg, target from starboard crosses ahead of / passes astern of the attacker |
| crossing_port_bow / _stern | Rule 15, attacker stands on | course difference -120..-60 deg, target from port crosses ahead / passes astern |
| parallel_port / parallel_starboard | parallel lanes (the merge-line geometry of the data set) | same course and speed, target abeam on port / starboard at the band distance, 0.5-2 domain_a astern |

Groups: `head_on`, `crossing_starboard`, `crossing_port`, `crossing`, `parallel`, `mix` (all 8); comma lists are allowed.  Overtaking and overtaken are not in the set because the speeds are equal.

Initial DCPA bands (the meeting-point offset is solved so the initial DCPA is exact): `collision` [0, 1 L), `close` [domain_b, d_safe), `passing` [d_safe, 2 d_safe) = safe passage if nobody manoeuvres (default: the attacker has to create the danger), `wide` [2 d_safe, 4 d_safe), `v2` = the v2 offset sampling.  For parallel lanes the band sets the abeam distance.

Catalogue figures: `runs/enc_check/catalogue_arena_<band>.png`, `runs/enc_check_maritime/catalogue_maritime_passing.png`.

### Scripted baselines (arena scale, 12 episodes per cell, seed 7; runs/enc_check/baseline_summary.md)

Success rate as hold course / intercept / pure pursuit:

| scenario | collision | close | passing | wide | v2 |
|---|---|---|---|---|---|
| head_on_port | 1.00 / 1.00 / 1.00 | 0.00 / 1.00 / 1.00 | 0.00 / 1.00 / 1.00 | 0.00 / 1.00 / 1.00 | 0.83 / 1.00 / 1.00 |
| head_on_starboard | 1.00 / 1.00 / 1.00 | 0.00 / 1.00 / 1.00 | 0.00 / 1.00 / 1.00 | 0.00 / 1.00 / 1.00 | 0.83 / 1.00 / 1.00 |
| crossing_starboard_bow | 1.00 / 1.00 / 0.00 | 0.42 / 1.00 / 0.00 | 0.00 / 1.00 / 0.00 | 0.00 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 |
| crossing_starboard_stern | 1.00 / 1.00 / 0.17 | 0.42 / 1.00 / 0.25 | 0.00 / 1.00 / 0.25 | 0.00 / 1.00 / 0.42 | 1.00 / 1.00 / 0.25 |
| crossing_port_bow | 1.00 / 1.00 / 0.00 | 0.75 / 1.00 / 0.00 | 0.00 / 1.00 / 0.00 | 0.00 / 0.67 / 0.00 | 1.00 / 1.00 / 0.00 |
| crossing_port_stern | 1.00 / 1.00 / 0.00 | 0.75 / 1.00 / 0.00 | 0.00 / 1.00 / 0.00 | 0.00 / 1.00 / 0.25 | 1.00 / 1.00 / 0.00 |
| parallel_port and _starboard | 0.25 / 0.83 / 0.25 | 0.00 / 0.42 / 0.42 | 0.00 / 0.83 / 0.83 | 0.00 / 1.00 / 0.67 | 0.00 / 0.33 / 0.25 |

Reading: in the `passing` and `wide` bands holding course never succeeds (head-on and crossing encounters clear as safe passages, parallel lanes stay unresolved), so the attacker has to create the event.  The intercept heading (constant bearing) solves every head-on and crossing case.  Pure pursuit fails every crossing because an equal-speed stern chase never closes; those episodes end unresolved at the cap.  In the `close` band holding course still infringes the domain in 42-75 % of crossings because the ellipse is long along the target's course.  Maritime scale (runs/enc_check_maritime, 4 episodes per cell, passing band) shows the same pattern: hold 0 everywhere, intercept 1.00 for head-on and crossing and 0.75 for parallel lanes, pursuit 0 for bow crossings.  Its events are domain violations rather than collisions because the turning radius is 5 L.

### Renewed display (chart_viz.py)

* Header: scenario, rule, meaning, initial range / DCPA / TCPA / encounter type, and the thresholds of the standard.
* Full-scope map: the fixed target track, the attacker's track coloured by severity, time ticks, start labels and the closest-approach segment.
* Detail display: a smoothed camera, hulls magnified when too small (factor printed), the target domain filled when violated, the d_safe ring, collision circle, attacker domain, COLREG sectors, CPA prediction, and a data box (phase, range, bearing, encounter type, DCPA, TCPA, CRI, severity, hold count, domain margin).
* Five panels with the lifecycle phase shaded: range + DCPA, TCPA with the warning window, domain margin + CRI, severity, turn command.
* An outcome banner held for 2 s at the end.
* Storyboards at the event frames (start, first warning, domain entered, closest approach, end); grid GIFs with the outcome per cell; `catalogue()` for the scenario set.
* Older runs (obstacle scenes, ScenarioAttackEnv) still render with their own severity definition.

### Training on the attack scenarios (runs/enc_base, arena scale, passing band, 3000 episodes, CPU)

`python run_matrix.py --out runs/enc_base --tasks enc --enc_scenarios mix --bands passing --seeds 1 2 3 --episodes 3000 --eval_every 250 --num_eval 24`, then the same with the 8 single scenarios and seed 1.  22 runs, 14 min wall time in total, no failures.

| training set | reward | seeds | first evaluation at success 1.0 (episode) | final evaluation success | final length (decisions) |
|---|---|---|---|---|---|
| mix (all 8) | dense | 3 | 2167 +- 425 | 0.96 | 35 |
| mix (all 8) | sparse | 3 | 2417 +- 514 | 0.94 | 39 |
| each single scenario | dense / sparse | 1 | 750-1250 / 500-1750 | 0.96-1.00 | 12-44 |

Per-scenario scorecard of the six mix checkpoints (best checkpoint, deterministic policy, 24 episodes per scenario, fresh seed):

| scenario | success, mean of 6 checkpoints | time to event | how it ends |
|---|---|---|---|
| head_on_port | 0.99 | 186-296 s | collision and domain violation about half each |
| head_on_starboard | 1.00 | 186-238 s | as above |
| crossing_starboard_bow | 0.97 | 201-219 s | mostly domain violation, event score 51-63 |
| crossing_starboard_stern | 0.99 | 206-220 s | mostly collision, score 75-84 |
| crossing_port_bow | 0.96 | 178-192 s | mostly domain violation, score 40-55 |
| crossing_port_stern | 1.00 | 214-267 s | collision and domain violation |
| parallel_port | 0.92 | 56-116 s | failures are clear or unresolved |
| parallel_starboard | 0.97 | 53-67 s | as above |

Reading: under the collision standard and the v3 definition all 8 attack scenarios are learnable by the unchanged PPO loop.  The mix costs about 2-3 times the episodes of a single scenario.  The bow crossings and the port parallel lane are the weakest cells; the bow crossings end as domain violations with low event scores, because the attacker cuts in front of a target that crosses ahead instead of reaching its hull.  The learned paths are lead-intercept curves (grid GIF), not the pure pursuit that fails the crossings.  Replay and report of the best mix run: `runs/enc_base/enc_mix_passing_dense_s3/viz/` (report.html, replay_episode1.gif, chart/grid8.gif).

## Full attacker control with limits on the six ownship parameters (2026-09-15)

Owner request: the simulator in full action, with control of the six parameters and a range limit for each one.  The six parameters are the ones the reference scripts build the attacker from, `ownship(lat, long, sp, cog, a, rot)`.  `EncounterAttackEnv(control='full')` is now the default; `control='turn'` keeps the earlier preset (cruise speed fixed, 3 turn actions) so the `runs/enc_base` checkpoints still load.

| parameter | limit (default) | option: `main_attack_ppo_enc.py` / `EncounterAttackEnv` | what happens at the limit |
|---|---|---|---|
| long (east) | the encounter map: the region both ships cover holding course over the cap, plus r_clear | `--long_range lo,hi` [m] / `long_range` | leaving the area ends the episode as `off_map` (failure); checked before success |
| lat (north) | as above | `--lat_range lo,hi` [m] / `lat_range` | as above |
| sp (speed) | 3.0-12.0 m/s (the v2 tanker's steady speeds 3.3 / 7.7 / 11.9 m/s); starts at the 6 m/s cruise speed | `--speed_range lo,hi` / `v_range` (must contain the cruise speed) | speed held at the limit; commands beyond it counted in `speed_limit_hits` |
| cog (course) | free | `--cog_limit deg` / `cog_limit` (maximum change from the start course) | course held at the limit; `course_limit_hits`, `max_course_change_deg` |
| a (acceleration) | -0.1..+0.1 m/s^2 (reference scripts' A_MAX) | `--accel_range lo,hi` / `a_range` (must contain 0) | commands clamped |
| rot (turn rate) | +-1.96 deg/s at arena scale (turning radius 5 L at cruise), +-0.28 maritime | `--turn_rate deg/s` / `turn_rate_deg` | commands clamped |

Actions: `--acc_levels` x `--rot_levels` discrete levels (odd counts, zero always a level), default 3 x 3 = 9: decel / hold / accel x port / straight / starboard.  The observation gains 5 values under full control (speed within its range, previous acceleration and turn command, distance to the area edge, course change), 25 in total.  `evaluation()` adds control, min_speed, max_speed, n_accel_cmds, n_decel_cmds, mean_abs_accel_cmd, speed_limit_hits, course_limit_hits, max_course_change_deg.  The ownship model applies the limits itself (optional per-instance a_min / a_max, v_min, cog_limit in `env_moving_obj.ownship.advance`); ships without them behave exactly as before.

Checks (`test_full.py`, random and intercept actions, limits measured from the integrated 1 s sub-steps):

| configuration | speed | acceleration | turn rate | course change | position | outcomes |
|---|---|---|---|---|---|---|
| default full, 30 episodes | 3.00-12.00 m/s | -0.100..+0.100 m/s^2 | +-1.96 deg/s | free (up to 259 deg) | inside the area until the step that leaves it | 10 success, 17 clear, 3 off_map |
| custom: 4-8 m/s, -0.05..+0.20 m/s^2, course +-45 deg, area x -100..1500 m, y -600..600 m, 30 episodes | 4.00-8.00 | -0.050..+0.200 | +-1.96 | max 45 deg | as above | 21 off_map, 6 success, 3 clear |
| 5 x 5 levels, 10 episodes | 3.90-12.00 | levels -0.1, -0.05, 0, 0.05, 0.1 | levels +-1.96, +-0.98, 0 | free | as above | 6 success, 4 clear |
| turn mode (enc_base run) | 6.00 fixed | 0 | +-1.96 | free | as above | enc_base checkpoint 16/16 as before |

Invalid limits are rejected (speed range not containing the cruise speed, acceleration range not containing 0, even level counts).  One ordering fix came out of the checks: a step that both leaves the area and completes the event used to count as success; the area limit is now checked first.

Scripted baselines under full control (arena scale, passing band, 12 episodes per cell, `runs/enc_check_full/baseline_summary.md`), success rate with time to the event:

| scenario | hold course | intercept (sprints to 12 m/s) | pure pursuit (at 12 m/s) | turn-only intercept for comparison |
|---|---|---|---|---|
| head_on_port / _starboard | 0.00 | 1.00, 131 s | 1.00, 131 s | 1.00, 178 s |
| crossing_starboard_bow | 0.00 | 1.00, 109 s | 1.00, 118 s | 1.00, 188 s |
| crossing_starboard_stern | 0.00 | 1.00, 122 s | 1.00, 127 s | 1.00, 184 s |
| crossing_port_bow | 0.00 | 1.00, 102 s | 1.00, 109 s | 1.00, 222 s |
| crossing_port_stern | 0.00 | 1.00, 119 s | 1.00, 128 s | 1.00, 186 s |
| parallel_port / _starboard | 0.00 (unresolved) | 0.25 (a third leave the area) | 0.25 | 0.83 |

Reading: with the speed margin (12 vs 6 m/s) the event comes 35-55 % earlier, and pure pursuit, which could never close a stern chase at equal speed, now succeeds in every crossing (as domain violations).  In the parallel lanes the scripted rules get worse: sprinting ahead of a target that starts astern overshoots and leaves the area.  This is a weakness of the heuristic, not of the limits, and a learned policy has to find the slower merge.

### Training with full control (runs/enc_full, arena scale, passing band, all 8 scenarios mixed, 3000 episodes)

`python run_matrix.py --out runs/enc_full --tasks enc --control full --enc_scenarios mix --bands passing --seeds 1 2 3 --episodes 3000 --eval_every 250 --num_eval 24`, 6 runs, 4 min wall time, no failures.

| attacker control | reward | first evaluation at success 1.0 (episode, 3 seeds) | final evaluation success | final mean length (decisions) |
|---|---|---|---|---|
| full (speed 3-12 m/s, accel +-0.1 m/s^2, turn +-1.96 deg/s) | dense | 1667 +- 236 | 1.00 | 24.9 |
| full | sparse | 1583 +- 236 | 1.00 | 26.1 |
| turn only (runs/enc_base) | dense | 2167 +- 425 | 0.96 | 35.4 |
| turn only | sparse | 2417 +- 514 | 0.94 | 39.2 |

Per-scenario scorecard of the six full-control checkpoints (best checkpoint, deterministic policy, 24 episodes per scenario per checkpoint = 144 per row):

| scenario | success | time to event | max / min speed used [m/s] | accel / decel commands per episode | commands against a speed limit | ends as |
|---|---|---|---|---|---|---|
| head_on_port | 1.00 | 217 s | 6.4 / 3.7 | 4.2 / 11.7 | 6.3 | 54 collision, 90 domain violation |
| head_on_starboard | 1.00 | 207 s | 6.8 / 4.1 | 6.6 / 9.5 | 4.4 | 58 / 86 |
| crossing_starboard_bow | 1.00 | 178 s | 8.3 / 5.5 | 8.9 / 6.2 | 0.8 | 105 / 39 |
| crossing_starboard_stern | 1.00 | 218 s | 7.4 / 4.5 | 9.4 / 10.0 | 2.5 | 108 / 36 |
| crossing_port_bow | 1.00 | 145 s | 9.3 / 5.7 | 11.1 / 5.7 | 0.3 | 104 / 40 |
| crossing_port_stern | 1.00 | 213 s | 8.2 / 4.5 | 13.5 / 10.3 | 0.3 | 126 / 18 |
| parallel_port | 1.00 | 60 s | 6.2 / 4.2 | 2.8 / 4.3 | 0.5 | 94 / 50 |
| parallel_starboard | 1.00 | 53 s | 6.5 / 4.7 | 2.1 / 2.7 | 0.3 | 102 / 42 |

1152 of 1152 scorecard episodes succeed (turn only: 0.92-1.00 per scenario).  Reading: the learned attacker uses the speed control selectively rather than sprinting like the scripted intercept.  It slows down to about 4 m/s in head-on encounters to time the meeting, speeds up to 8-9 m/s to cut in front of the bow-crossing targets, and merges into the parallel lanes near the cruise speed, where the 12 m/s sprint of the scripted rule overshot.  The two cells that were weakest turn-only (crossing_port_bow 0.96, parallel_port 0.92) are now solved.  It does not reach the speed limits in most episodes; head-on episodes press against the 3 m/s floor.  Replay and report: `runs/enc_full/enc_mix_passing_full_dense_s2/viz/` (report.html, replay_episode1.gif, chart/grid8.gif).

## Trace routes through the data_prep chain (2026-09-24, mending plan part 1, step 4)

Until now a `--trace` file reached the target through one global projection, a box-5 point average, a rotation on
points 0 to 3 and a segment-heading playback.  On correctly timed arrays that chain moved routes up to 215 m off the
window polyline, left the shapes unscaled while every other length of the standard is multiplied by 0.143, and produced
heading steps of up to 11.5 deg/s at the route vertices (`data_prep/check_out/results.md`, T7).  The chain of
`data_prep/ais_prep.py` (reviewed with Codex, `review/MENDING_PLAN_PART1.md`) is now wired in:

| step | where | what |
|---|---|---|
| load | `scenario_targets.load_trace_shapes` | each trace at full size on its own tangent plane (`ais_prep.ll_to_xy` at its first point), no smoothing |
| accept | `scenario_targets.prepare_trace_routes`, once per environment | `ais_prep.fit_route_to_limit` at the cruise speed, the target's turn rate and the standard's scale (`std.scale` in `EncounterAttackEnv`, `trace_scale` in `ScenarioAttackEnv`, default 1.0): C2 spline table every 5 m, the smallest spatial smoothing that keeps speed x curvature under the limit, rejected when that would move the route by more than 50 m; counts in `env.trace_info` |
| place | `scenario_targets.place_trace_route` (called by `make_attack_scenario` / `make_scenario`) | `ais_prep.place_route` scales the table, turns it onto the scenario course along its own start tangent, `extend_route` continues it straight to the episode length; the scenario dict carries `route` (25 m polyline for the map, the drawings and the reactive presets) and `route_table` |
| sail | `scenario_targets.TargetShip` (fixed track) | position and heading from `ais_prep.route_eval(table, s)`, so the curvature that was checked is the curvature that is sailed; first heading = the table's start tangent |
| report | `EncounterAttackEnv.evaluation`, `ScenarioAttackEnv.evaluation` | `data_shaped`, `route_smooth_m`, `route_max_yaw_deg_s` |

Without a trace nothing changes: the routes are the straight chords as before and `check_attack_scenarios.selftest`
passes at both scales.  Checks in `data_prep/check_env_wiring.py` (output `data_prep/check_out/results_env_wiring.md`,
synthetic windows of the plan check; all eight scenarios x 3 seeds x 2 attackers per scale, yaw rate from the heading the
target reports at every sub-step, `TargetShip.psi_track`): 26 of 32 traces accepted at both scales (23 % after a smoothing
of at most 150 m; the 50 m displacement bound is two-sided); the played heading stays under the limit (arena 1.949 against
1.964 deg/s, maritime 0.279 against 0.281); first heading equals the scenario course within the 0.1 deg rounding of the
metadata; played track within 0.41 m of the placed route; fixed track identical under the hold and the intercept attacker;
the old chain on the same windows has 296 heading steps above 1 deg/s, the wired chain none.  The `runs/` checkpoints were
all trained without a trace, so they are unaffected.  `main_attack_ppo_scen.py --trace_scale` (default 1.0, also through
`run_matrix.py`) selects the length factor of the 2 km arena, 0.143 = the arena scale of the standard.

`Ship_envre_v2` / `v3` are NOT on this chain (external review, round 4): their targets steer on a polyline route by pure
pursuit under a yaw-rate limit and keep `route_from_trace`.  With `data_prep` arrays pass `--trace_dt 20 --smooth_sigma 0`
(the point-count smoothing stays the CLI default for the legacy arrays); the data scenario then takes the smoother's median
speed of the window when `windows_speed_course.npy` lies next to `windows_lonlat.npy`; v3 `straighten_tol` defaults to
50 m and is measured over the whole route before the extension.  Review record: `.aris/traces/research-review/2026-09-21_run03/round4.md`.

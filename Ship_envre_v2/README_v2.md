# Ship_envre_v2 — regenerated attack-scenario pipeline

A second, self-contained build of the critical-scenario (attack / catch-up)
trainer.  It keeps the ShipAI vessel physics and the generated AIS traces of
the original `Ship_envre` folder but follows the structure of
`Desktop/main_attack_ppo.py` (episode loop, periodic evaluation, success rate,
best checkpoint, txt/csv logs, scene figures).  The original folder is left
untouched.

```
Ship_envre_v2/
  ship_dynamics.py        3-DOF vessel model ported from simulator.py (see "physics changes"), rate-limited rudder
  encounter.py            maritime danger criteria: DCPA/TCPA, ship domain, CRI, COLREG encounter type, severity 0-3
  target_ship.py          ship-like target: smoothed route, constant speed/course, limited turn rate, COLREG evasion,
                          automation levels none / manual / assisted / autonomous
  scenarios.py            COLREG scenario families: head_on, crossing_starboard, crossing_port, overtaking, overtaken, data
  grading.py              event grade 0-3 in maritime terms, event score 0-100, COLREG-compliance check, aggregation
  evaluate_agent.py       scorecard of a checkpoint over scenarios x automation levels (csv, md, json, heatmaps)
  selfplay.py             alternate training: attack agent vs navigation policy, navigation agent vs attack policy
  ship_env_v2.py          ShipAttackEnv with task='attack' | 'navigate'
  ppo_agent.py            PPO with the interface main_attack_ppo.py expects from `PPO`
  main_attack_ppo_ship.py regenerated Desktop/main_attack_ppo.py  (attack task)
  main_ppo_ship.py        regenerated Desktop/main_ppo.py         (navigation task, same environment)
  make_synthetic_data.py  stand-in (n,100,2) lon/lat traces so the code runs without the AIS data
  animate_scenes.py       replay episodes as GIF animations + PNG storyboards (agent or pursuit baseline)
  SCENARIO_SETTINGS_CHECK.md  audit of every scenario-generation setting in force (data chain, synthetic set, env)
  checkpoints/            ppo_attack_best.pth / ppo_navigate_best.pth: 400-episode smoke checkpoints used for the animations
  animations/             GIFs and storyboards produced by animate_scenes.py
  data/                   created on first run (synthetic traces)
  runs/<name>/            args.json, models/, train_logs/, eval_logs/, scenes/, train_rl_returns.png
```

## Requirements

Python >= 3.8 with `numpy`, `torch`, `matplotlib`; `tqdm` and `gymnasium`/`gym`
are optional.  No scipy, no shapely, no `viewer` module.  Tested with Python
3.14 + torch 2.x CPU; the Python 3.7 + torch 1.13 environment of the original
`requirements.txt` is also sufficient.

## Running

```
# train on synthetic traces (created in ./data on first run)
python main_attack_ppo_ship.py --save_dir runs/attack_v2 --num_episodes 3000

# train on the generated data of the original pipeline
python main_attack_ppo_ship.py --guideline <..>/dataset_1.csv.npy --mergeline <..>/dataset_2.csv.npy

# evaluate a checkpoint on fresh scenarios (writes scenes + summary json)
python main_attack_ppo_ship.py --mode eval --num_eval 50 \
    --ckpt runs/attack_v2/models/PPO_ShipAttackEnv_seed0_best.pth

# evaluate the scripted pure-pursuit baseline only
python main_attack_ppo_ship.py --mode eval --num_eval 50

# device selection (default auto = cuda if torch sees a GPU, else cpu)
python main_attack_ppo_ship.py --device cpu ...
python main_attack_ppo_ship.py --device cuda ...
```

Input traces are `(n, 100, 2)` arrays of `[longitude, latitude]` exactly as
written by `Train_VAE_full/data_csv2npy.py` and the generators.  A `(100, 2)`
array is accepted as a single sample.

## Scenario families (scenarios.py)

`--scenario` selects how the target is placed relative to the own ship's
initial state, using the COLREG definitions:

| name | rule | geometry sampled per episode |
|---|---|---|
| head_on | Rule 14 | course difference 174-186 deg, meeting in 15-30 min, DCPA offset +-0.3 nm, target 4-9 m/s |
| crossing_starboard | Rule 15, own ship gives way | course difference 60-120 deg, target crosses from starboard |
| crossing_port | Rule 15, own ship stands on | course difference -60 to -120 deg, target crosses from port |
| overtaking | Rule 13, own ship gives way | same course +-15 deg, target at 0.4-0.65 x own speed ahead, own 8-11 m/s |
| overtaken | Rule 13, own ship stands on | same course, target at 1.4-2.2 x own speed astern, own 3-5 m/s |
| data | - | the generated merge line as recorded (reversed): the data set's own geometry |

`mix` draws one family per training episode; evaluation cycles through them
so every family gets the same number of episodes, and `eval_scenarios_*.txt`
reports success rate, mean event score and grade histogram per family.  The
target's route SHAPE is still a generated merge-line sample, rotated to the
scenario course, so the generated data remain in the loop.  A comma list
(`--scenario head_on,crossing_starboard`) restricts training to a subset.

## Arrays from the mended sampling chain (2026-09-24)

`PPO_scenario_generate/data_prep/ais_prep.py` replaces `Train_VAE_full/data_csv2npy.py`
(spline over the row index, noise in degrees) and `Ship_envre/interpolation.py`
(9 s assumption): its `windows_lonlat*.npy` arrays have the old `(n, 100, 2)` shape
but one point every 20 s of real time.  With them pass `--trace_dt 20
--smooth_sigma 0` (the point-count smoothing below was made for the old arrays and
moves correctly timed routes).  When `windows_speed_course.npy` lies next to
`windows_lonlat.npy`, the `data` scenario takes the smoother's median speed of the
window as the target's cruise speed (`speed_source` in the scenario metadata); the
split arrays fall back to route length over trace duration.  The route model of
this build is otherwise unchanged: the target steers on the polyline route by
pure pursuit under `--target_yaw_rate`, it does not play a curvature-checked
table back the way the fixed-track environments of `PPO_scenario_generate` do,
and `route_from_trace` still applies the point-count smoothing whenever
`--smooth_sigma` is above 0 (the default stays 2 for the legacy arrays).  See
`PPO_scenario_generate/review/MENDING_PLAN_PART1.md`, step 4.

## Target-ship model (target_ship.py)

The generated merge-line trace is smoothed into a route (`--smooth_sigma`
points, Gaussian) and the target sails it like a ship: constant speed derived
from the trace (route length / duration, times a per-episode factor), course
held by pure pursuit on a look-ahead point (`--target_lookahead`), turn rate
limited to `--target_yaw_rate` deg/s.  Course and speed are otherwise never
touched.  The only exception is an emergency from the target's own point of
view: when the other ship's DCPA drops below `--alert_dcpa` with
0 <= TCPA < `--alert_tcpa`, the target alters course to starboard by
`--evade_angle` (COLREG Rules 8, 14-17), speed unchanged, and keeps that
until the approach has been clear for `--clear_steps` checks; then pure
pursuit brings it back onto its route.  `--target_evasion none` gives a
passive target.

### Automation levels of the other ship (`--automation`)

Stand-in parameterisations of the IMO MASS degrees; every field can be
overridden (`--latency`, `--alert_dcpa`, `--alert_tcpa`, `--evade_angle`,
`--target_evasion`).

| level | reaction time | evasion trigger (DCPA / TCPA) | manoeuvre | detection | MASS degree |
|---|---|---|---|---|---|
| none | - | never | - | - | passive ship |
| manual | 90 s | 0.5 nm / 12 min | 30 deg starboard | 6 nm | conventional crew (0-1) |
| assisted | 30 s | 1 nm / 15 min | 40 deg starboard | 8 nm | decision support / remote (2) |
| autonomous | 0 s | 1 nm / 20 min | best of 20/40/60 deg by predicted DCPA, speed cut to 70 % | 10 nm | algorithmic (3-4) |
| agent | - | learned | learned | - | policy-driven ship on the full vessel dynamics (`--other_policy <ckpt>`) |

With `--automation agent` the other ship is a `PolicyShip`: the same vessel
model as the own ship, steered by a trained checkpoint (normally the
navigation agent when training the attacker and the attack agent when
training the navigator).  `selfplay.py` alternates the two:

```
python selfplay.py --rounds 3 --out runs/selfplay --scenario mix --num_episodes 600
```

## Danger criteria (encounter.py) and the success / failure rules

| Level | Name | Condition | Default parameters |
|---|---|---|---|
| 1 | CPA warning | DCPA < d_safe and 0 <= TCPA < t_safe | `--d_safe` 926 m (0.5 nm), `--t_safe` 900 s |
| 2 | domain violation | the other ship is inside either ship's domain ellipse | semi-axes `--domain_a` 4 L along course, `--domain_b` 1.6 L across (Fujii-type 8 L x 3.2 L ellipse) |
| 3 | collision | centre distance < collision_L x L | `--collision_L` 1.0 (245 m) |

DCPA/TCPA are computed from the two velocity vectors at every step; the
collision-risk index `CRI = 0.2 ** ((DCPA + v * TCPA) / 2000)` is the formula
that was already in `Ship_envre/Ship_env.py` (`cr_cal`, unused there).  The
COLREG situation (head-on, crossing from starboard/port, overtaking) is
classified from relative bearings and logged.

* **attack** (`main_attack_ppo_ship.py`): success when severity >=
  `--success_severity` (default 2) is held for `--hold_steps` (default 2)
  consecutive decision steps, or on collision.  Failure: leaving the map,
  timeout, or "passed" (CPA behind, range opening beyond 2 d_safe for 6 steps).
  Shaped reward: range closure + DCPA reduction + alignment, a one-off bonus
  when a higher severity is first reached, a time cost and a manoeuvre cost
  (`--rudder_change_cost` per unit rudder-level change, half of it per unit
  throttle change) so the attacker moves like a ship.
* **navigate** (`main_ppo_ship.py`): success when the own ship is within
  `--dest_radius` of the end of its route with heading within
  `--dest_heading_tol` of the route's final course (the reference's
  navigation_target).  Failure: severity 2 or 3, leaving the map, timeout.
  Shaped reward: progress along the route, cross-track penalty, a penalty per
  step at severity 1, time and manoeuvre costs.

The rudder in the vessel model is rate-limited (3 deg/s), so a level change
takes up to 10 s to complete, and the previous rudder level is part of the
observation.  The observation has 19 entries (see `ShipAttackEnv.OBS_NAMES`):
range, bearing, own surge/sway/yaw rate, relative target velocity, course
difference, time, DCPA, TCPA, severity, destination range/bearing,
cross-track error, previous rudder.

## Event grading and agent scorecard (grading.py, evaluate_agent.py)

Every episode is marked with an event grade in maritime terms and a 0-100
event score:

| grade | name | condition |
|---|---|---|
| 0 | safe passage | no CPA warning at any time |
| 1 | close-quarters situation | DCPA < d_safe within t_safe, no domain infringement |
| 2 | domain infringement (near miss) | a ship domain was violated, no collision |
| 3 | collision | centre distance below the collision distance |

score = 40 (1 - min DCPA / d_safe) + 25 severity/3 + 15 max CRI
        + 10 min(time in domain / 5 min, 1) + 10 min(closing speed at min range / 10 m/s, 1)

`evaluate_agent.py` runs a checkpoint (or the scripted baseline) over the
matrix scenario families x automation levels and writes `episodes.csv`,
`scorecard.md/json` and two heat maps.  The ability index it reports:

* attack: success rate over the matrix, mean event score, efficiency
  (baseline steps / agent steps on successes), manoeuvre effort (mean
  rudder-level change per step), robustness (minimum success rate over the
  automation levels), hardest scenario and hardest automation level;
* navigate: safety score (100 - event score), COLREG-compliance rate (give-way
  ships must alter to starboard within 5 min of the first CPA warning,
  stand-on ships must hold course), success rate, effort.

```
python evaluate_agent.py --task attack --ckpt checkpoints/ppo_attack_best.pth --n 10 --out scorecards/attack
python evaluate_agent.py --task navigate --ckpt checkpoints/ppo_navigate_best.pth --n 10 --out scorecards/navigate
python evaluate_agent.py --task attack --ckpt <ckpt> --automation_levels agent --other_policy checkpoints/ppo_navigate_best.pth
```

## What the success rate means

`SuccessRate` in `eval_logs/eval_result.txt` is the fraction of `--num_eval`
evaluation episodes that end with success_flag == 1: for the attack task a
dangerous encounter of the required severity, for the navigation task the
destination reached without any domain violation.  Evaluation episodes use the deterministic policy
(argmax action), fresh random scenarios from the evaluation environment (its
own seed), and the same start distribution as training.  It is the quantity
main_attack_ppo.py computes from `1 in success_flags and -1 not in
success_flags`, and it decides which checkpoint is saved as `_best`.

The `success rate (last N)` printed during training is different: it counts
stochastic training episodes with exploration noise, so it is always lower
than the deterministic evaluation number (0.42 vs 0.95 at episode 300 in the
smoke run).  The baseline row is a scripted policy evaluated on the same scenarios: pure
pursuit (steer to the bearing, full throttle) for the attack task, guideline
following with a starboard turn while the CPA warning is active for the
navigation task.  A learned policy is only interesting once it matches the
baseline or wins on a metric the heuristic does not optimise (steps to the
event, min DCPA, max severity, manoeuvre effort, in `eval_dicts_*.csv`).

## The four fixes (relative to Ship_envre/Ship_env.py)

| # | Original | v2 |
|---|----------|----|
| 1 Observation | `[d, theta, vx, vy, thetadot]`, `d` up to 1e6 m, no bearing, `d` computed from the previous position | 11-dim, scaled to O(1): range, cos/sin bearing to target (body frame), surge, sway, yaw rate, target velocity relative to own (body frame), cos/sin of course difference, time fraction; all from the current state |
| 2 Reward | `(1-d/1e5)^2` every step (~1, almost constant, rewards surviving); phase switch never triggered | closure progress per step (+1 per 100 m), time cost, alignment bonus, +50 capture, −20 out of map, −10 timeout; `--reward_type two_phase` follows the guideline until the target is within `engage_distance`, then chases; `final_attack_reward` is the sparse variant |
| 3 Termination | success at d<300 or observation leaves its box; no time limit; target index wraps through the array | severity ladder from encounter.py (CPA warning / domain violation / collision) decides success (attack) or failure (navigate); leaving the map, timeout and "passed" end the episode; `step` returns `(obs, reward, done, success_flag)` with flag in {1, 0, −1} |
| 4 Geometry / data | each file projected around its own first point, coordinates folded with `abs`; `(n,100,2)` flattened to one line; every episode uses sample 0 | one common origin, signed coordinates; sample dimension kept; a fresh guideline / mergeline pair drawn per episode; traces smoothed into routes; target sails its route at constant speed and course (target_ship.py) instead of interpolating noisy points |

Train and eval draw from the same start distribution (`--own_speed_min/max`,
`--heading_noise_deg`), removing the 2 m/s train vs 10 m/s eval mismatch.

## Physics changes in ship_dynamics.py

The vessel constants and the hull / propeller / rudder force model are the
ShipAI ones.  Four things were changed, each verified with a quick run:

* State integrated in the body frame (Fossen form).  The original rotated
  body accelerations directly into global accelerations.
* `M nu_dot = tau − (C_RB + D) nu`.  The original added the Coriolis matrices.
* Physical advance ratio `J = u/(nD)`, so throttle sets the steady speed
  (0.3 → 3.3 m/s, 0.65 → 7.7 m/s, 1.0 → 11.9 m/s).  In the original the
  terminal speed was independent of throttle.
* A yaw-rate damping moment (`Nr_prime`, `Nrr_prime`) is added and the
  added-mass Coriolis matrix is dropped.  Without it the sway-yaw loop diverges
  after a few hundred seconds and the ship spins, which is what produced the
  "Smashed" (heading out of box) endings in the original.  Full rudder now
  gives a steady turning radius of about 5 ship lengths.

## Reference-structure mapping

| main_attack_ppo.py | main_attack_ppo_ship.py |
|---|---|
| `ownship(...)`, `obstacle(...)`, `navigation_target(...)` | `OwnShipInit`, `TargetShipSpec` (+ `EncounterParams`), route end = navigation target, trace `.npy` files |
| `main_ppo.py` (reach destination, avoid obstacle) | `main_ppo_ship.py` (`--task navigate`, failure = domain violation) |
| `MassTestingEnv(... reward_type, X_LEN, Y_LEN)` | `ShipAttackEnv(guideline, mergeline, ..., reward_type)`; map from trace extent + `--border_margin` |
| `env.step -> state, reward, done, success_flag` | same |
| `env.evaluation()`, `env.show_scenes()`, `env.destination_step` | same names |
| `PPO(...)` from `PPO.py` | `ppo_agent.PPO`, same constructor and methods |
| eval every `--eval_every` episodes, best ckpt by success rate, `eval_dicts_*.csv` | same, plus a pure-pursuit baseline success rate logged before training |

## Animations

```
python animate_scenes.py --ckpt checkpoints/ppo_attack_best.pth --episodes 3 --grid 6 --out animations
python animate_scenes.py --task navigate --ckpt checkpoints/ppo_navigate_best.pth --episodes 3 --out animations
python animate_scenes.py --episodes 2 --out animations                 # attack baseline (pure pursuit)
python animate_scenes.py --task navigate --episodes 2 --out animations # navigation baseline
```

Each single-episode GIF has three parts.  Left, a plain FULL-SCOPE map with
a fixed extent (the region both ships use over the whole episode plus the own
route), white background, routes, tracks with 5-minute marks, the two ships
with heading ticks and the target's domain outline, so routes and the
relative position of the ships are always visible without the camera moving.
Storyboards and grid GIFs use this plain map.  Centre, the detailed display
drawn like a navigation display: sea background
with a nautical-mile graticule, scale bar and north arrow, a camera that
follows the two ships plus an overview inset, true-scale hull silhouettes with
point markers, velocity vectors (6 min of run), fading trails with 1-minute
dots, the target's COLREG sectors (head-on red, starboard orange, port green,
astern grey), its domain ellipse (filled when violated) and d_safe ring, the
predicted positions of both ships at the CPA joined by the DCPA segment, a
star while the target is evading, the destination circle in the navigation
task, and a data box (range, bearing, DCPA, TCPA, CRI, severity, encounter
type, speeds, rudder, automation state).  Right, the side panels: range and
DCPA, TCPA, severity with CRI, rudder and throttle.  A 4-panel storyboard PNG is written
beside every GIF; `--scenario` and `--automation` select what is replayed.  The grid GIF
animates several scenarios simultaneously.  Extra arguments are passed to the
environment (`--capture_radius`, `--guideline`, ...), so any training
configuration can be replayed.  Frames are one per decision step (10 s);
`--fps 8` means 80x real time.  No ffmpeg is needed (Pillow GIF writer).

## Smoke tests (2026-09-14, synthetic routes, CPU, seed 1)

`python ship_env_v2.py` runs the scripted baselines over all scenario
families at two automation levels.  Training runs with `--eval_every 100`:

| run | scenarios | other ship | scripted baseline | eval success by 100-episode block | best checkpoint |
|---|---|---|---|---|---|
| attack, 400 ep | data | assisted | 1.00 | 0.10 / 0.45 / 0.70 / 1.00 | `checkpoints/ppo_attack_best.pth` |
| navigate, 400 ep | data | assisted | 0.95 | 0.20 / 0.40 / 0.20 / 0.25 | `checkpoints/ppo_navigate_best.pth` |
| attack, 500 ep | mix | assisted | 0.83 | 0.00 / 0.17 / 0.33 / 0.29 / 0.33 | `checkpoints/ppo_attack_mix_best.pth` |
| navigate, 500 ep | mix | assisted | 0.83 | 0.00 / 0.00 / 0.00 / 0.04 / 0.00 | `checkpoints/ppo_navigate_mix_best.pth` |
| self-play round 1, 120 ep each | mix | agent | 0.50 (attack) / 0.58 (navigate) | attack 0.50, navigate 0.00 | `checkpoints/ppo_attack_selfplay_r1.pth` |

Scorecards (`scorecards/`, 3 episodes per cell, 6 families x 4 automation levels):

| checkpoint | success rate | mean event score | robustness (min over automation) | hardest family / level |
|---|---|---|---|---|
| attack scripted baseline | 0.89 | 63 | 0.89 | overtaken / none |
| `ppo_attack_mix_best.pth` | 0.21 | 45 | 0.11 | head_on / none |
| navigate scripted baseline | 0.74 (safety 57, COLREG compliance 0.62) | - | 0.61 | head_on / assisted |

Reading: on the single data geometry the attack agent matches the pure
pursuit baseline; with six scenario families and the same 500 episodes it
solves starboard crossing and overtaking against reacting targets but not
head-on or port crossing, and it is worse against a passive target than
against the reacting ones it was trained with.  The navigation agent is not
learning the mixed task at this budget: of 24 final evaluations, 13 end by
leaving the map and 9 by timeout, i.e. route following and the destination
heading tolerance are not yet found.  Both need thousands of episodes per
family, several seeds, and for navigation probably a denser
destination-alignment reward and a curriculum from `data` to `mix`.  All
numbers are for the synthetic routes; the real generated data set the actual
difficulty.

## Environment compatibility (checked 2026-09-14 on this laptop)

| Item | Finding |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB, driver 561.00 (CUDA 12.6 runtime), compute capability 8.9 |
| conda | not installed; no CUDA toolkit (`nvcc`) either |
| Working stack | plain venv, Python 3.14, `torch==2.14.0+cu126` from `https://download.pytorch.org/whl/cu126` (matches the driver); `torch.cuda.is_available()` = True |
| `main_attack_ppo_ship.py --device cuda` | runs end to end (train, checkpoints, eval-only), GPU utilisation ~0 %, 116 MiB VRAM |
| Speed | 300 episodes: CPU 139 s, GPU 190 s. The policy is a 64x64 MLP fed one observation per step, so the per-step host-to-device copies cost more than the compute. Use `--device cpu`; a GPU only pays off with vectorised environments or much larger networks |
| `Desktop/main_attack_ppo.py`, `main_ppo.py` | not runnable anywhere on this machine: they import `PPO` and `env_moving_attack` / `env_moving_obj`, which are not on disk, and `gym` is not installed |
| Original `Ship_envre/ppo_sb3_rl.py` | its pinned stack (Python 3.7, gym 0.14, stable-baselines3 2.0, torch 1.13.1) cannot be installed on Python 3.14 (no torch 1.13 wheel); it needs a conda env with `python=3.7`, plus the missing `viewer.py` and the two data `.npy` files |

Seeding: `--seed` is applied unconditionally (torch, numpy, random, and the
two environments).  The reference script skipped seeding for seed 0, and
without it three identical 300-episode runs ended at 0.95, 0.45 and 0.15
evaluation success, so short runs are dominated by initialisation variance.
Compare configurations over several seeds and more episodes, not one run.


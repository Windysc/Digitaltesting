# Scenario-generation settings: what is currently in force, and what to fix

Checked 2026-09-14 against the code in this repository.  Three layers produce a
scenario: (A) the trace-generation chain of the original project, (B) the
synthetic stand-in generator in this folder, and (C) the environment that turns
two traces into an episode.  The numbers below are the values in the code, not
recommendations, unless marked "fix".

## A. Original trace-generation chain (Train_*/, Evaluation/)

| Step | File | Setting in force | Assessment |
|---|---|---|---|
| A1 CSV → 100 points | `Train_VAE_full/data_csv2npy.py` | cubic spline over the *index* of the CSV rows, resampled to 100 points; columns `longitude_degrees, latitude_degrees` | The 100 points are equally spaced in row index, not in time. The point spacing in seconds is (CSV duration)/99 and is unknown to everything downstream. **Fix:** resample on the timestamp column and record the spacing (it becomes `--trace_dt`). |
| A2 augmentation | same file | `tsgm GaussianNoise.generate(n_samples=500, variance=0.0001)` on raw degrees | TSGM takes `variance` and uses `sigma = sqrt(variance)` = 0.01 degrees per point, independently per point. 0.01 degrees is 1.11 km in latitude and 1.11 km x cos(lat) in longitude (0.6 km at 57 N, 0.85 km at 40 N). A 6 km trace with 1 km independent jitter per point is not a plausible ship track, and the target-speed estimate from consecutive points is dominated by noise. **Fix:** use a variance of about 1e-8 (sigma 1e-4 degrees, 11 m) or add noise in metres after projection, and prefer smooth (low-frequency) perturbations. |
| A3 stray output | same file | `np.save('trajectory_data.npy', lat_lon + speed)` | Adds the speed column to both lon and lat by broadcasting. The file is not read anywhere; delete the line. |
| A4 VAE | `Train_VAE_full/train_vae_full.py` (not marked `(u)`) | min-max over `dataset[:, i]`, i.e. jointly over lon *and* lat per time step; denormalised with mean/std, also joint | Normalisation and denormalisation are different transforms and mix lon with lat, so outputs collapse toward the joint mean. Not usable as written. The README marks only `(u)` files as usable, consistent with this. |
| A5 "cGAN" | `Train_nGAN_full/Train_cGAN/train_cGAN_full(u).ipynb` | TSGM `BetaVAE` (a VAE, not a GAN) fitted on the raw-degree array; decoder output activation `sigmoid` | Sigmoid outputs lie in [0, 1] and can never reproduce longitudes of tens of degrees; there is no normalisation step in the notebook. Either the array was normalised elsewhere or the samples are unusable. **Fix:** per-feature min-max before fitting and the inverse after `generate`, as in A6. |
| A6 U-Net diffusion | `Train_Diff_Full/train_Unet_full(u).py` | per-feature min-max over all samples and steps (correct); 1000 epochs, batch 64, 1000 diffusion steps; corrupts with `x + noise * sqrt(beta_t)` but samples with the DDPM reverse formula; outputs not clipped to [0, 1] | The forward corruption is not the DDPM forward process the sampler assumes (`sqrt(alpha_bar_t) x + sqrt(1 - alpha_bar_t) noise`), so generated traces can drift or exceed the data range. **Fix:** use the standard forward process and clip to [0, 1] before denormalising. Otherwise this is the healthiest generator. |
| A7 DiffTraj | `Train_Diff_Full/train_DiffTraj_full(u).ipynb` | trains on `np.random.randn(766, 91, 2)`; sampler `x = x - pred_noise` | Placeholder data and a non-diffusion sampler. Skeleton only. |
| A8 interpolation | `Ship_envre/interpolation.py` | original spacing assumed 9 s, cubic to 1 s, downsampled to 20 s | This is where the 20 s per point of the environment comes from, but the file is stand-alone and its output (`downsampled_trajectory2.npy`) is not what `Ship_env.py` loads. |
| A9 evaluation | `Evaluation/jsd_matrix.ipynb` | JSD of total distance and of segment lengths, bootstrap | Segment-length JSD is exactly the statistic the A2 noise destroys; with 1 km jitter it will report a large distance for every generator regardless of quality. |
| A10 scenario pairing | `Evaluation/scenario_creation.ipynb` | picks a random sample index in 0..199 from each of two files; file paths are empty | Template only. |

## B. Synthetic stand-in generator (`make_synthetic_data.py`)

Used only because the real `.npy` files are not on this machine.

| Setting | Value |
|---|---|
| samples n, points T | 200, 100 |
| point spacing | 20 s (`trace_dt`) |
| origin | lon 11.9, lat 57.6 (arbitrary) |
| own guideline | heading 45 deg, 3.0 m/s, random bend N(0, 0.15 rad) over the length, lateral start scatter sigma 150 m |
| merge line | parallel lane offset 800 m + N(0, 250 m), 500 to 1500 m ahead, speed 3.5 m/s x U(0.8, 1.2), bend N(0.25, 0.15 rad) |
| per-point noise | sigma 15 m (about 1.3e-4 degrees), i.e. 70 times smaller than A2 |

Difficulty is low on purpose: the pure-pursuit heuristic captures 100 % of
these scenarios, which is what makes the synthetic set a smoke test rather than
a benchmark.

## C. Environment settings (`ship_env_v2.py`, defaults of the two main scripts)

| Setting | Default | Effect |
|---|---|---|
| `--task` | attack (`main_attack_ppo_ship.py`) / navigate (`main_ppo_ship.py`) | success and failure rules, reward |
| `--decision_interval` | 10 s | one action per 10 s, RK4 with 10 substeps, rudder rate 3 deg/s |
| `--trace_dt` | 20 s | seconds per trace point; sets the target's cruise speed (route length / duration); must match the real data (A1) |
| `--smooth_sigma` | 2 points | Gaussian smoothing of both traces into routes |
| target direction | reversed (`--target_forward` to disable) | target starts at the far end of the merge line and comes back |
| `--target_speed_min/max` | 0.8 / 1.2 | per-episode factor on the cruise speed; speed then constant for the episode |
| `--target_yaw_rate`, `--target_lookahead` | 0.5 deg/s, 600 m | course held by pure pursuit, turns limited |
| `--target_evasion`, `--evade_angle` | colreg, 40 deg | starboard alteration when DCPA < `--alert_dcpa` (1852 m) and TCPA < `--alert_tcpa` (900 s); resumes after `--clear_steps` (6) clear checks |
| own start | route point 0, heading of the first segment +- `--heading_noise_deg` 10, speed U(`--own_speed_min` 3, `--own_speed_max` 10) m/s | same distribution for training and evaluation |
| trace pairing | uniform random guideline sample x uniform random merge sample per episode | the generated set is actually used |
| `--d_safe`, `--t_safe` | 926 m, 900 s | CPA warning (severity 1) |
| `--domain_a`, `--domain_b` | 4 L, 1.6 L | ship-domain ellipse (severity 2) |
| `--collision_L` | 1.0 L | collision (severity 3) |
| `--success_severity`, `--hold_steps` | 2, 2 | attack success rule |
| `--dest_radius`, `--dest_heading_tol` | 200 m, 15 deg | navigation success rule |
| `--border_margin` | 1500 m | map = bounding box of all traces + margin; leaving it fails the episode |
| `--max_ep_len` | 300 steps = 50 min | timeout, counted as failure |
| `--reward_type` | shaped | see README "Danger criteria"; `final_attack_reward` = terminal rewards only |
| `--rudder_change_cost` | 0.05 | manoeuvre cost per unit rudder-level change (half per unit throttle change) |
| actions | 5 rudder levels x 3 throttle levels = 15 discrete (`--continuous 1` for Box) | |
| observation | 19 values, all O(1) | `ShipAttackEnv.OBS_NAMES` |
| vessel | ShipAI tanker, L 244.7 m, steady speed 3.3 / 7.7 / 11.9 m/s at throttle 0.3 / 0.65 / 1.0, turning radius about 5 L | |

Original `Ship_envre/Ship_env.py` for comparison: own start speed U(2.0, 2.2)
m/s and heading U(0.5, 0.6) rad in training, 10 m/s in evaluation; target one
raw trace point every 2 steps (20 s); success at 300 m; no time limit; sample 0
always; the DCPA/TCPA/CR functions existed but were commented out of `step`.

Note on the synthetic set: its two lanes are 800 m apart, which is inside the
target's 1 nm alert DCPA, so the target evades for most of a head-on passage
even when the own ship only follows its route.  With real routes, set
`--alert_dcpa` to the passing distance that the traffic actually keeps.

## C2. Scenario families and other-ship automation (scenarios.py, target_ship.py)

| Setting | Default | Effect |
|---|---|---|
| `--scenario` | data | data (recorded merge line, reversed) or head_on / crossing_starboard / crossing_port / overtaking / overtaken / mix / comma list |
| meeting time | 15-30 min (12-25 for overtaking) | how far ahead the meeting point is placed on the own course |
| DCPA offset | +-0.3 nm | lateral offset of the meeting point (0 = exact collision course) |
| course difference | 174-186 / 60-120 / -60 to -120 / +-15 deg | per family, see README |
| target speed | 4-9 m/s, or 0.4-0.65 x own (overtaking), 1.4-2.2 x own (overtaken) | own speed forced to 8-11 m/s (overtaking) or 3-5 m/s (overtaken) |
| route shape | rotated generated merge-line sample, extended straight past the meeting point | `shape_trace=None` gives a straight route |
| map | bounding box of both routes + `--border_margin`, recomputed per episode | |
| `--automation` | assisted | none / manual / assisted / autonomous / agent (see README table) |
| `--other_policy` | - | checkpoint driving the other ship on the full dynamics (`--automation agent`) |
| evaluation | cycles the families | `eval_scenarios_*.txt` per-family success, event score, grade histogram |

The scripted attack baseline (pure pursuit, full throttle) gets a domain
violation in every family except `overtaken`, where the faster target passes
before it can be caught; the scripted navigation baseline fails against a
`manual` target in the port-crossing family (stand-on own ship, late
starboard turn of the target) and against a learned attacker in about half
the episodes.  Those are the cells where a learned agent has to prove itself.

## D. What to set when the real generated data arrive

1. Determine the real point spacing from the CSV timestamps (A1) and pass it as
   `--trace_dt`.  If the spacing differs between the two files, resample one.
2. Check the direction of the merge line relative to the guideline with
   `python animate_scenes.py --episodes 1` before training; use
   `--target_forward` if the target should travel the recorded direction.
3. Look at the scripted baseline success rate printed at the start of
   training.  Near 1.0 means the scenario set is too easy for the learned agent
   to be informative; raise the target speed factor, demand
   `--success_severity 3`, lengthen `--hold_steps`, enlarge the domain, or
   shorten `--max_ep_len`.  Near 0.0 means the routes do not meet at all
   within the map or the time limit.
4. Regenerate the traces with a sane augmentation noise (A2) before drawing any
   conclusion from JSD (A9) or from the trained agent: with sigma 0.01 degrees
   the target ship jumps about 1 km between consecutive points, which the
   environment interpolates into speeds of 50 m/s and more.

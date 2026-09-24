# Check of step 4 of the mending plan for part 1: the route chain wired into the environments

Run 2026-09-24 14:52:15. Code as of: scenario_targets.py 2026-09-24 14:35:16, attack_scenarios.py 2026-09-24 14:39:48, ais_prep.py 2026-09-24 14:35:36. Trace set: check_out/prep_raw_off/windows_lonlat.npy (2026-09-24 14:45:27).

## W1 Loader: one tangent plane per trace, no smoothing

Loaded trace (tangent plane at its first point) against windows_xy.npy (the voyage plane, shifted to the window start), 32 windows: as loaded 4.0 / 33.7 / 62.9 m, after the best-fit rotation about the first point 0.005 / 0.062 / 0.102 m; that rotation is 0.020 / 0.171 / 0.227 deg (median / P95 / max).

- W1 loader shape within 1 m of windows_xy after alignment: **pass** (max 0.102 m)
- W1 frame rotation below 0.5 deg (meridian convergence): **pass** (max 0.227 deg)

## W2 prepare_trace_routes: acceptance under the standard at both scales

- W2 arena: every accepted route within the limit: **pass** (largest 1.963 vs limit 1.964 deg/s)
- W2 maritime: every accepted route within the limit: **pass** (largest 0.281 vs limit 0.281 deg/s)
| scale | length factor | yaw limit [deg/s] | accepted / traces | share needing smoothing | largest smoothing [m] | speed x curvature of accepted routes [deg/s] P50 / P95 / max |
|---|---|---|---|---|---|---|
| arena | 0.143 | 1.964 | 26 / 32 | 23 % | 150 | 0.687 / 1.946 / 1.963 |
| maritime | 1.000 | 0.281 | 26 / 32 | 23 % | 150 | 0.098 / 0.278 / 0.281 |

## W3 EncounterAttackEnv with the trace set: played track against the standard

All scenarios of the mix, 3 seeds each, hold-course and intercept attackers, 200 decisions per episode. The yaw rate is taken from the heading the target reported at every sub-step (TargetShip.psi_track).

- W3 arena: played yaw rate within the limit: **pass** (largest 1.949 vs limit 1.964 deg/s over 24 episodes)
- W3 arena: first heading equals the scenario course (metadata rounded to 0.1 deg): **pass** (0.049 deg)
- W3 arena: track on the placed route: **pass** (max 0.41 m)
- W3 arena: fixed track identical under hold and intercept: **pass** (0.0e+00 m)
- W3 arena: every episode data-shaped: **pass** (24 of 24)
- W3 maritime: played yaw rate within the limit: **pass** (largest 0.279 vs limit 0.281 deg/s over 24 episodes)
- W3 maritime: first heading equals the scenario course (metadata rounded to 0.1 deg): **pass** (0.049 deg)
- W3 maritime: track on the placed route: **pass** (max 0.06 m)
- W3 maritime: fixed track identical under hold and intercept: **pass** (0.0e+00 m)
- W3 maritime: every episode data-shaped: **pass** (24 of 24)
| scale | episodes (hold) | largest yaw rate of the played heading [deg/s] P50 / P95 / max | same from 1 s position chords (discretisation, information only) | limit | first heading error [deg] | track to route [m] P50 / P95 / max | hold vs intercept track difference [m] | outcomes (hold) |
|---|---|---|---|---|---|---|---|---|
| arena | 24 | 0.543 / 1.916 / 1.949 | 0.532 / 1.919 / 1.948 | 1.964 | 0.049 | 0.09 / 0.39 / 0.41 | 0.0e+00 | clear, running, success |
| maritime | 24 | 0.080 / 0.268 / 0.279 | 0.084 / 0.293 / 0.308 | 0.281 | 0.049 | 0.02 / 0.06 / 0.06 | 0.0e+00 | running, success |

## W4 ScenarioAttackEnv (2 km arena) with the trace set

Three families, 3 seeds each, own ship holding course, 100 decisions per episode; yaw rate from the reported heading.

- W4 trace_scale 1.000: yaw within the target turn rate: **pass** (largest 0.181 vs 1.000 deg/s)
- W4 trace_scale 1.000: first heading equals the scenario course (metadata rounded to 0.1 deg): **pass** (0.041 deg)
- W4 trace_scale 0.143: yaw within the target turn rate: **pass** (largest 0.892 vs 1.000 deg/s)
- W4 trace_scale 0.143: first heading equals the scenario course (metadata rounded to 0.1 deg): **pass** (0.041 deg)
| trace_scale | accepted / traces | largest yaw rate of the played heading [deg/s] P50 / P95 / max | limit | first heading error [deg] | data-shaped episodes |
|---|---|---|---|---|---|
| 1.000 | 32 / 32 | 0.047 / 0.177 / 0.181 | 1.000 | 0.041 | 9 / 9 |
| 0.143 | 20 / 32 | 0.458 / 0.892 / 0.892 | 1.000 | 0.041 | 9 / 9 |

## W5 The chain before step 4 against the wired chain, same windows, 6 m/s, unscaled

| chain | largest yaw rate of the playback [deg/s] P50 / P95 / max | heading steps above 1 deg/s (all routes) |
|---|---|---|
| before step 4: box-5 + rotate_translate + segment playback | 1.98 / 8.49 / 11.46 | 296 |
| wired: build_route + place_route + table playback | 0.18 / 0.49 / 0.64 | 0 |

- W5 wired chain has no heading steps above 1 deg/s: **pass** (0 steps)
## W6 Without a trace: routes unchanged (straight chord) and the geometry self-test

- W6 arena: straight chord along the scenario course (metadata rounded to 0.1 deg): **pass** (0.033 deg)
- W6 arena: check_attack_scenarios.selftest: **pass** (all assertions hold)
- W6 maritime: straight chord along the scenario course (metadata rounded to 0.1 deg): **pass** (0.033 deg)
- W6 maritime: check_attack_scenarios.selftest: **pass** (all assertions hold)
- W6 arena episode without trace runs: **pass** (19 decisions, outcome success, data_shaped 0)
## Result

All checks pass.

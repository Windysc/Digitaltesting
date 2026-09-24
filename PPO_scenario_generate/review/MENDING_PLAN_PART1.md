# Mending plan for part 1 (data sampling and smoothing), checked

2026-09-21. Part 1 is the step that turns the private AIS CSV files into the trace arrays for the generators and into routes for the scenario environments. The plan checked here is the one agreed with the external reviewer in `SAMPLING_SMOOTHING_FEEDBACK.md`.

## 1. How the plan was checked

The real data stay private, so the check ran on synthetic voyages whose true path is known at every second.

- **Development fleet.** 24 voyages of 50 to 80 min, speeds 2 to 11 m/s, turns of 0.15 to 0.5 deg/s. Reports follow the Class A schedule with 20 % message loss and reception gaps of 1 to 10 min. The position error has a slowly varying part (3 m, 100 s) plus a white part (2 m), and timestamps are rounded to the second. Three report regimes: as received, one report per 30 s, one per 60 s. A fourth set carries bad fixes, runs of bad fixes, duplicated rows, missing SOG and the AIS "not available" values.
- **Second fleet.** Other seeds and harsher regimes (T12). The finished chain ran on it once, and no setting was changed for it.
- **Old chain.** The original `interpolation_downsample` function was lifted from `data_csv2npy.py` and executed unchanged. The environment parts call the real `scenario_targets.py`, `attack_scenarios.py` and v2 `target_ship.py`.
- **New chain.** A reference implementation, `data_prep/ais_prep.py` (numpy; pandas to read CSV). All tables are in `data_prep/check_out/results.md`, which opens with the time stamps of the code that produced it.
- **External review.** Codex gpt-5.6-sol read the first version of this check (round 3, trace in `.aris/traces/research-review/2026-09-21_run03/round3.md`). It found one real bug and several weak points; all are fixed below. The closing round 4 ran on 2026-09-24 (trace `round4.md`, new session at high effort, after step 4 had been applied): the round-3 fixes are confirmed, the evidence against the long-block holdout is accepted, and `prepare` with the default positions-only settings plus training with `--trace` on its output are cleared. It found two remaining leaks in `prepare` (the fallback SOG-unit vote was taken before the split; an empty training split fell back to all voyages silently), a one-sided displacement test in `fit_route_to_limit`, a missing `--trace_scale` on the scenario training command, and asked for the v2 / v3 route path and the scope of the wiring check to be stated. All were fixed or written down the same day; the table in `round4.md` lists them.

Limits: both fleets are synthetic, so the numbers size each effect and are no acceptance thresholds for real traffic. The TSGM noise could not be executed here; its formula was read from the tsgm 0.1.0 source.

## 2. Verdict

The plan repairs the defect it was written for: time is back in the arrays. Seven of its items needed a change, the check found two further defects in the environments, and the external review found one bug in my first route builder. After those changes the mended arrays follow the truth within about 8 m at the 95th percentile, and the routes play back with a continuous heading and a curvature that matches the one that was checked.

| Quantity (reports as received) | Old chain | Mended chain (default, positions only) |
|---|---|---|
| Time that one array point stands for | 13 to 218 s inside one trace (longest / shortest 13, worst voyage 158) | 20 s, every point |
| Speed read from the array / true speed (P5 to P95) | 0.34 to 1.60 | 0.98 to 1.02 |
| Position error if the array is read with one fixed step | 625 m median, 3.5 km P95 | 3.6 m median, 7.7 m P95, 12.7 m max |
| Course error P95 | 30 deg | 1.0 deg |
| Turn-rate error P95 | 0.22 deg/s (true turns are 0.15 to 0.5) | 0.04 deg/s |
| Largest spline overshoot across a gap | 493 m (1371 m at one report per 60 s) | none, gaps are split |
| v2 `route_from_trace` speed at `trace_dt = 20` / true speed | 1.91 median, 2.35 max | 0.98 |
| Closest approach of a recorded crossing pair (truth 300 m) | 1222 m, the two ships' clocks differ by up to 528 s | 311 m |

With a receiver COG declared and accepted, the position error falls to 2.9 / 6.1 / 10.0 m. At one report per 60 s the default reaches 4.2 / 9.1 / 20.0 m. On the second fleet the default gives 3.5 / 7.1 / 12.4 m under the same noise and 3.6 / 7.7 / 11.9 m with turns of 0.4 to 1.0 deg/s. With the position error doubled and timestamps up to 5 s late it gives 13.9 / 26.2 / 40.4 m; late timestamps move the ship along its track and no smoothing can take that back. Speed stays within 3 % and the course error P95 within 1.9 deg in all four regimes.

## 3. Item by item

| # | Plan item | Verdict | Evidence and change |
|---|---|---|---|
| A1 | Sort by time, drop duplicates, keep SOG and COG | Kept, extended | SOG 102.3 kn and COG 360 deg (AIS "not available") are set to missing. A repeated position is dropped only when the reported speed says the ship moved more than 10 m; slow ships and rounded coordinates repeat positions legitimately. The SOG unit is detected per voyage; where a voyage cannot tell, the majority of the training voyages decides (round 4: the vote used to see every voyage, before the split). Tested on nine input variants (`check_inputs.py`). |
| A2 | Project each voyage to local metres | Kept | Tangent plane per voyage: 0.2 m error at 37 km. My first draft called the old global-mean projection "well under 0.1 %"; at that distance its east scale is off by about 0.9 %. |
| A0 | Split train / val / test by voyage before windowing | **Moved forward** | The split now comes before anything is estimated. The smoothing strength, the COG tests and the augmentation limits use the training voyages only. Groups are vessel ids, folders with `--pairs`, else files. |
| A3 | Split at gaps above 60 s | **Changed** | A gap that hides part of a turn costs 10 m (P95) at 60 s, 21 m at 120 s, 58 m at 180 s and 164 m at 300 s in this fleet. A fixed 60 s would shred data that arrive once per 60 s. Rule: max(60 s, 2 x the voyage's own median report interval), at most 120 s; above that the report flags the voyage as too sparse for a 20 s grid. The report also lists the report interval inside every window, because a window fed by 60 s reports carries interpolated 20 s kinematics. |
| A4 | Three-point outlier rule | **Changed** | A speed threshold fails at short report intervals: 5 m of noise over 2 s already looks like 3.5 m/s. The rule works in distance (plausible speed x interval plus a noise allowance). It removes single bad fixes and runs of up to 12 fixes or 180 s, removes a bad first or last fix, and cuts the track at a jump that never returns. A residual pass catches the rest and takes back good neighbours that the bad fixes had dragged. Under the tested fault model (isolated jumps of 150 to 3000 m, runs of 2 and of 3 to 10 fixes offset by 300 to 800 m, 5 % missing SOG, 1 % sentinels): 251 bad fixes injected, 0 still used, 0 good fixes removed; without the rules the grid error reaches 1170 m. A fleet turning at 0.8 to 1.5 deg/s lost no fix. Offsets below the gate (about 170 m at a 10 s interval for a 6 m/s ship) pass as small bends. |
| A5 | RTS smoother, sigma 10 m, acceleration noise 0.03 m/s2 tuned on a blocked holdout | Smoother kept, **tuning changed** | The smoother is the right tool: turn-rate error P95 on straights is 0.24 deg/s for linear interpolation on timestamps, 0.09 for Savitzky-Golay, 0.04 for RTS. Its strength is weakly identified from the data. The error against the truth is flat over a decade, and position, course and turn rate prefer the same q: 0.001 to 0.01 m2/s3 in every regime, which is also what ship motion implies (0.3 deg/s at 6 m/s is 0.03 m/s2). The plain holdout minimum under-smooths by up to a factor of 10, because AIS position errors are correlated in time and a hidden fix is predicted best by a track that follows the error of its neighbours. Long hidden blocks with guard bands, which the reviewer proposed, do worse (q = 0.1 where the truth prefers 0.003; `check_long_block_holdout.py`). Rule now: short-block holdout on the training voyages, ties within 2 % go to the smaller q, and the result stays inside 0.001 to 0.01. Its cost against the best possible q is at most 21 % on a composite of position, course and turn-rate error; the unbounded plain minimum costs up to 56 %. |
| A5b | SOG / COG from the ship's receiver | Added, **opt-in** | A receiver COG comes from the Doppler velocity and does not share the position error. Used as a second referee for q and as a velocity measurement, it cuts the position error P95 from 7.7 to 6.1 m and the turn-rate error in turns from 0.060 to 0.026 deg/s. The data cannot prove where a COG column comes from: a course computed from moving-averaged positions passes an independence test, and forced into the smoother it pushes the error to 26 to 78 m at P95 and up to 740 m (`check_forced_aiding.py`). So COG counts only when you declare it (`--cog_source receiver`), and then only if three tests pass: independence from the raw fix-to-fix course, an estimated lag of at most 5 s, and hidden fixes predicted at least as well with SOG / COG in the smoother as without. The third test rejects the moving-average cases (ratio 4.3 to 9.3) and keeps a true receiver COG (0.77 to 0.94). A COG that trails the ship by 10 s brings nothing, one that trails by 20 s makes the track worse; both are rejected. |
| A6 | 20 s grid, 100 points, stride 50 | Kept, **one addition** | The grid is anchored to absolute time (multiples of 20 s), so the two ships of one encounter stay on one clock. |
| A7 | Speed and course from the smoother | Kept | Saved next to the positions. |
| A8 | Augmentation: offset SD 10 to 20 m, correlation 120 to 300 s, speed x0.95 to 1.05, at most 4 per window | Kept | 10 to 20 m is the standard deviation of the lateral offset; the largest shift inside a window reaches 27 m (median) and 60 m (max). Both the reviewed values and my first draft keep the turn-rate distribution of the real windows. The correlation time is what matters: a 30 s correlation raises the median turn rate from 0.01 to 0.20 deg/s and invents manoeuvres. Limits come from the training windows. |
| B1 | Route by arc length, 50 m, C1 spline | **Changed twice** | Resampling the polygon of the 20 s points carried its corners into the route (169 heading jumps in my first build). My second build, a C1 Hermite curve on 50 m knots, had the bug the reviewer found: the curvature peaks between the knots were up to 3 times the knot values that the acceptance check read. The route is now a C2 spline through the points, tabulated every 5 m with the heading and curvature of that same curve, and the playback reads the table. Speed x largest table curvature is 0.184 / 0.495 / 0.643 deg/s (median / P95 / max over routes) and the yaw rate sampled from the playback every 0.1 s is 0.185 / 0.495 / 0.643. On an exact arc of radius 1200 m the curvature is within 0.7 % along the whole route. |
| B2 | End tangents from a line fit over 300 m | **Changed** | The route is placed and extended along its own end tangent, so the target's first heading equals the scenario course exactly. Against the true course that tangent is off by 0.35 / 1.04 deg (median / P95) on straights and 0.51 / 1.59 deg in turns; the current three-point rule gives 5.8 / 11.5 deg in turns. The 300 m fit stays as a diagnostic. |
| B3 | Heading from the spline | Kept | Current code: heading steps of 2.0 deg (median), 8.5 (P95), 11.4 (max) per second at the route vertices, 296 steps above 1 deg over 32 routes. Mended: none, largest yaw rate 0.64 deg/s. |
| B4 | Check speed x curvature against 0.5 deg/s | **Changed** | The limit is the turn rate of the scenario standard at its scale: 0.28 deg/s maritime, 1.964 deg/s arena. 0.5 deg/s is the v2 / v3 target default and stays valid there. |
| B5 | One cruise speed for v2 / v3 from the smoother or SOG | Kept | |
| B6 | Auto-straighten tolerance 50 m over the whole route | Kept | Half of the synthetic windows are straight within 50 m; the others hold real turns. |
| C | The private-data checks | Kept, reworked | The report no longer lists a value per voyage and no longer prints a projection figure from which the latitude could be worked out. Section 5 gives reference values. |

### Two defects in the environments that the check found (both removed by step 4 on 2026-09-24)

1. **Box-5 smoothing in `load_trace_shapes` harmed correctly timed arrays too.** With 120 to 180 m between points, a 5-point average cuts the corners of real turns: the route moved 55 m (median), 117 m (P95), 215 m (max) off the window polyline, measured perpendicular to it. The mended route stays within 5 m.
2. **Trace shapes entered the arena unscaled.** Every other length of the standard is multiplied by 0.143; the route was left at full size, so the target turned 7 times more gently than the standard implies (0.19 / 0.49 / 0.64 deg/s against a limit of 1.964). After scaling, 38 % of the synthetic routes exceed the limit. `fit_route_to_limit` applies the smallest spatial smoothing that meets it and accepts 81 %; none of the accepted routes exceeds the limit in playback.

## 4. The plan, in order

**Step 0. See how far the existing arrays are off (timestamps only).**

```
python data_prep/ais_prep.py legacy-check --input <folder with the private CSV files>
```

**Step 1. Run the mended chain on the private CSV files.**

```
python data_prep/ais_prep.py prepare --input <folder> --out <out folder> --pairs
```

- The CSV needs a timestamp column. Column names are found automatically (`longitude_degrees`, `latitude_degrees`, `speed`, plus common names for time, COG and vessel id) or set with `--col_time`, `--col_cog` and the like.
- `--pairs` treats the files of one folder (for example `head-on/1/`) as one encounter: one clock, one split.
- Add `--cog_source receiver` only if the SOG and COG columns are the fields of decoded AIS position reports. If a provider may have computed them, leave it out; the default uses positions only and loses about 1.5 m at P95.
- If the tracks are shorter than 33 min, lower `--window_points` (for example 60) and `--stride`.
- `--augment 2` only if the generators need it. Augmented windows go to the training split and never count as samples.
- Outputs: `windows_lonlat.npy` of shape (n, 100, 2) in the old format, so every existing consumer loads it; the same split into train / val / test; `windows_xy.npy`, `windows_speed_course.npy`, `windows_meta.csv`; `report.md` and `report.json`.

**Step 2. Read `report.md` against section 5.** It can be shared: it holds counts and error statistics only. `windows_meta.csv` names files and stays with the data.

**Step 3. Retrain the generators on `windows_lonlat_train.npy`,** keep `val` and `test` for evaluation. The number of independent samples is the number of voyages, which the report states; overlapping windows of one voyage are one sample seen twice.

**Step 4. Wire the route chain into the environments** (APPLIED 2026-09-24; checked by `data_prep/check_env_wiring.py`, tables in `data_prep/check_out/results_env_wiring.md`):

| File | Change made |
|---|---|
| `scenario_targets.py`, `load_trace_shapes` | Returns each trace at full size on its own tangent plane (`ais_prep.ll_to_xy` at the trace's first point), no smoothing. `smooth_path` stays in the module only for the before / after comparison in the checks. |
| `scenario_targets.py`, new `prepare_trace_routes` | Runs `fit_route_to_limit(xy, speed, turn_rate, scale)` on every trace once, when an environment is built; traces that cannot meet the turn rate within 50 m are left out, and the count is in `env.trace_info`. On the synthetic windows 26 of 32 pass at both scales, 23 % of them after a spatial smoothing of at most 150 m. |
| `scenario_targets.py`, new `place_trace_route`, and `attack_scenarios.make_attack_scenario` / `scenario_targets.make_scenario` | `place_route(table, start, course, scale=std.scale)` then `extend_route` straight to the episode length; the scenario dict carries `route` (a 25 m polyline for the map and the drawings) and `route_table`. `ScenarioAttackEnv` takes `trace_scale` (default 1.0) because its arena is hand-scaled and has no standard. |
| `scenario_targets.TargetShip` | Accepts a route table; the fixed track reads position and heading from `route_eval(table, s)`, the first heading is the table's start tangent. Reactive presets keep projecting on the polyline. |
| `attack_scenarios.EncounterAttackEnv.evaluation` | Adds `route_smooth_m` and `route_max_yaw_deg_s` beside `data_shaped`. |
| `data_prep/check_mending_plan.py`, T7 | Runs the old chain from a local copy of the old loader, so the before / after comparison is unchanged. |
| v2 / v3 | `trace_dt 20` is now true for `data_prep` arrays (pass `--smooth_sigma 0` with them). When `windows_speed_course.npy` lies next to `windows_lonlat.npy`, the data scenario takes the smoother's median speed of the window as the cruise speed (`speed_source` in the scenario metadata); split arrays fall back to route length over duration. `straighten_tol` defaults to 50 m and `route_deviation` measures the whole route before the extension. **Their route path is otherwise unchanged** (round-4 finding): v2 / v3 targets steer on a polyline route by pure pursuit under a yaw-rate limit (`target_ship.py`), they do not play a table back, and `route_from_trace` still applies the point-count smoothing when `--smooth_sigma` is above 0 (the CLI default stays 2 for the legacy arrays). The C2 table chain and the wiring check apply to the two fixed-track environments of `PPO_scenario_generate` only. |
| `main_attack_ppo_scen.py`, `run_matrix.py`, `viz_tool.py` | `--trace_scale` (default 1.0) exposes the length factor of `ScenarioAttackEnv`, so the checked 0.143 configuration can be trained and replayed (round-4 finding). |

What the wiring check shows (same synthetic windows as the plan check; all eight scenarios, three seeds and two attackers per scale, the yaw rate read from the heading the target reports at every sub-step): the played heading never exceeds the standard's turn rate (arena 1.949 against 1.964 deg/s, maritime 0.279 against 0.281), the first heading equals the scenario course within the 0.1 deg rounding of the metadata, the played track lies within 0.41 m of the placed route, the fixed track is identical under the hold and the intercept attacker, and without a trace the routes and the geometry self-test are unchanged. The old chain on the same windows had heading steps of up to 11.5 deg/s at the route vertices; the wired chain has none above 1 deg/s. The check covers the two fixed-track environments of `PPO_scenario_generate`; `Ship_envre_v2` / `v3` are outside it (next row).

**Step 5. Retire the old step.** `data_csv2npy.py` (index spline, degree noise) and `Ship_envre/interpolation.py` (9 s assumption) are replaced by step 1. Arrays made with them should stay out of any set that holds the new ones.

## 5. Reading the private report

These are reference values from the synthetic fleets. Real receivers, real COG noise and other ship types will shift them, so read the pattern first and the number second. Table T8 in `check_out/results.md` shows the patterns on one data set smoothed too much, in range, and too little.

| Section of `report.md` | On the synthetic fleets | What a different picture means |
|---|---|---|
| 1. Removed fixes, cuts | Below 3 % of reports even in the dirty set; no cuts on clean data | Many removals or cuts: look at the source (mixed vessels under one id, a second receiver) before trusting the rest. |
| 2. Report intervals | Median 9 s as received | Median above 60 s: the voyage is flagged as too sparse, and turns between reports are lost. A high share of windows with a median interval above two grid steps: their 20 s kinematics are interpolated; consider `--grid_dt 60` for that source. |
| 2. Consistency line | "consistent" | "INCONSISTENT" is a bug in the run; report it. |
| 3. q used, bound active | Inside 0.001 to 0.01 | A bound that is active together with the signs below: widen `q_bounds` and tell me. |
| 3. Holdout RMS | 5 to 6 m as received, close to the position noise; block RMS within 1.3 x single | Both large: too much smoothing, or bad fixes left in. |
| 3. Course against COG by speed and turning (positions only) | P95 about 1.6 deg on straights and 2.1 deg in turns | The turning value several times the straight value: the track lags in turns. Both high: noise passes through. High only below 3 m/s: normal, COG degrades at low speed. |
| 3. COG tests (when declared) | Independent, lag 0 to 2 s, holdout ratio below 1 | A failed test: the chain already left COG out. A lag above 5 s is a property of the receiver, no fault of the data. |
| 5. Residual RMS and lag-1 autocorrelation | 2 to 4 m; autocorrelation between -0.2 and 0 | Large RMS with autocorrelation above +0.3: too much smoothing. RMS near zero with autocorrelation below -0.6: too little. |
| 4. Speed against SOG | Ratio near 1 (0.514 when SOG is in knots) | Any other ratio: a time or unit error in the export. |
| 6-7. Routes | Raw fixes within about 5 m of the route (P95); start tangent within about 1 deg of the 300 m fit | Larger distances or a start tangent far from the fit: report them. |
| 8. Split | No overlap | The test compares group keys. One voyage stored in several files under different ids is outside its reach; merge such files first. |

## 6. What this check does not cover

- A lasting position offset below the distance gate (about 170 m at a 10 s interval), for example after a receiver change. It passes as a small bend.
- Timestamps that are reception times with a long or variable delay. T12 shows the size: up to 5 s of delay costs about 14 m along the track.
- Whether the COG column of the private files is a receiver value. Only the data source can tell.
- Real vessel behaviour: the fleets turn and change speed in a simple way, and the thresholds above (the 120 s cap, the q range, the healthy ranges) come from them.

## 7. Files

| File | Role |
|---|---|
| `data_prep/ais_prep.py` | Reference implementation: `prepare`, `legacy-check`, smoother, route builder, report |
| `data_prep/synth_ais.py` | Synthetic voyages with a known truth |
| `data_prep/legacy_stage.py` | Runs the original resampling function (needs scipy) |
| `data_prep/check_mending_plan.py` | Tests T0 to T12 |
| `data_prep/check_inputs.py`, `check_long_block_holdout.py`, `check_forced_aiding.py` | Input variants; the evidence on long hidden blocks; a wrongly trusted COG |
| `data_prep/check_env_wiring.py` | Step 4: the wired environments against the standard (W1 to W6) |
| `data_prep/check_out/results.md`, `results_inputs.md`, `results_long_block_holdout.md`, `results_forced_aiding.md`, `results_env_wiring.md` | All tables of this check |
| `review/RESEARCH_REVIEW_ROUND_3.md`, `RESEARCH_REVIEW_ROUND_4.md` | Briefs for the external reviewer; round 4 closes the review and covers step 4 |

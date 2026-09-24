# Review brief, round 4 (closing): your round-3 findings, what changed, one disagreement

Date: 2026-09-21. Same thread and scope: sampling and smoothing of trace data only; the real AIS data stay private. Verify against the files. The code was frozen before the results were regenerated; `data_prep/check_out/results.md` opens with the time stamps of the code that produced it.

## What changed since round 3

All paths are under `C:\Users\ASUS\Desktop\PPO_scenario_generate\`.

| Your finding | Change | Where to verify |
|---|---|---|
| q, units and augmentation limits estimated before the split | Voyages are split first (by vessel id, by folder with `--pairs`, else by file). q, the COG tests and the augmentation limits use the training voyages only. The synthetic voyages now carry one vessel id each, so the split is exercised. | `data_prep/ais_prep.py`, `prepare` |
| Development and validation on the same fleet | T12: the finished chain is run once on a second fleet with other seeds and four regimes (same noise; noise doubled and slower with timestamps up to 5 s late and 35 % loss; that at one report per 30 s; turns of 0.4 to 1.0 deg/s). No setting was changed for it. | `check_mending_plan.py`, `t12`; `check_out/results.md` T12 |
| COG too favourable; guard blind to filtered position-derived COG; lag still chose q | COG is opt-in (`cog_source='receiver'`). It then has to pass three tests: independence, estimated lag of at most 5 s, and hidden fixes predicted at least as well with SOG / COG in the smoother as without. A COG that fails any test neither referees q nor aids. Your moving-average cases are in T3b and in `check_forced_aiding.py`: the first two tests pass them, the third rejects them (ratio 4.3 to 9.3 against 0.77 to 0.94 for a true receiver COG). Forced into the smoother they cost 26 to 78 m at P95 and up to 740 m (`check_out/results_forced_aiding.md`). T3c has receiver lags of 0 to 20 s with speed-dependent noise. The report stratifies course against COG by speed and by turning, with turning labelled from COG itself. | `choose_q`, `cog_diagnostics`; T3b, T3c |
| Truth-preferred q judged on position only | T3 lists position, course and turn-rate RMS and their composite. They prefer the same q in every case. | T3 |
| T4 not the production configuration | Reported timestamps and the chain's q. New figures for a gap that hides part of a turn: 10 m (P95) at 60 s, 21 m at 120 s, 58 m at 180 s (q = 0.01, the value the chain chose on the training voyages). | T4 |
| T5 too easy | Runs of 3 to 10 fixes with a lasting offset, 5 % missing SOG, 1 % AIS "not available" values, and a second fleet turning at 0.8 to 1.5 deg/s to test false removals. The distance rule now removes runs and cuts the track at a lasting jump. | `distance_outliers`; T5, T11 |
| Gap threshold from one fleet-wide median | Per voyage. The report lists the report interval inside each window and the share of windows whose median interval exceeds two grid steps. | `prepare`, report section 2 |
| `place_route` rotated by the fitted course | Placement and extension use the curve's own end tangent. The 300 m fit is a diagnostic. | `place_route`, `extend_route` |
| Route acceptance checked knot curvature (your factor 1.74 median, 2.97 max) | The route is a C2 spline (not-a-knot) tabulated every 5 m with its own heading and curvature; playback reads that table. T7 compares speed x largest table curvature with the yaw rate sampled from the playback every 0.1 s, and reports how many accepted routes exceed the limit after scaling. | `build_route`, `route_eval`, `fit_route_to_limit`; T7, T11 |
| Crash with no fix above the speed mask; sentinels; one unit for all voyages; one missing COG column; stale-repeat rule; split key; circular turning label; one-sided route deviation | Fixed. Input variants are run in `check_inputs.py` (no SOG / COG column, SOG all zero, m/s, mixed units, epoch milliseconds, other column names, 24 vessels shuffled into one file, positions rounded to 1e-5 deg). | `check_out/results_inputs.md` |
| Report leaks latitude and lists per-voyage values | The legacy projection percentage is gone; the per-voyage list is replaced by quantiles. | `_build_report`, `check_out/cli_dirty/report.md` |
| Claims too strong | `review/MENDING_PLAN_PART1.md` is rewritten: thresholds are findings from a synthetic fleet, healthy ranges are reference values, "0 good fixes lost" is tied to the fault model, the SD is the model's own figure, augmentation amplitudes are standard deviations. | the plan |

## One disagreement

You proposed tuning q on contiguous holdouts of 120 to 300 s with guard bands of 60 s. I tested it (`data_prep/check_long_block_holdout.py`, output in `check_out/results_long_block_holdout.md`). It chooses q = 0.1 on the as-received data where position, course, turn rate and their composite all prefer 0.003 (composite cost 2.1 against 1.0), and it is off in the same direction at 30 s and 60 s reports. Its score is set by the few blocks that hide a turn, where a large q lets the end velocities follow the most recent motion. That rewards extrapolation across a hole of four to seven minutes, a different task from smoothing between reports seconds apart.

What the chain does instead: short-block holdout with ties (2 %) to the smaller q, kept inside 0.001 to 0.01 m2/s3. The truth prefers that range in every regime I ran, and it is what ship motion implies (0.3 deg/s at 6 m/s is 0.03 m/s2 of lateral acceleration). The report shows the whole grid and says when a bound is active.

On your point that one shared scalar covariance cannot represent SOG / COG errors: I kept it. A receiver's velocity error is close to isotropic in velocity space; the COG error grows as 1/speed exactly because the velocity error stays the same. Aiding is gated by the three tests instead of by a finer error model.

## Addendum 2026-09-24: step 4 of the plan (route chain wired into the environments) is applied

This round was written on 2026-09-21 but could not be sent (usage limit). Since then step 4 was applied; please cover it as well.

| File | Change | Where to verify |
|---|---|---|
| `scenario_targets.py` | `load_trace_shapes`: full-size traces on their own tangent plane, no smoothing. New `prepare_trace_routes` (runs `ais_prep.fit_route_to_limit` on every trace once per environment, leaves out traces that cannot meet the turn rate within 50 m, reports counts in `env.trace_info`), `place_trace_route` (`place_route` with the standard's scale, `extend_route` straight to the episode length), `route_polyline`, `route_meta`. `TargetShip` accepts a route table; the fixed track reads position and heading from `route_eval`. `ScenarioAttackEnv` takes `trace_scale` (its arena is hand-scaled and has no standard). | `load_trace_shapes`, `prepare_trace_routes`, `place_trace_route`, `TargetShip.__init__ / reset / advance`, `ScenarioAttackEnv.__init__ / _new_scenario` |
| `attack_scenarios.py` | `make_attack_scenario` places the table with `scale=std.scale`; `EncounterAttackEnv` builds the accepted routes at construction and draws from them; `evaluation` adds `route_smooth_m`, `route_max_yaw_deg_s`. | `make_attack_scenario`, `EncounterAttackEnv.__init__ / _new_scenario / evaluation` |
| `Ship_envre_v2/ship_env_v2.py`, `Ship_envre_v3/ship_env_v3.py`, `target_ship_v3.py`, `runner_v3.py` | `load_trace_speeds`: with `windows_speed_course.npy` next to `windows_lonlat.npy` the data scenario takes the smoother's median speed as the cruise speed (`speed_source` in the metadata); split arrays fall back. `straighten_tol` default 50 m; `route_deviation` over the whole route, measured before the extension. | the named functions |
| `data_prep/check_mending_plan.py` | T7 runs the old chain from a local copy of the old loader (`legacy_load_trace_shapes`), so the before / after row keeps its meaning. `results.md` was regenerated with the wired code. | `t7` |
| `data_prep/check_env_wiring.py` | New check W1 to W6, output `check_out/results_env_wiring.md`: loader against `windows_xy.npy` after the meridian-convergence rotation, acceptance counts at both scales, played heading of the target against the turn rate, first heading against the scenario course, track against the placed route, identical fixed track under two attackers, the old chain against the wired one on the same windows, and the no-trace path unchanged (geometry self-test). | the file and its output |

Two things I decided that you may disagree with: (a) the yaw-rate acceptance is done once per environment on the full trace set (deterministic, logged), instead of redrawing per episode; (b) the loader's plane has its north at each trace's first point, so a trace differs from the voyage-plane `windows_xy.npy` by the meridian convergence (up to 0.23 deg, 63 m over 12 km on the synthetic set), which the route chain does not see because a route is placed by its own start tangent.

## Questions

1. Are the round-3 defects fixed? Please check the curvature path and the split-first order in particular.
2. Do you accept the evidence against the long-block holdout, and the bounded rule that replaces it? If not, what would you run?
3. Is the step-4 wiring correct? Look for a place where the old chain still reaches a route (box smoothing, unscaled shapes, segment-heading playback), a mismatch between what the fit checks and what the target sails, and anything in `check_env_wiring.py` that flatters the result.
4. Is anything left that should stop the owner from running `prepare` on the private data, reading the report with section 5 of the plan, and training with `--trace` on the output?

Keep it short. Say plainly what is correct.

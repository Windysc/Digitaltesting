# Review brief, round 3: check of the mending plan for part 1 (sampling and smoothing)

Date: 2026-09-21. Same thread as rounds 1 and 2. Scope is unchanged: sampling and smoothing of trace data only; the real AIS data exist and stay private. Verify everything against the files; my notes are not evidence.

## What happened since round 1

The owner asked for the mending plan of part 1 to be checked. I built a reference implementation of the chain you and I agreed in round 1 and tested it against synthetic voyages with a known truth. The check changed five items of the agreed plan and found two further defects.

## Artifacts

All under `C:\Users\ASUS\Desktop\PPO_scenario_generate\`:

| File | Role |
|---|---|
| `review\MENDING_PLAN_PART1.md` | The checked plan: verdict, item-by-item table, steps, reading guide for the private report |
| `data_prep\ais_prep.py` | Reference implementation (numpy; pandas for CSV): cleaning, distance-form outlier rule, residual pass, CWNA Kalman + RTS smoother on irregular times, choice of q, absolute 20 s grid, windows, split, augmentation, route builder (`build_route`, `place_route`, `fit_route_to_limit`, `route_eval`), aggregate report |
| `data_prep\synth_ais.py` | Synthetic voyages: truth at 1 s, Class A report schedule, message loss, gaps, Gauss-Markov + white position error, second-rounded timestamps with latency, SOG / COG with noise and quantisation, outliers, duplicated rows, a crossing pair |
| `data_prep\legacy_stage.py` | Executes the ORIGINAL `interpolation_downsample` of `data_csv2npy.py` (taken from its source with `ast`) on the synthetic CSV files |
| `data_prep\check_mending_plan.py` | Tests T0 to T11; calls the real `scenario_targets.py`, `attack_scenarios.py`, v2 `target_ship.py` |
| `data_prep\check_out\results.md` | All result tables |
| `data_prep\check_out\cli_dirty\report.md` | Example of the aggregate report the owner would share |

This answers three of your round-1 objections to my probe: the original S1 function is now executed, the real environment code is called for S4 / S6 / S7, and shape damage is measured perpendicular to the polyline. TSGM still cannot be installed here; S2 rests on the source line you accepted in round 2.

## Changes to the plan we agreed, for you to confirm or refute

1. **Gap rule.** max(60 s, 2 x median report interval), capped at 120 s, instead of a fixed 60 s. Basis: T4 (a gap that hides part of a turn costs 9 m P95 at 60 s, 20 m at 120 s, 57 m at 180 s) and the fact that a fixed 60 s shreds data that arrive once per 60 s.
2. **Outlier rule.** Distance form with a noise allowance (a speed threshold raises false alarms at 2 s intervals), end-point tests, and a residual pass that takes good neighbours back. Basis: T5.
3. **Choice of q.** The blocked holdout under-smooths when the position error is correlated in time (T3: plain minimum up to 10 times too large). New rule: smooth positions only, take the q where the smoothed course agrees best with COG, ties within 2 % go to the smaller q; a guard detects a COG computed from consecutive positions and falls back to the holdout with the tie rule (T3b). Velocity aiding with SOG / COG is switched on only when the guard passes.
4. **Route construction.** Resampling the 20 s polygon carried its corners into the route. The route now runs through a C1 curve of the points, its end tangents follow its own knots, and the fit over 300 m only orients the route, with an automatic line / quadratic choice (T7, T11).
5. **Yaw limit and scale.** The limit is the scenario standard's turn rate at its scale, and the route has to be scaled by the same factor as every other length (T7, scale table).

Further defects: box-5 smoothing in `load_trace_shapes` moves correctly timed routes up to 216 m off the window polyline; trace shapes enter the arena unscaled.

## Questions

1. **Is the test harness sound?** Look for circularity or leakage that flatters the mended chain. Points I am unsure about:
   - The COG rule is tested with synthetic COG that has white noise and no lag. Real receivers filter COG over a few seconds and COG degrades at low speed. Does the rule survive that, and what guard or stratification would you add?
   - The position error model (Gauss-Markov 3 m / 100 s plus white 2 m plus timestamp rounding). Is there a realistic error pattern that would break the chain and that I did not test?
   - T3 judges q by the true position RMS at grid points and by the turn-rate error. Is that the right target for generator arrays?
2. **Are changes 1 to 5 right?** Say plainly where one is wrong or overfitted to my synthetic fleet.
3. **Bugs in `ais_prep.py`.** Please read `rts_smooth` (scalar covariance recursion shared by both axes, position-only and position-plus-velocity updates, RTS backward pass), `distance_outliers`, `residual_outliers`, `choose_q`, `build_route`, `route_eval`, `place_route`, `fit_route_to_limit`, and the window / split logic in `prepare`.
4. **Privacy of the report.** Does `report.md` / `report.json` leak positions, absolute times or vessel identity?
5. **Anything in `MENDING_PLAN_PART1.md` that a domain reviewer would reject,** including the reading guide in its section 5.

Keep proposals in scope: no public datasets, no hashing, no infrastructure. Say plainly what is correct.

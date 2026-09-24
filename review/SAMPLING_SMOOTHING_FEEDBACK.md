# Feedback on trace sampling and smoothing (revised after external review)

2026-09-21. Owner's scope: the real AIS data exist but are private, and only the sampling and smoothing code is reviewed.

**Status.** The first draft (2026-09-18) was reviewed by Codex gpt-5.6-sol at high effort; the trace is in `.aris/traces/research-review/2026-09-21_run03/round1.md`. The reviewer confirmed the central defect: time has been removed from the data. It also found errors in the draft's evidence and in several of its parameter values. This version takes those corrections on board.

**Checked on 2026-09-21.** The chains below were built and tested against synthetic voyages with a known truth, and the test itself was reviewed externally. Seven items changed (order of the split, gap rule, outlier rule, choice of the smoothing strength, use of COG, route construction, yaw limit and scale). The result is in `MENDING_PLAN_PART1.md`. Where the two documents differ, that one holds.

**Evidence and its limits.** `sampling_smoothing_probe.py` uses a synthetic AIS-like track whose true path is known. It *re-implements* the S2 and S5 logic rather than calling it, so it shows the size of each effect, not exact values. S7 was not probed.

## Verdict per step

| Step | Code | Finding | Status |
|---|---|---|---|
| S1: resample to 100 points | `Train_VAE_full/data_csv2npy.py:16,33` | The spline runs over the row index, and timestamps never enter. The 100 points therefore have no valid fixed dt, so the 9 s and 20 s used later have no basis. The damage is serious for generator arrays and for the v2/v3 speed estimate, but mostly harmless in the current PPO scenarios, which take only the route *shape*. All 500 samples come from one CSV, so they are not 500 independent tracks. | Confirmed |
| S2: noise augmentation | same file, line 43 | i.i.d. noise in degrees is anisotropic in metres, temporally white, and physically implausible. Its size is **checked against the TSGM source**: tsgm 0.1.0 (the latest on PyPI) has `sigma = variance**0.5`, applied independently per point and per feature. So variance 1e-4 means σ = 0.01°: 1.1 km north–south, 0.6 km east–west at 57.6 N. **Please confirm** that your installed tsgm version matches. | Confirmed for tsgm 0.1.0 |
| S3: `Ship_envre/interpolation.py` | lines 5, 9 | Invents a fixed 9 s spacing. It reads a separate file (`denormalized_dataset1.npy`), so a live S1 → S3 chain is not proven. The ×4 factor in the draft applies only to the synthetic track. | Partly |
| S4/S5: smoothing | `scenario_targets.py:98`, `Ship_envre_v2/target_ship.py:48` | Windows are counted in points, and ends are padded by repeating the last point. On irregular points the physical width is unknown; after regular resampling a point window is fine. **Correction:** the draft's "475 m shift" mostly moves points *along* straight legs. That does not damage the shape of the route. Shape must be measured by perpendicular or Hausdorff distance and by curvature change. | Partly |
| S5: speed from the trace | `target_ship.py:190` | Length divided by an assumed duration is unreliable with an unknown dt, but valid once the timing is correct. The simulator needs one scalar speed, so use the median smoothed speed or the median valid SOG. | Partly |
| S6: start course and extension | `scenario_targets.py:77,86`, `attack_scenarios.py:151` | Uses points 0→3 and the last segment only, so it is sensitive to local noise and to turns near the ends. Minor on clean data. | Confirmed, low |
| S7: fixed-track playback | `scenario_targets.py:270` | The heading jumps at every vertex, so the turn rate is an impulse. The 1 s sub-steps only sample the jump. Harmless on straight routes; wrong for traced routes with bends. | Confirmed |
| Found in review | `scenario_targets.py:113,119` | NaN points are dropped and their neighbours joined, which silently bridges gaps. The projection uses one latitude mean for the whole set (fine only if the data cover a compact area). | New |
| Found in review | `Ship_envre_v3/target_ship_v3.py:38,58` | Deviation is measured over the first 15 km, but the *whole* route is straightened, so a later bend can be erased. The 500 m tolerance also removes real bends. | New |

## Recommended chains

### A. Generator training arrays

1. Sort each vessel's voyage by timestamp and drop duplicates. Keep SOG and COG alongside the positions.
2. Project each voyage separately to local metres.
3. **Split at gaps longer than 60 s** and never interpolate across a split. Adjust the threshold from your Δt histogram.
4. Outliers: remove a middle fix only when *both* adjacent segments are implausible and the direct previous→next segment is plausible. Use SOG/COG to corroborate.
5. **Constant-velocity RTS (Kalman) smoother** on the raw irregular fixes: position σ 10 m per axis, acceleration noise σ 0.03 m/s², tuned within 0.02–0.05 on the blocked holdout below.
6. Sample the smoothed state at **20 s**. Use windows of **100 points** (1980 s) with **stride 50**. Split train/validation/test **by voyage before windowing**.
7. Speed and course come from the smoother's velocity; keep dt = 20 s with the array.
8. Augmentation, if any: at most **4 variants per training window**. Lateral correlated displacement σ 10–20 m, correlation 120–300 s, speed ×0.95–1.05, applied in along-track/cross-track metres. Reject variants outside your data's speed, acceleration and turn-rate envelope.

### B. Scenario routes (geometry only; the scenario sets the speed)

1. Start from the cleaned RTS track and parameterise it by arc length.
2. Resample at **50 m**; fit a C¹ spline and take the heading from its tangent.
3. Start and end directions: a line fit over the first and last **300 m**.
4. Playback evaluates the spline at s = speed × t. Check speed × curvature ≤ 0.5 deg/s, the target's yaw limit. If a route exceeds it, smooth it spatially or leave it out at that speed.
5. v2/v3 data scenarios needing one cruise speed: the median RTS speed or median reliable SOG.
6. Straightening is a design choice. Use explicit straight mode for steady targets. For auto mode, use a **50 m** maximum perpendicular deviation, measured over the whole route.

## Checks you can run on the private data (report aggregates only)

1. **Counts at each stage:** raw reports, duplicates or non-monotonic times, rejected fixes, split tracks, windows kept, and the share of grid points that are interpolated.
2. **Gaps:** a Δt histogram split into moving and stationary periods, and confirmation that no output interval crosses the gap threshold.
3. **Blocked holdout:** hide single fixes and 20–60 s blocks, smooth the rest, and report the median and P95 reconstruction error. This is the main tuning check.
4. **Speed vs SOG and course vs COG:** use circular differences for course, only at adequate SOG, stratified by report interval, speed, and turning vs straight.
5. **Residuals:** along-track and cross-track median, RMS and P95, for interiors and ends separately. Also the RTS innovation bias and lag-1 autocorrelation. *A large structured residual means over-smoothing or lag; a residual near zero means under-smoothing.* (The draft had this backwards.)
6. **Route geometry before and after:** Hausdorff or maximum perpendicular distance, path-length ratio, total turning, maximum curvature, and start/end tangent change.
7. **Playback at scenario speed:** maximum yaw rate, number of heading jumps, and any breach of 0.5 deg/s.
8. **Augmentation:** real vs augmented distributions of speed, acceleration, turn rate, length and curvature. Confirm that each voyage appears in one split only.
9. **Area:** the geographic span of the data and the resulting projection error.
10. **Existing arrays:** for the current `.npy` files, the real time span each point covers, to size how far off they are.

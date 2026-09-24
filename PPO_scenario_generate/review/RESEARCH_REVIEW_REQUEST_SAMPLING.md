# Review brief (round 2, narrowed): trace sampling and smoothing only

Date 2026-09-18. Executor: Claude Opus 5. Reviewer: Codex gpt-5.6-luna. Verify everything against the files; my notes are not evidence.

## Scope set by the owner

- Real AIS data **exist**, but they are sensitive and cannot be shared. Do **not** recommend public datasets or data acquisition.
- Review **only** the current data sampling (resampling, augmentation) and smoothing code. Leave aside generators, the JSD metric, and benchmark design.

## Code in scope

All paths are absolute. The original project is read-only.

| ID | Step | Code |
|---|---|---|
| S1 | CSV lon/lat → 100 points by cubic spline over the **row index** | `C:\Users\ASUS\Desktop\Digitaltesting-main\Digitaltesting-main\Train_VAE_full\data_csv2npy.py`, `interpolation_downsample` |
| S2 | Augmentation: `tsgm GaussianNoise(variance=0.0001)` on raw degrees, 500 copies | same file |
| S3 | Assumes 9 s spacing; cubic `interp1d` to 1 s, then 20 s | `...\Ship_envre\interpolation.py` |
| S4 | Equirectangular projection to metres about the set mean, then `smooth_path` (box window 5, end padding by repetition) | `C:\Users\ASUS\Desktop\PPO_scenario_generate\scenario_targets.py`, `load_trace_shapes`, `smooth_path` |
| S5 | Gaussian `smooth_path(sigma_points=2)` on a point-count kernel; `route_from_trace`: speed = smoothed length / ((n−1)·trace_dt), trace_dt default 20 s | `...\Ship_envre_v2\target_ship.py`, used by `ship_env_v2.py` / `ship_env_v3.py` |
| S5b | v3 `straighten_route`: replaces the route by its chord when its deviation is below 500 m | `...\Ship_envre_v3\target_ship_v3.py` |
| S6 | `rotate_translate`: initial course taken from point 3 − point 0; route extended along the last segment | `PPO_scenario_generate\scenario_targets.py`; also `attack_scenarios.make_attack_scenario` |
| S7 | Fixed-track playback: linear arc-length interpolation, heading = segment heading (steps at vertices) | `scenario_targets.py`, `TargetShip.advance` |

## Evidence

- **Probe:** `C:\Users\ASUS\Desktop\PPO_scenario_generate\review\sampling_smoothing_probe.py`, output in `sampling_smoothing_probe_output.txt`. It runs the chain's own logic on a synthetic AIS-like track with a known truth:
  - 60 min at 57.6 N; 6 m/s, then 4 m/s;
  - one 90° turn at 0.3 deg/s;
  - reports every 2–10 s, plus gaps of 120, 360 and 120 s;
  - 5 m GPS noise.

Summary of the probe output:

| Step | Result |
|---|---|
| S1 | Real time between the 100 points is 17.5–247 s, while downstream assumes a constant 36.4 s. Timing error is median 53 s, max 302 s. Implied speed is 2.0–36.7 m/s against a truth of 4/6. Position error at the true time is median 5 m, max 101 m (spline overshoot across gaps). |
| S1 alternative | Linear interpolation on timestamps to 20 s, then box 3: median 3.4 m, max 35 m, speed p5–p95 3.86–6.13 m/s. |
| S2, variance 1e-4 | Median displacement 1039 m per point. Implied speed median 43 m/s, max 115 m/s. After box 5: residual 442 m, path length ×2.04 (so v2 `route_from_trace` speed ×2.04), initial course error −85°. After Gaussian 2: residual 351 m, ×1.30, −74°. |
| S2, variance 1e-8 | Median 9 m. After box 5: residual 19 m, length ×0.99, course error −2°. |
| S4, clean trace | Box 5 moves points by up to 475 m, exactly where the index spline leaves uneven point spacing (a 1333 m segment beside 140 m ones). End points move 117 m and 74 m. Arc-length resampling before smoothing gives max 91 m (at the ends) and interior max 23 m. |
| S3 | The S1 output has a real spacing of 36.4 s. Read as 9 s, time is compressed ×4 and speeds are inflated ×4. |

## My claimed defects, for you to confirm or refute

- **D1. S1 resamples by index, not time,** so the timing and speed information in the trace is destroyed. Downstream steps that turn points into time (S3, the S5 speed, the trace_dt of 20 s) are wrong by an unknown factor.
- **D2. S2 noise is σ = 0.01° i.i.d. in degrees:** about 1.1 km in latitude and 0.6 km in longitude, so it is also anisotropic. It destroys shape, speed and initial course. It is white noise, not a plausible track perturbation.
- **D3. S4 and S5 smooth over point counts, not distance or time.** On unevenly spaced points this creates large artificial shifts. Edge padding by repetition pulls the ends inward. The window is not tied to a physical scale such as turn radius or seconds.
- **D4. The S5 speed from path length** is biased up by noise and down by smoothing.
- **D5. S6 takes the course from only 3 segments,** so noise sensitivity goes straight into the scenario course. Extending along the last segment has the same weakness.
- **D6. S3's 9 s assumption** is inconsistent with the S1 output.
- **D7. S7's vertex heading steps** mean turn rate is undefined at the vertices. Is this acceptable given the 1 s sub-steps?

## Questions

1. For each of D1–D7: correct, wrong, or partly? Is any of it harmless in practice? Did I miss a defect in these files?
2. What is a sound sampling and smoothing chain for private AIS tracks feeding (a) generative-model training arrays and (b) scenario routes? Name the concrete method and parameters. Candidates:
   - time-based resampling with gap splitting;
   - outlier removal by implied speed;
   - arc-length vs time parameterisation;
   - Savitzky–Golay or a constant-velocity / constant-turn Kalman (RTS) smoother;
   - smoothing-spline weights from GPS error;
   - augmentation in metres with low-frequency (Gaussian-process) perturbations and physical limits.
3. What checks should run on the owner's private data, which I cannot see, so the owner can confirm the chain is sound without sharing the data? Examples: speed and turn-rate sanity against SOG and COG, residual RMS, and a gap histogram.

Keep it concrete and in scope. No hashing, no infrastructure. Say plainly what is correct.

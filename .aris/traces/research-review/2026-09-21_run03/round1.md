# research-review run 03, round 1: sampling and smoothing

- Date: 2026-09-21
- Backend: Codex MCP. Model gpt-5.6-sol. Reasoning effort high: the owner asked for this, which is below the skill's usual xhigh floor. Sandbox read-only.
- threadId: 01a0c314-85c9-7a30-918b-db863338b2fd
- Brief: `review/RESEARCH_REVIEW_REQUEST_SAMPLING.md`, plus `review/SAMPLING_SMOOTHING_FEEDBACK.md`

## Reviewer verdict

The central timestamp defect is right, but the review is not reliable as written.

**D1–D7**

| Item | Verdict |
|---|---|
| D1 | Correct. Mostly harmless in the PPO scenario path, where the trace supplies shape only. |
| D2 | Partly. The probe never called TSGM, so the size of the damage is unproved. |
| D3 | Partly. The 475 m shift is mostly along the track on straight legs, so it is not damage to the route's shape. "Never in points" is too broad once the data are regularly resampled. |
| D4 | Partly. Length divided by duration is valid once the data are correct in time. The simulator needs one scalar speed, so use the median RTS speed or the median SOG. |
| D5 | Correct, low severity. |
| D6 | Partly. There is no proven live link from S1 to S3, and the ×4 factor holds only for the synthetic track. |
| D7 | Correct. |

**Missed defects and flaws in the probe**

- The probe re-implements S2 and S5.
- The linear-interpolation alternative interpolates straight through gaps.
- The "S4/S5" test runs the box filter only.
- S7 was not probed.
- The 500 copies all come from one CSV.
- `load_trace_shapes` bridges gaps where NaN points were dropped.
- The projection uses a single global-mean origin.
- v3 straightens the whole route while measuring deviation over only the first 15 km.

**Recommended chain**

For generator arrays:
- Gap split at 60 s.
- Three-point outlier rule.
- Constant-velocity RTS smoother: σ_meas 10 m, σ_acc 0.03 m/s² (tune within 0.02–0.05).
- 20 s grid, 100-point windows, stride 50, split by voyage.
- At most 4 augmentations per window: lateral 10–20 m, correlation 120–300 s, speed ×0.95–1.05.

For scenario routes:
- Parameterise by arc length, resample at 50 m, C¹ spline.
- End tangents fitted over 300 m.
- Heading taken from the spline.
- Check that v·κ ≤ 0.5 deg/s.
- Auto-straighten tolerance 50 m instead of 500 m.

**Private-data checks**

The proposed checks are not sufficient. The feedback reads the residual backwards: a large residual means over-smoothing or lag. Add:
- counts at each stage;
- a blocked holdout of 20–60 s blocks;
- circular speed and course errors, stratified;
- along-track and cross-track residuals, innovation bias and lag-1 autocorrelation;
- Hausdorff distance and curvature, measured before and after;
- a yaw-rate check at playback;
- real-vs-augmented distributions, split by voyage;
- the projection error given the span of the data.

(The full reviewer text is in the task-notification result of MCP task kkfxk52j2.)

## Executor follow-up

**D2, checked against the TSGM source.** tsgm 0.1.0, the latest release on PyPI (sdist read on 2026-09-21), has:
- `GaussianNoise.generate`: `sigma = variance**0.5`;
- `np.random.normal(mean, sigma, (n_samples, T, F))` when per_channel is set;
- noise added to `X[seeds_idx]`.

So σ = 0.01° i.i.d. per point per feature holds for the latest release. The owner's installed version is still unverified.

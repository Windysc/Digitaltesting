# research-review run 03, round 3: check of the mending plan for part 1

- Date: 2026-09-21
- Call: codex-reply on thread 01a0c314-85c9-7a30-918b-db863338b2fd (gpt-5.6-sol, effort high)
- Brief: `review/RESEARCH_REVIEW_ROUND_3.md`
- Full reply: result of MCP task k6r7bpkap

## Reviewer verdict

A useful engineering prototype that shows convincingly that timestamp-aware resampling beats the legacy row-index chain. Not yet a sound validation of the full method. The three largest problems:
- leakage between tuning and the test split;
- an unreliable COG-independence test;
- a real curvature bug in the route acceptance check.

The reviewer also noted that the stored tables no longer matched the code, because the executor had edited files while the review ran.

## Findings and what was done

| # | Reviewer finding | Executor check | Action |
|---|---|---|---|
| 1 | q, units and augmentation limits were estimated before the voyage split | Correct | Voyages are split first. q, the COG tests and the augmentation limits now use the training voyages only. |
| 2 | Rules were developed and "confirmed" on the same 24 voyages | Correct | T12: the finished chain is run once on a second fleet with other seeds and harsher error regimes; nothing was tuned on it. |
| 3 | Synthetic COG was truth plus white noise: too favourable | Correct | T3c: receiver lag of 0 to 20 s plus speed-dependent velocity noise. |
| 4 | The COG guard misses a COG computed from moving-averaged positions | Confirmed in T3b. If such a COG is forced into the smoother, position error rises to 26–78 m (P95) and 291–740 m (max) (`check_forced_aiding.py`). | COG is now opt-in (`--cog_source receiver`). It must then pass three tests: independence, lag ≤ 5 s, and hidden fixes predicted at least as well with SOG/COG in the smoother as without. The third test rejects the moving-average COG (ratio 4.3–9.3) and keeps a true receiver COG (0.77–0.94). Default: positions only. |
| 5 | A lagged COG still chose q | Correct | A COG that fails any test neither referees q nor aids. |
| 6 | "Truth-preferred q" used position RMS only | Checked: a composite of position, course and turn-rate RMS prefers the same q in every case | T3 now shows all three plus the composite. |
| 7 | T4 used true report times and a fixed q = 0.01 | Correct | T4 now uses reported timestamps and the chain's own q. P95 figures for a gap hiding a turn in the final run (q = 0.01, chosen on the training voyages): 10 m at 60 s, 21 m at 120 s, 58 m at 180 s. |
| 8 | T5 was an easy outlier test | Correct | The fault model now also has runs of 3–10 fixes with a lasting offset, 5 % missing SOG and 1 % AIS "not available" values. A second fleet turns at 0.8–1.5 deg/s to test false removals. The distance rule now removes runs and cuts the track at lasting jumps. |
| 9 | Gap threshold came from one fleet-wide median | Correct | The threshold is set per voyage. The report lists the report interval inside each window and the share of windows whose 20 s kinematics are interpolated. |
| 10 | Suggested tuning q on long holdout blocks (120–300 s) with 60 s guard bands | **Tested and rejected.** It picks q = 0.1 where every truth criterion prefers 0.003. The score is set by the few blocks that hide a turn. (`check_long_block_holdout.py`) | The short-block holdout is kept, with ties going to the smaller q. It is bounded to 0.001–0.01 m²/s³, the range the truth prefers in all regimes and the range ship motion implies. |
| 11 | `place_route` rotated by the fitted course while playback starts along the curve tangent | Correct | Placement and extension now use the curve's own end tangent; the 300 m fit is kept as a diagnostic. |
| 12 | `fit_route_to_limit` checked knot curvature, not the played cubic: peak underestimated ×1.74 (median), ×2.97 (max); 7 "ok" routes violated the limit | **Real bug, confirmed** | The route is now a C² spline tabulated every 5 m with its own heading and curvature, and playback reads the same table. T7 (final run): speed × largest table curvature = 0.184 / 0.495 / 0.643 deg/s; yaw sampled from playback every 0.1 s = 0.185 / 0.495 / 0.643. Accepted routes above the limit: 0 %. |
| 13 | Code defects: crash when no fix passes the speed mask; AIS sentinel values pass; one SOG unit for all voyages; one missing COG column disables aiding everywhere; stale-repeat rule drops legitimate repeats; split key is only a string; turning label is circular; route deviation is one-sided | All correct | All fixed. Sentinels go to NaN; unit per voyage; aiding per voyage; stale only when SOG × dt > 10 m; grouping by vessel id when present; turning label taken from COG itself; both directions of route-to-fix distance. Input variants are tested in `check_inputs.py`. |
| 14 | A shared scalar covariance is wrong for SOG/COG aiding (anisotropic errors) | Partly. A receiver's Doppler velocity error is close to isotropic in velocity space. The angular error grows as 1/speed precisely because the velocity error stays constant. | Fixed σ_v kept; aiding is gated by the three tests instead. |
| 15 | The report leaks latitude (legacy projection percentage plus extent) and lists a value per voyage | Correct | Percentage removed; per-voyage list replaced by quantiles. |
| 16 | Claims too strong: "COG is the reliable referee", "0 good fixes lost", the 120 s cap and q values as thresholds, "0 heading jumps" as sufficient, SD as uncertainty, "10–20 m" as a maximum, the gap check as a "pass" | Correct | The plan is reworded. Thresholds are described as findings from a synthetic fleet, and healthy ranges as reference values. |

## Status after round 3

All results were regenerated from frozen code; `check_out/results.md` opens with the code timestamps of that run. No further round was run. Thread id for resumption: 01a0c314-85c9-7a30-918b-db863338b2fd.

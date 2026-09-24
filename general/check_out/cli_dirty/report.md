# ais_prep report (aggregates only: no positions, no absolute times, no latitude, no file names)

## 1 Counts

| item | n |
|---|---|
| files | 24 |
| voyages | 24 |
| raw_reports | 9407 |
| unusable_rows | 0 |
| non_monotonic_rows | 0 |
| duplicate_timestamps | 94 |
| stale_repeats | 0 |
| sog_or_cog_not_available | 714 |
| removed_by_distance_rule | 245 |
| removed_by_residual_pass | 1 |
| cuts_at_lasting_jumps | 4 |
| segments | 65 |
| segments_too_short | 7 |
| windows | 21 |
| windows_too_slow | 0 |

## 2 Report intervals, gaps and windows

Median interval 9.0 s, 95th percentile 20.0 s. Gap threshold per voyage (median, P95, max): [60.0, 60.0, 60.0] s. Voyages too sparse for this grid: 0.

| interval [s] | share |
|---|---|
| 0 to 3 | 0.069 |
| 3 to 6 | 0.168 |
| 6 to 12 | 0.606 |
| 12 to 30 | 0.139 |
| 30 to 60 | 0.014 |
| 60 to 120 | 0.000 |
| 120 to 600 | 0.003 |
| 600 to inf | 0.000 |

Consistency: longest report interval inside a window minus its voyage threshold = -1.0 s (consistent). Median report interval inside a window (median, P95, max): [7.0, 10.0, 10.0] s. Share of windows whose median interval exceeds two grid steps (their kinematics between reports are interpolated): 0.0.

## 3 Smoothing strength (estimated on the training voyages)

q used: 0.003 m^2/s^3. Rule: holdout, ties to the smaller q. Holdout choice 0.003, COG choice 0.003, bounds [0.001, 0.01], bound active: False. Estimated on: training voyages.
Holdout minimum per voyage (q: number of voyages): {'0.0003': 1, '0.001': 1, '0.003': 5, '0.01': 7, '0.03': 1, '0.1': 2}.
COG: source declared unknown; test of independence from the positions True (difference to the raw fix-to-fix course 3.48 deg RMS, correlation on straights -0.06); estimated lag 0.00 s (course against COG by lag: {'0': 0.836, '2': 0.89, '5': 1.084, '10': 1.566, '15': 2.128, '20': 2.718, '30': 3.922}); used as a referee and for aiding: False.
Holdout RMS with SOG / COG in the smoother over positions only: - (has to be at most 1).

| q | single RMS [m] | single P95 [m] | block RMS [m] | block P95 [m] | n single | n block | positions-only course vs COG RMS [deg] | speed vs SOG RMS [m/s] |
|---|---|---|---|---|---|---|---|---|
| 0.0003 | 7.042 | 14.17 | 8.07 | 16.829 | 601 | 746 | 1.232 | 0.091 |
| 0.001 | 5.556 | 9.917 | 6.077 | 10.982 | 601 | 746 | 0.953 | 0.079 |
| 0.003 | 5.099 | 9.02 | 5.56 | 9.698 | 601 | 746 | 0.836 | 0.078 |
| 0.01 | 4.978 | 8.703 | 5.503 | 9.662 | 601 | 746 | 0.829 | 0.086 |
| 0.03 | 5.012 | 8.568 | 5.622 | 10.145 | 601 | 746 | 0.9 | 0.104 |
| 0.1 | 5.147 | 8.885 | 5.921 | 10.579 | 601 | 746 | 1.057 | 0.14 |

Positions-only course minus COG by speed and by turning (turning is read from COG itself):

| stratum | n | median abs | RMS | P95 abs | mean |
|---|---|---|---|---|---|
| below 3 m/s, straight | 66 | 0.7 | 1.083 | 2.212 | 0.212 |
| below 3 m/s, turning | 38 | 0.681 | 0.903 | 1.904 | 0.119 |
| 3 to 6 m/s, straight | 1639 | 0.507 | 0.779 | 1.555 | -0.057 |
| 3 to 6 m/s, turning | 348 | 0.641 | 0.973 | 1.753 | 0.012 |
| above 6 m/s, straight | 3009 | 0.46 | 0.684 | 1.336 | -0.018 |
| above 6 m/s, turning | 821 | 0.68 | 1.272 | 2.086 | 0.069 |

## 4 Final track against SOG and COG

SOG units detected (voyages): {'kn': 24, 'ms': 0, 'None': 0}; where a voyage cannot tell, the majority (kn) of the training voyages decides. Position speed over SOG (median, P95, max over voyages): [0.514, 0.515, 0.515]. Velocity aiding: False.

| quantity | n | median abs | RMS | P95 abs | mean |
|---|---|---|---|---|---|
| speed_minus_sog_ms | 8465 | 0.05 | 0.078 | 0.146 | -0.001 |
| course_minus_cog_deg | 8464 | 0.5 | 0.819 | 1.537 | -0.002 |

## 5 Residuals, raw fix minus smoothed track [m]

| part | n | median abs | RMS | P95 abs | mean |
|---|---|---|---|---|---|
| along_interior | 8215 | 2.421 | 3.653 | 7.221 | 0.001 |
| cross_interior | 8215 | 1.505 | 2.193 | 4.272 | 0.0 |
| along_ends | 835 | 2.205 | 3.393 | 6.582 | -0.014 |
| cross_ends | 835 | 1.387 | 2.222 | 4.187 | 0.045 |

Lag-1 autocorrelation: along -0.119, cross -0.034. Mean innovation: along 0.823 m, cross 0.023 m.

Grid: 4015 points; position SD of the model (white errors of sigma_p, not a calibrated uncertainty) median 4.0 m, P95 4.9 m; share without a report within one step 0.001.

## 6-7 Windows as scenario routes (median, P95, max over windows)

| quantity | values |
|---|---|
| n | 21 |
| p95_distance_raw_fix_to_route_m | [4.256, 4.779, 4.903] |
| largest_distance_raw_fix_to_route_m | [6.629, 9.452, 9.48] |
| p95_distance_route_to_raw_polyline_m | [3.393, 4.058, 4.384] |
| length_route_over_raw_polyline | [1.003, 1.008, 1.035] |
| total_turning_deg | [43.991, 113.51, 156.077] |
| max_curvature_of_the_played_curve_1_per_km | [0.582, 1.188, 1.618] |
| playback_speed_ms | 6.0 |
| playback_max_yaw_deg_s | [0.2, 0.408, 0.556] |
| yaw_limit_deg_s | 0.5 |
| share_over_yaw_limit | 0.048 |
| share_usable_after_spatial_smoothing | 1.0 |
| start_tangent_against_300m_fit_deg | [0.125, 0.522, 1.116] |
| chord_deviation_m | [228.075, 3683.947, 4596.287] |
| share_straight_within_50m | 0.476 |

## 8 Split

Grouped by vessel id. Windows {'train': 15, 'test': 1, 'val': 5}, groups {'train': 17, 'test': 3, 'val': 4}, overlap between splits: False. One physical voyage stored in several files under different keys is not detected.

## 9 Area

Largest distance from a voyage origin 36.8 km; tangent-plane error there 0.205 m.


# Check of the mending plan for part 1 (sampling and smoothing) on synthetic voyages with a known truth

Run 2026-09-24 14:45:20. Code as of: ais_prep.py 2026-09-24 14:35:36, synth_ais.py 2026-09-24 14:15:05, check_mending_plan.py 2026-09-24 14:40:50. Every table below was produced by this run.

Development fleet: 24 voyages of 50-80 min, speeds 2-11 m/s, turns of 0.15-0.5 deg/s, Class A report schedule with 20 % message loss, reception gaps of 1-10 min, position error = Gauss-Markov 3 m (100 s) + white 2 m + timestamps rounded to the second. Regimes: raw, ds30 and ds60 (a provider that keeps one report per 30 s / 60 s). The rules of the chain were worked out on this fleet; T12 runs the finished chain on a second one.

## T0 What the row-index resampling did to time (original function of data_csv2npy.py)

| regime | shortest step [s] | median step [s] | longest step [s] | longest / shortest | median step / 20 s | median step / 9 s |
|---|---|---|---|---|---|---|
| raw | 13 | 35 | 218 | 13 (worst 158) | 1.73 | 3.84 |
| ds30 | 18 | 35 | 217 | 9 (worst 100) | 1.76 | 3.91 |
| ds60 | 17 | 35 | 167 | 9 (worst 36) | 1.77 | 3.94 |

Each of the 100 legacy points stands for a different time step. A consumer that reads the array with one fixed step sees speeds multiplied by the last two columns on average and by the longest / shortest ratio within a trace.

## T1 Legacy arrays against mended arrays, both against the truth

| chain | position error at the time the point really stands for [m] (P50 / P95 / max) | position error when read with a uniform step [m] | implied speed / true speed (P5 to P95) | speed factor when read at 20 s | course error [deg] | turn-rate error [deg/s] |
|---|---|---|---|---|---|---|
| raw legacy | 4.7 / 11.3 / 492.6 | 625 / 3550 / 7678 | 0.34 to 1.60 | 1.59 | 0.8 / 30.5 / 179.9 | 0.03 / 0.22 / 5.28 |
| raw mended, positions only | 3.6 / 7.7 / 12.7 | 4 / 8 / 13 | 0.98 to 1.02 | 1.00 | 0.3 / 1.0 / 2.4 | 0.01 / 0.04 / 0.11 |
| raw mended, receiver SOG/COG declared | 2.9 / 6.1 / 10.0 | 3 / 6 / 10 | 0.99 to 1.01 | 1.00 | 0.2 / 0.6 / 1.4 | 0.01 / 0.02 / 0.05 |
| ds30 legacy | 5.2 / 20.8 / 1154.2 | 201 / 2747 / 7252 | 0.54 to 1.28 | 1.78 | 0.7 / 20.0 / 180.0 | 0.03 / 0.24 / 5.80 |
| ds30 mended, positions only | 3.9 / 8.3 / 17.0 | 4 / 8 / 17 | 0.98 to 1.02 | 1.00 | 0.3 / 1.1 / 4.9 | 0.01 / 0.04 / 0.21 |
| ds30 mended, receiver SOG/COG declared | 3.4 / 6.9 / 14.7 | 3 / 7 / 15 | 0.99 to 1.01 | 1.00 | 0.3 / 0.8 / 1.5 | 0.01 / 0.03 / 0.08 |
| ds60 legacy | 5.7 / 54.1 / 1370.9 | 243 / 2693 / 6921 | 0.55 to 1.23 | 1.79 | 0.5 / 21.9 / 180.0 | 0.01 / 0.23 / 5.78 |
| ds60 mended, positions only | 4.2 / 9.1 / 20.0 | 4 / 9 / 20 | 0.98 to 1.02 | 1.00 | 0.3 / 1.1 / 4.1 | 0.01 / 0.04 / 0.20 |
| ds60 mended, receiver SOG/COG declared | 3.7 / 7.7 / 14.1 | 4 / 8 / 14 | 0.99 to 1.01 | 1.00 | 0.3 / 0.9 / 2.3 | 0.01 / 0.04 / 0.11 |

## T2 Which smoother: errors of the 20 s arrays against the truth (finite differences of positions, the view a generator has)

Regime raw (q used: positions only 0.01, with SOG/COG 0.003; aiding on: True)

| smoother | position error [m] P50 / P95 / max | same, while turning | speed error P95 [m/s] | course error P95 [deg] | turn-rate error P95 on straights [deg/s] | turn-rate error P95 in turns [deg/s] |
|---|---|---|---|---|---|---|
| linear on timestamps | 4.4 / 9.6 / 17.1 | 4.3 / 8.9 / 14.0 | 0.45 | 2.98 | 0.238 | 0.266 |
| linear + Savitzky-Golay 100 s | 3.9 / 8.5 / 14.1 | 3.6 / 7.6 / 11.5 | 0.23 | 1.63 | 0.094 | 0.099 |
| RTS, positions only | 3.6 / 7.7 / 12.7 | 3.4 / 6.7 / 9.7 | 0.13 | 1.03 | 0.038 | 0.060 |
| RTS with receiver SOG/COG | 2.9 / 6.1 / 10.0 | 2.7 / 4.9 / 7.4 | 0.06 | 0.57 | 0.023 | 0.026 |

Regime ds60 (q used: positions only 0.01, with SOG/COG 0.01; aiding on: True)

| smoother | position error [m] P50 / P95 / max | same, while turning | speed error P95 [m/s] | course error P95 [deg] | turn-rate error P95 on straights [deg/s] | turn-rate error P95 in turns [deg/s] |
|---|---|---|---|---|---|---|
| linear on timestamps | 4.4 / 10.6 / 44.0 | 9.0 / 23.7 / 44.0 | 0.18 | 1.97 | 0.063 | 0.490 |
| linear + Savitzky-Golay 100 s | 4.3 / 10.7 / 38.0 | 9.6 / 21.3 / 38.0 | 0.18 | 1.45 | 0.048 | 0.177 |
| RTS, positions only | 4.2 / 9.1 / 20.0 | 5.3 / 12.9 / 20.0 | 0.14 | 1.11 | 0.029 | 0.092 |
| RTS with receiver SOG/COG | 3.7 / 7.7 / 14.1 | 4.3 / 9.4 / 13.9 | 0.11 | 0.92 | 0.034 | 0.056 |

True turn rates in this fleet are 0.15-0.5 deg/s, so a turn-rate error of 0.1 deg/s or more on straights means the array shows manoeuvres that did not happen.

## T3 Smoothing strength q: what the data can tell (holdout, COG) against what the truth prefers

The truth columns are RMS errors of the 20 s arrays; "composite" is the mean of the three, each divided by its own best value, so 1.000 marks the q that serves position, course and turn rate best together.

Gauss-Markov 3 m + white 2 m, as received. Truth prefers q = 0.003 (position), 0.003 (course), 0.003 (turn rate), 0.003 (composite). Plain holdout minimum 0.03; holdout with ties to the smaller q 0.01; COG choice 0.003; chain uses 0.01 without a declared COG source and 0.003 with receiver COG declared.

| q [m^2/s^3] | true position RMS [m] | true course RMS [deg] | true turn-rate RMS [deg/s] | composite | holdout RMS single / block [m] | positions-only smooth against COG [deg] / SOG [m/s], RMS |
|---|---|---|---|---|---|---|
| 1e-05 | 21.21 | 1.975 | 0.0338 | 4.014 | - | - |
| 0.0001 | 6.78 | 0.880 | 0.0230 | 1.767 | - | - |
| 0.0003 | 4.84 | 0.599 | 0.0182 | 1.283 | 6.82 / 7.95 | 1.17 / 0.090 |
| 0.001 | 4.26 | 0.443 | 0.0142 | 1.020 | 5.42 / 6.36 | 0.91 / 0.080 |
| 0.003 | 4.23 | 0.428 | 0.0140 | 1.000 | 4.95 / 5.71 | 0.81 / 0.079 |
| 0.01 | 4.35 | 0.508 | 0.0196 | 1.206 | 4.78 / 5.41 | 0.81 / 0.088 |
| 0.03 | 4.50 | 0.629 | 0.0300 | 1.559 | 4.77 / 5.41 | 0.89 / 0.106 |
| 0.1 | 4.69 | 0.803 | 0.0474 | 2.122 | 4.87 / 5.77 | 1.05 / 0.139 |
| 1 | 5.12 | 1.248 | 0.0961 | 3.664 | - | - |

same noise, one report per 30 s. Truth prefers q = 0.01 (position), 0.01 (course), 0.003 (turn rate), 0.01 (composite). Plain holdout minimum 0.01; holdout with ties to the smaller q 0.01; COG choice 0.01; chain uses 0.01 without a declared COG source and 0.01 with receiver COG declared.

| q [m^2/s^3] | true position RMS [m] | true course RMS [deg] | true turn-rate RMS [deg/s] | composite | holdout RMS single / block [m] | positions-only smooth against COG [deg] / SOG [m/s], RMS |
|---|---|---|---|---|---|---|
| 1e-05 | 50.71 | 3.460 | 0.0429 | 6.504 | - | - |
| 0.0001 | 16.61 | 1.660 | 0.0314 | 2.785 | - | - |
| 0.0003 | 9.57 | 1.122 | 0.0261 | 1.859 | 14.65 / 15.28 | 1.40 / 0.120 |
| 0.001 | 5.93 | 0.737 | 0.0210 | 1.268 | 8.70 / 9.82 | 1.09 / 0.094 |
| 0.003 | 4.89 | 0.558 | 0.0176 | 1.021 | 6.44 / 7.44 | 0.93 / 0.086 |
| 0.01 | 4.77 | 0.536 | 0.0187 | 1.020 | 5.79 / 6.56 | 0.88 / 0.089 |
| 0.03 | 4.96 | 0.643 | 0.0268 | 1.252 | 5.85 / 6.59 | 0.94 / 0.103 |
| 0.1 | 5.25 | 0.832 | 0.0423 | 1.683 | 6.15 / 7.00 | 1.08 / 0.127 |
| 1 | 5.73 | 1.149 | 0.0705 | 2.445 | - | - |

same noise, one report per 60 s. Truth prefers q = 0.01 (position), 0.01 (course), 0.01 (turn rate), 0.01 (composite). Plain holdout minimum 0.1; holdout with ties to the smaller q 0.03; COG choice 0.03; chain uses 0.01 without a declared COG source and 0.01 with receiver COG declared.

| q [m^2/s^3] | true position RMS [m] | true course RMS [deg] | true turn-rate RMS [deg/s] | composite | holdout RMS single / block [m] | positions-only smooth against COG [deg] / SOG [m/s], RMS |
|---|---|---|---|---|---|---|
| 1e-05 | 69.47 | 4.245 | 0.0468 | 7.787 | - | - |
| 0.0001 | 23.91 | 2.102 | 0.0350 | 3.398 | - | - |
| 0.0003 | 13.76 | 1.433 | 0.0294 | 2.252 | 18.06 / 29.11 | 1.84 / 0.149 |
| 0.001 | 7.85 | 0.922 | 0.0237 | 1.471 | 11.45 / 17.83 | 1.39 / 0.113 |
| 0.003 | 5.70 | 0.653 | 0.0197 | 1.102 | 8.77 / 12.16 | 1.14 / 0.100 |
| 0.01 | 5.21 | 0.564 | 0.0187 | 1.000 | 7.81 / 9.36 | 1.01 / 0.099 |
| 0.03 | 5.34 | 0.614 | 0.0217 | 1.093 | 7.80 / 8.54 | 0.98 / 0.105 |
| 0.1 | 5.53 | 0.690 | 0.0258 | 1.222 | 7.97 / 8.31 | 0.99 / 0.111 |
| 1 | 5.76 | 0.745 | 0.0287 | 1.320 | - | - |

white 10 m (the assumption behind sigma_p = 10 m), as received. Truth prefers q = 0.001 (position), 0.001 (course), 0.0003 (turn rate), 0.001 (composite). Plain holdout minimum 0.001; holdout with ties to the smaller q 0.001; COG choice 0.001; chain uses 0.001 without a declared COG source and 0.001 with receiver COG declared.

| q [m^2/s^3] | true position RMS [m] | true course RMS [deg] | true turn-rate RMS [deg/s] | composite | holdout RMS single / block [m] | positions-only smooth against COG [deg] / SOG [m/s], RMS |
|---|---|---|---|---|---|---|
| 1e-05 | 21.22 | 1.982 | 0.0338 | 3.065 | - | - |
| 0.0001 | 6.79 | 0.903 | 0.0233 | 1.326 | - | - |
| 0.0003 | 5.00 | 0.676 | 0.0198 | 1.026 | 15.92 / 15.96 | 1.21 / 0.093 |
| 0.001 | 4.72 | 0.663 | 0.0205 | 1.013 | 15.56 / 15.40 | 1.03 / 0.090 |
| 0.003 | 5.09 | 0.851 | 0.0309 | 1.307 | 15.59 / 15.52 | 1.09 / 0.104 |
| 0.01 | 5.78 | 1.244 | 0.0576 | 2.003 | 15.80 / 15.97 | 1.42 / 0.143 |
| 0.03 | 6.59 | 1.787 | 0.1015 | 3.074 | 16.17 / 16.76 | 1.97 / 0.207 |
| 0.1 | 7.66 | 2.627 | 0.1787 | 4.871 | 16.80 / 18.26 | 2.95 / 0.314 |
| 1 | 10.22 | 4.886 | 0.4050 | 10.000 | - | - |

Summary: q chosen by each rule, with the composite cost against the truth in brackets (1.000 is the best possible)

| case | truth-preferred q | plain holdout minimum | holdout, ties to the smaller q | same, inside the bounds 0.001 to 0.01 | with receiver COG declared (smaller of holdout and COG, inside the bounds) | fixed q = 0.003 |
|---|---|---|---|---|---|---|
| Gauss-Markov 3 m + white 2 m, as received | 0.003 | 0.03 (1.559) | 0.01 (1.206) | 0.01 (1.206) | 0.003 (1.000) | 0.003 (1.000) |
| same noise, one report per 30 s | 0.01 | 0.01 (1.020) | 0.01 (1.020) | 0.01 (1.020) | 0.01 (1.020) | 0.003 (1.021) |
| same noise, one report per 60 s | 0.01 | 0.1 (1.222) | 0.03 (1.093) | 0.01 (1.000) | 0.01 (1.000) | 0.003 (1.102) |
| white 10 m (the assumption behind sigma_p = 10 m), as received | 0.001 | 0.001 (1.013) | 0.001 (1.013) | 0.001 (1.013) | 0.001 (1.013) | 0.003 (1.307) |

## T3b Can the data tell a receiver COG from one that was computed from the positions?

| COG and SOG columns | COG against the raw fix-to-fix course on straights, RMS [deg] | correlation of the two deviations | test says independent | q by COG | q by holdout | if the owner declares receiver COG: q, aiding, true position error [m] P50 / P95 / max | with the source left undeclared (default) |
|---|---|---|---|---|---|---|---|
| receiver COG (Doppler), as in the other tests | 3.39 | -0.02 | True | 0.003 | 0.01 | 0.003, aiding on: 2.9 / 6.1 / 10.0 | 0.01, aiding off: 3.6 / 7.7 / 12.7 |
| course between consecutive fixes | 0.03 | 1.00 | False | 0.03 | 0.01 | 0.01, aiding off: 3.6 / 7.7 / 12.7 | 0.01, aiding off: 3.6 / 7.7 / 12.7 |
| course over two steps | 2.87 | 0.46 | False | 0.03 | 0.01 | 0.01, aiding off: 3.6 / 7.7 / 12.7 | 0.01, aiding off: 3.6 / 7.7 / 12.7 |
| course between moving averages of 3 fixes | 3.40 | -0.19 | True | 0.1 | 0.01 | 0.01, aiding off: 3.6 / 7.7 / 12.7 | 0.01, aiding off: 3.6 / 7.7 / 12.7 |
| course between moving averages of 5 fixes | 3.39 | -0.13 | True | 0.01 | 0.01 | 0.01, aiding off: 3.6 / 7.7 / 12.7 | 0.01, aiding off: 3.6 / 7.7 / 12.7 |
| course between moving averages of 9 fixes | 3.36 | -0.08 | True | 0.003 | 0.01 | 0.01, aiding off: 3.6 / 7.7 / 12.7 | 0.01, aiding off: 3.6 / 7.7 / 12.7 |
| course between moving averages of 15 fixes | 3.41 | -0.06 | True | 0.003 | 0.01 | 0.01, aiding off: 3.6 / 7.7 / 12.7 | 0.01, aiding off: 3.6 / 7.7 / 12.7 |

The test catches a COG taken between raw fixes. It cannot catch one taken from smoothed positions, so a pass is no proof. That is why COG counts only when the owner declares it a receiver value, and why the default leaves it out.

## T3c A real receiver COG lags and gets noisy at low speed: does its use survive that?

| regime | true COG lag, velocity noise | estimated lag [s] | COG accepted | q by COG | q by holdout | q used | true position RMS [m] at the used q: positions only / with SOG and COG | aiding chosen | true RMS of the chain as run [m] | best true RMS over all q and both modes [m] |
|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0 s, 0.05 m/s | 0 | True | 0.003 | 0.01 | 0.003 | 4.23 / 3.51 | on | 3.51 | 3.49 |
| raw | 5 s, 0.10 m/s | 2 | True | 0.003 | 0.01 | 0.003 | 4.23 / 3.74 | on | 3.74 | 3.71 |
| raw | 10 s, 0.10 m/s | 10 | False | 0.003 | 0.01 | 0.01 | 4.35 / 4.17 | off | 4.35 | 4.12 |
| raw | 20 s, 0.15 m/s | 20 | False | 0.001 | 0.01 | 0.01 | 4.35 / 5.44 | off | 4.35 | 4.23 |
| ds60 | 0 s, 0.05 m/s | 0 | True | 0.03 | 0.03 | 0.01 | 5.21 / 4.50 | on | 4.50 | 4.50 |
| ds60 | 5 s, 0.10 m/s | 2 | True | 0.03 | 0.03 | 0.01 | 5.21 / 4.76 | on | 4.76 | 4.76 |
| ds60 | 10 s, 0.10 m/s | 5 | True | 0.01 | 0.03 | 0.01 | 5.21 / 4.83 | on | 4.83 | 4.83 |
| ds60 | 20 s, 0.15 m/s | 15 | False | 0.01 | 0.03 | 0.01 | 5.21 / 5.48 | off | 5.21 | 5.21 |

A COG that trails the ship by 10 s brings nothing and one that trails by 20 s makes the aided track worse, so the chain takes SOG / COG only when the estimated lag is at most 5 s.

## T4 How long a reception gap may be bridged: error inside the gap against the truth (reported timestamps, q = 0.01 as chosen by the chain)

| gap [s] | on a straight, RTS [m] P50 / P95 | on a straight, linear | gap hides part of a turn, RTS | same, RTS with SOG/COG | same, linear |
|---|---|---|---|---|---|
| 30 | 4 / 8 | 4 / 9 | 4 / 9 | 3 / 7 | 7 / 13 |
| 60 | 4 / 9 | 4 / 9 | 4 / 10 | 3 / 7 | 12 / 27 |
| 90 | 4 / 9 | 4 / 9 | 5 / 15 | 3 / 8 | 23 / 54 |
| 120 | 4 / 11 | 5 / 13 | 6 / 21 | 4 / 9 | 39 / 94 |
| 180 | 5 / 18 | 4 / 22 | 12 / 58 | 6 / 30 | 76 / 204 |
| 300 | 7 / 27 | 5 / 45 | 41 / 164 | 26 / 132 | 182 / 449 |
| 600 | 12 / 78 | 6 / 130 | 134 / 447 | 120 / 404 | 433 / 1057 |

What happens inside a gap that hides part of a turn is lost, and the error there grows roughly with the square of the gap length. These figures belong to this fleet (turns of 0.2 to 0.5 deg/s); they size the effect and are no general bound.

## T5 Outlier rules under a wider fault model

Dirty regime: 0.4 % isolated bad fixes of 150 to 3000 m, one run of two bad fixes, one run of 3 to 10 fixes with a lasting offset of 300 to 800 m, 1 % duplicated rows, 5 % of reports without SOG, 1 % with the AIS "not available" values (102.3 kn, 360 deg).

| setting | bad fixes injected | of those still used | good fixes removed | by the distance rule | by the residual pass | cuts at lasting jumps | SOG / COG values not available | SOG units detected | grid position error [m] P50 / P95 / max |
|---|---|---|---|---|---|---|---|---|---|
| rules on | 251 | 0 | 0 | 245 | 1 | 4 | 714 | {'kn': 24, 'ms': 0, 'None': 0} | 3.4 / 7.3 / 12.3 |
| rules off | 251 | 249 | 0 | 0 | 0 | 0 | 714 | {'kn': 24, 'ms': 0, 'None': 0} | 4.4 / 148.6 / 1169.6 |

Hard manoeuvres must not be mistaken for bad fixes: a second fleet turns at 0.8 to 1.5 deg/s (small craft), with no bad fixes injected.

| fleet | reports | removed by the distance rule | removed by the residual pass | cuts | q used | grid position error [m] P50 / P95 / max | same, while turning |
|---|---|---|---|---|---|---|---|
| turns of 0.8 to 1.5 deg/s | 6536 | 0 | 0 | 0 | 0.01 | 3.7 / 7.6 / 12.9 | 4.7 / 8.1 / 10.0 |

Limits of this test: offsets below the distance gate (about 170 m at a 10 s interval for a 6 m/s ship) pass as small lateral bends; runs longer than 12 fixes or 180 s are cut out as separate segments instead of being removed.

## T6 Two ships of one encounter: is the closest approach preserved?

| chain | closest approach [m] | time of closest approach [s] | note |
|---|---|---|---|
| truth | 300 | 1532 | - |
| mended, common absolute 20 s grid | 311 | 1540 | 129 common samples; both files in split ['train'] |
| legacy, each ship to 100 points by row index, paired by index | 1222 | ship 1 at 1350 s, ship 2 at 1743 s | the two clocks differ by up to 528 s |

## T7 Scenario routes: the environment code before step 4 (until 2026-09-24) against the mended route, on correctly timed windows

| route chain | largest yaw rate of the playback at 6 m/s [deg/s] P50 / P95 / max | samples above 1 deg/s (all routes) | largest distance between route and window polyline [m] | start heading against the true course [deg] |
|---|---|---|---|---|
| before step 4: load_trace_shapes (box 5) + rotate_translate (points 0 to 3) + TargetShip.advance (segment heading), 1 s steps | 1.99 / 8.49 / 11.42 | 296 | 54.6 / 117.0 / 215.1 | 0.23 / 4.35 / 15.40 |
| mended: build_route (C2 spline, 5 m table) + route_eval, sampled every 0.1 s | 0.19 / 0.49 / 0.64 | 0 | 0.6 / 2.9 / 5.0 | 0.32 / 1.10 / 1.42 |

Check that the curvature which is tested is the curvature which is sailed: speed x largest table curvature = 0.184 / 0.495 / 0.643 deg/s, largest yaw rate sampled from the playback every 0.1 s = 0.185 / 0.495 / 0.643 deg/s (P50 / P95 / max over routes).

Start course on 839 window starts (67 of them in a turn)

| start-course rule | error when the window starts on a straight [deg] P50 / P95 / max | error when it starts in a turn [deg] |
|---|---|---|
| points 0 to 3 of the window (current rule) | 0.26 / 0.84 / 7.46 | 5.83 / 11.54 / 14.02 |
| tangent of the spline at its first point (used to place the route) | 0.35 / 1.04 / 2.14 | 0.51 / 1.59 / 3.72 |
| least-squares fit over the first 300 m (diagnostic) | 0.29 / 0.98 / 4.02 | 0.61 / 1.62 / 4.04 |

Scale: a real-size route inside the arena standard (scale 35 / 244.74 = 0.143, yaw limit 1.964 deg/s, 6 m/s), yaw rate from the playback of the scaled route sampled every 0.1 s

| route geometry | largest yaw rate at 6 m/s [deg/s] P50 / P95 / max | share above the arena limit |
|---|---|---|
| unscaled, as the current code places it | 0.19 / 0.49 / 0.64 | 0.00 |
| scaled by 0.143 like every other length of the standard | 1.29 / 3.46 / 4.45 | 0.38 |
| scaled, after fit_route_to_limit: routes it accepts (81 % of all) | 0.68 / 1.95 / 1.96 | 0.00 |

## T8 How the private-data checks read: the same data smoothed too much, in range, and too little (positions only)

| setting | TRUE position error P95 straight / turning [m] | TRUE turn-rate error P95 on straights [deg/s] | residual RMS along / cross [m] | residual lag-1 autocorrelation along / cross | holdout RMS single / block [m] | course against COG, RMS [deg] | course against COG P95, straight / turning (mean over speed bands) [deg] |
|---|---|---|---|---|---|---|---|
| too much smoothing, q = 1e-6 | 116.4 / 278.0 | 0.051 | 40.3 / 73.9 | 0.39 / 0.51 | 94.6 / 101.0 | 5.36 | 9.6 / 17.9 |
| q = 1e-4 | 12.1 / 26.1 | 0.017 | 5.5 / 6.7 | 0.27 / 0.41 | 9.9 / 11.3 | 1.46 | 2.0 / 4.4 |
| q = 3e-3 | 7.5 / 6.8 | 0.023 | 3.6 / 2.2 | -0.11 / -0.02 | 5.0 / 5.7 | 0.81 | 1.6 / 2.1 |
| too little smoothing, q = 10 | 9.9 / 8.0 | 0.284 | 1.5 / 0.8 | -0.72 / -0.71 | 6.1 / 11.4 | 2.69 | 6.4 / 8.4 |

## T9 Compatibility with the consumer of the arrays (EncounterAttackEnv)

The mended windows are written in the old (n, 100, 2) lon / lat layout.  The environment loads them through the route chain of ais_prep.py and plays the accepted routes back.  (Until 2026-09-24 this table also held the speed that the route model of the removed Ship_envre_v2 build read from the arrays.)

EncounterAttackEnv(trace = mended windows, scale = maritime) runs: 285 decisions, outcome clear, data_shaped = 1, routes accepted 26 of 32.

## T10 Augmentation values: reviewed plan against the first draft

| set | largest lateral shift inside a window [m] P50 / P95 / max | abs acceleration [m/s^2] P50 / P95 / max | abs turn rate [deg/s] P50 / P95 / max | mean of the per-window largest turn rate |
|---|---|---|---|---|
| real windows | - | 0.0018 / 0.0060 / 0.0339 | 0.013 / 0.172 / 0.611 | - |
| reviewed plan: offset SD 10-20 m, correlation 120-300 s, speed x0.95-1.05 | 26.7 / 47.4 / 59.5 | 0.0018 / 0.0059 / 0.0358 | 0.013 / 0.173 / 0.614 | 0.20 |
| first draft: offset SD 20-50 m, correlation 300-600 s, speed x0.9-1.1 | 45.7 / 97.1 / 146.9 | 0.0019 / 0.0059 / 0.0379 | 0.013 / 0.171 / 0.606 | 0.19 |
| for contrast: offset SD 20 m with a 30 s correlation | 52.6 / 68.5 / 80.7 | 0.0025 / 0.0085 / 0.0404 | 0.197 / 0.638 / 1.163 | 0.80 |

## T11 Unit checks of the building blocks on exact geometry

| check | result |
|---|---|
| projection round trip within 40 km of the origin | largest error 3.68e-04 m |
| tangent plane against closed forms, 0.1 deg away (about 11 km north, 6 km east) | north: 11137.140 m against meridian arc 11137.146 m (chord effect 5.1e-07); east: 5979.0778 m against N cos(lat) sin(dlon) = 5979.0778 m |
| route through an exact arc of radius 1200 m sampled every 120 m | curvature x radius between 0.9987 and 1.0070 over the whole route; start heading error 0.0102 deg; largest distance from the circle 0.003 m |
| playback of that arc at 6 m/s, every 0.1 s (exact yaw rate 0.2865 deg/s) | yaw rate between 0.2861 and 0.2884 deg/s |
| route through a straight line | largest curvature 0.00e+00 1/m, length 12369.317 m (exact 12369.317) |
| place_route (scale 0.143, course 0.7 rad) and extend_route by 200 m | start (100.0, -50.0), first heading 0.7000 rad, curvature x 1200 x 0.143 = 0.9999, heading step at the joint 0.0e+00 rad, length 188.8 -> 389.0 m |
| smoother on a noise-free constant-velocity track, irregular times | largest position error 1.74e-02 m, velocity error 1.44e-03 m/s |
| scalar covariance recursion against a dense-matrix Kalman / RTS smoother (positions and velocities measured) | largest difference 4.55e-13 m in position, 7.99e-15 m/s in velocity |
| distance rule on a run of 7 fixes offset by 500 m | removed fixes [15, 16, 17, 18, 19, 20, 21], cuts 0 |
| distance rule on a lasting jump of 900 m | removed fixes [], cut after fix [19] |

## T12 The finished chain on a second fleet it was not developed on

Other seeds, 20 voyages per regime, no setting changed. Regime A: as the development fleet. Regime B: position error twice as large and slower (Gauss-Markov 6 m / 300 s plus white 4 m), timestamps late by up to 5 s, 35 % message loss. Regime C: B with one report per 30 s (Class B like). Regime D: turns of 0.4 to 1.0 deg/s with the development noise.

| regime | mode | q used | aiding | windows | position error [m] P50 / P95 / max | same, while turning | implied speed / true speed (P5 to P95) | course error [deg] P50 / P95 / max | turn-rate error P95 straight / turning [deg/s] |
|---|---|---|---|---|---|---|---|---|---|
| A | positions only (default) | 0.01 | off | 30 | 3.5 / 7.1 / 12.4 | 3.2 / 6.7 / 12.4 | 0.98 to 1.02 | 0.37 / 1.30 / 3.10 | 0.049 / 0.062 |
| A | receiver COG declared | 0.003 | on | 30 | 2.9 / 5.9 / 10.1 | 2.8 / 6.3 / 8.0 | 0.99 to 1.01 | 0.20 / 0.68 / 1.47 | 0.026 / 0.027 |
| B | positions only (default) | 0.001 | off | 21 | 13.9 / 26.2 / 40.4 | 13.9 / 26.8 / 38.5 | 0.98 to 1.02 | 0.31 / 1.05 / 4.35 | 0.020 / 0.075 |
| B | receiver COG declared | 0.001 | on | 21 | 13.7 / 24.7 / 34.8 | 12.4 / 23.7 / 34.8 | 0.99 to 1.01 | 0.24 / 0.72 / 1.98 | 0.017 / 0.039 |
| C | positions only (default) | 0.003 | off | 4 | 12.9 / 27.2 / 43.6 | 13.2 / 27.2 / 32.9 | 0.97 to 1.03 | 0.45 / 1.84 / 8.21 | 0.039 / 0.102 |
| C | receiver COG declared | 0.003 | on | 4 | 12.7 / 25.2 / 38.9 | 12.1 / 22.8 / 29.7 | 0.98 to 1.02 | 0.32 / 1.20 / 3.59 | 0.029 / 0.065 |
| D | positions only (default) | 0.01 | off | 29 | 3.6 / 7.7 / 11.9 | 4.1 / 8.7 / 11.8 | 0.98 to 1.02 | 0.33 / 1.19 / 3.94 | 0.042 / 0.083 |
| D | receiver COG declared | 0.01 | on | 29 | 3.0 / 6.4 / 10.0 | 3.4 / 6.9 / 9.0 | 0.99 to 1.01 | 0.22 / 0.72 / 2.09 | 0.035 / 0.032 |


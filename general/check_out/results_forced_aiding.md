# A wrongly trusted COG, and the third COG test

| COG and SOG columns | test 1 independent | test 2 lag [s] | test 3 holdout RMS with SOG/COG over positions only | accepted when declared | true position error, default [m] P50 / P95 / max | declared | forced into the smoother |
|---|---|---|---|---|---|---|---|
| receiver COG (Doppler) | True | 0.0 | 0.94 | True | 3.6 / 7.7 / 12.7 | 2.9 / 6.1 / 10.0 | 3.0 / 6.3 / 10.3 |
| receiver COG, one report per 60 s | True | 0.0 | 0.77 | True | 4.2 / 9.1 / 20.0 | 3.7 / 7.7 / 14.1 | 3.7 / 7.7 / 14.1 |
| course between consecutive fixes | False | 0.0 | - | False | 3.6 / 7.7 / 12.7 | 3.6 / 7.7 / 12.7 | 4.4 / 10.5 / 46.6 |
| course over two steps | False | 2.0 | - | False | 3.6 / 7.7 / 12.7 | 3.6 / 7.7 / 12.7 | 3.9 / 8.6 / 29.6 |
| course between moving averages of 3 fixes | True | 2.0 | 4.33 | False | 3.6 / 7.7 / 12.7 | 3.6 / 7.7 / 12.7 | 7.1 / 26.4 / 290.6 |
| course between moving averages of 5 fixes | True | 2.0 | 4.50 | False | 3.6 / 7.7 / 12.7 | 3.6 / 7.7 / 12.7 | 8.1 / 35.6 / 348.6 |
| course between moving averages of 9 fixes | True | 2.0 | 6.07 | False | 3.6 / 7.7 / 12.7 | 3.6 / 7.7 / 12.7 | 10.7 / 54.9 / 622.7 |
| course between moving averages of 15 fixes | True | 2.0 | 9.28 | False | 3.6 / 7.7 / 12.7 | 3.6 / 7.7 / 12.7 | 13.4 / 77.9 / 740.2 |

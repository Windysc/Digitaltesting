# Input variants

| input variant | runs | voyages | windows | SOG units found | grouped by | stale repeats dropped | q used | aiding | true grid position error [m] P50 / P95 / max |
|---|---|---|---|---|---|---|---|---|---|
| as generated | yes | 24 | 32 | {'kn': 24, 'ms': 0, 'None': 0} | vessel id | 0 | 0.003 | on | 2.9 / 6.1 / 10.0 |
| no SOG and no COG column | yes | 24 | 32 | {'kn': 0, 'ms': 0, 'None': 24} | vessel id | 0 | 0.01 | off | 3.6 / 7.7 / 12.7 |
| SOG all zero | yes | 24 | 32 | {'kn': 0, 'ms': 0, 'None': 24} | vessel id | 0 | 0.01 | off | 3.6 / 7.7 / 12.7 |
| SOG in m/s | yes | 24 | 32 | {'kn': 0, 'ms': 24, 'None': 0} | vessel id | 0 | 0.003 | on | 2.9 / 6.1 / 10.0 |
| SOG in knots for even files, m/s for odd files | yes | 24 | 32 | {'kn': 12, 'ms': 12, 'None': 0} | vessel id | 0 | 0.003 | on | 2.9 / 6.1 / 10.0 |
| timestamps as epoch milliseconds | yes | 24 | 32 | {'kn': 24, 'ms': 0, 'None': 0} | vessel id | 0 | 0.003 | on | 2.9 / 6.1 / 10.0 |
| columns named BaseDateTime, LON, LAT, SOG, COG | yes | 24 | 32 | {'kn': 24, 'ms': 0, 'None': 0} | vessel id | 0 | 0.003 | on | 2.9 / 6.1 / 10.0 |
| 24 vessels shuffled into one file with an mmsi column | yes | 24 | 32 | {'kn': 24, 'ms': 0, 'None': 0} | vessel id | 0 | 0.003 | on | 2.9 / 6.1 / 10.0 |
| positions rounded to 1e-5 deg | yes | 24 | 32 | {'kn': 24, 'ms': 0, 'None': 0} | vessel id | 0 | 0.003 | on | 2.9 / 6.1 / 9.9 |

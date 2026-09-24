# Long hidden blocks (120-300 s, 60 s guard bands) as a referee for q

| regime | q by long-block RMS (ties to the smaller q) | its plain minimum | q by long-block median | truth-preferred q (composite of position, course, turn rate) | composite cost of the long-block RMS choice | composite cost of the short-block choice inside the bounds (the chain) |
|---|---|---|---|---|---|---|
| raw | 0.1 | 0.3 | 0.0003 | 0.003 | 2.122 | 1.206 |
| ds30 | 0.03 | 0.1 | 0.0001 | 0.01 | 1.252 | 1.020 |
| ds60 | 0.1 | 0.3 | 0.001 | 0.01 | 1.222 | 1.000 |

The long-block score is set by the few blocks that hide a turn, where a large q lets the end velocities follow the most recent motion. It rewards extrapolation across a hole of four to seven minutes, which is a different task from smoothing between reports that are seconds apart.

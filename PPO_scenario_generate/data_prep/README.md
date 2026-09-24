# data_prep: timestamp-correct sampling and smoothing of AIS tracks

Reference implementation and tests of the mending plan for part 1. The plan, its check and the reading guide for the report are in `../review/MENDING_PLAN_PART1.md`.

## On the private data

```
python ais_prep.py legacy-check --input <folder with the CSV files>
python ais_prep.py prepare --input <folder> --out <out folder> --pairs
```

`legacy-check` reads timestamps only and prints the real time that each of the 100 index-resampled points stands for. `prepare` writes the arrays and `report.md` / `report.json`. The report holds counts and error statistics only (no positions, no absolute times, no latitude, no file names), so it can be shared while the data stay private. `windows_meta.csv` names files and stays with the data.

Needs numpy and pandas. The CSV must carry a timestamp column.

| Option | Default | Meaning |
|---|---|---|
| `--grid_dt` | 20 s | step of the common absolute time grid (the environments' `trace_dt`) |
| `--window_points`, `--stride` | 100, 50 | window length and step in grid points |
| `--gap_s` | auto | split threshold per voyage: max(60 s, 2 x its median report interval), at most 120 s |
| `--q` | auto | smoothing strength: blocked holdout on the training voyages, ties to the smaller q, kept inside 0.001 to 0.01 m2/s3 |
| `--cog_source` | unknown | `receiver` when SOG / COG are the ship's own GNSS velocity (decoded AIS messages). COG then also referees q, and SOG / COG enter the smoother, if three tests pass |
| `--use_sogcog` | auto | `off` never uses SOG / COG in the smoother, `on` forces it past the three tests (for `check_forced_aiding.py` only; the console and report section 4 say when aiding was forced) |
| `--pairs` | off | files in one folder are one encounter and share a split |
| `--augment` | 0 | variants per training window, at most 4 |
| `--col_time`, `--col_lon`, `--col_lat`, `--col_sog`, `--col_cog`, `--col_id` | found by name | column names |

Outputs: `windows_lonlat.npy` (n, points, 2) as lon, lat in the old format; `windows_lonlat_train/val/test.npy`; `windows_xy.npy` in metres from each window's first point; `windows_speed_course.npy`; `windows_meta.csv`; `report.md`, `report.json`.

Routes for the environments: `build_route`, `place_route`, `extend_route`, `fit_route_to_limit`, `route_eval` in `ais_prep.py`.

## In the environments (step 4 of the plan, applied 2026-09-24)

`scenario_targets.py` and `attack_scenarios.py` read a `--trace` file through this chain: each trace is projected on its own tangent plane, `prepare_trace_routes` runs `fit_route_to_limit` once per environment at the cruise speed, the target's turn rate and the standard's length scale (traces that cannot meet the limit within 50 m are left out; `env.trace_info` holds the counts), `place_route` turns the table onto the scenario course along its own start tangent, and the fixed-track target reads position and heading from the table (`route_eval`). The scenario metadata carries `data_shaped`, `route_smooth_m` and `route_max_yaw_deg_s`.

```
python ../main_attack_ppo_enc.py --trace <out folder>/windows_lonlat_train.npy ...
```

For `Ship_envre_v2` / `Ship_envre_v3` pass `--trace_dt 20 --smooth_sigma 0` with these arrays; when `windows_speed_course.npy` lies next to `windows_lonlat.npy`, the data scenario takes the smoother's median speed of the window as the cruise speed.

## Re-running the check

```
python check_mending_plan.py fleet
<python with scipy> legacy_stage.py check_out/csv check_out/legacy.npz
python check_mending_plan.py tests
python check_inputs.py
python check_long_block_holdout.py
python check_forced_aiding.py
python check_env_wiring.py
```

The second line runs the original resampling function of `data_csv2npy.py` and needs scipy (the traffic venv has it). `tests` writes `check_out/results.md` and takes about 10 min; the file opens with the time stamps of the code that produced it. `check_inputs.py` runs the chain on input variants (no SOG / COG, other units, other column names, several vessels per file). `check_long_block_holdout.py` holds the evidence against tuning on long hidden blocks, `check_forced_aiding.py` shows what a wrongly trusted COG does and how the third COG test catches it. `check_env_wiring.py` checks the wired environments against the standard (W1 to W6, `check_out/results_env_wiring.md`) and exits with 1 when a check fails.

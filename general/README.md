# general: extraction and generation

All code of the pipeline in one folder: the extraction chain (`ais_prep.py`) with its checks, the generation (the encounter environment, the PPO attacker, matrices and visualisation) and the trace generators (`generators/`). Run the scripts from this folder or from any work directory; logs and `runs/` land in the current directory. Requirements: `../requirements.txt`.

## Extraction: `ais_prep.py`

Reference implementation and tests of the mending plan for part 1 (data extraction). The plan, its check and the reading guide for the report are in `../review/MENDING_PLAN_PART1.md`.

On the private data:

```
python ais_prep.py legacy-check --input <folder with the CSV files>
python ais_prep.py prepare --input <folder> --out <out folder> --pairs
```

`legacy-check` reads timestamps only and prints the real time that each of the 100 index-resampled points of the old data step stood for. `prepare` writes the arrays and `report.md` / `report.json`. The report holds counts and error statistics only (no positions, no absolute times, no latitude, no file names), so it can be shared while the data stay private. `windows_meta.csv` names files and stays with the data.

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

### In the environments (step 4 of the plan, applied 2026-09-24)

`scenario_targets.py` and `attack_scenarios.py` read a `--trace` file through this chain: each trace is projected on its own tangent plane, `prepare_trace_routes` runs `fit_route_to_limit` once per environment at the cruise speed, the target's turn rate and the standard's length scale (traces that cannot meet the limit within 50 m are left out; `env.trace_info` holds the counts), `place_route` turns the table onto the scenario course along its own start tangent, and the fixed-track target reads position and heading from the table (`route_eval`). The scenario metadata carries `data_shaped`, `route_smooth_m` and `route_max_yaw_deg_s`.

```
python main_attack_ppo_enc.py --trace <out folder>/windows_lonlat_train.npy --scenario mix --dcpa_band passing --control full
```

## Generation

**Training.** `main_attack_ppo_enc.py` trains the attacker on the eight encounter scenarios of `attack_scenarios.py` under the collision standard and the lifecycle of `encounter_standard.py`:

```
python main_attack_ppo_enc.py --scenario mix --dcpa_band passing --control full --scale arena \
       --num_episodes 3000 --eval_every 250 --num_eval 24 --seed 1 --save_dir runs/enc_full_s1
```

`--scenario` takes one scenario, a group (`head_on`, `crossing_starboard`, `crossing_port`, `crossing`, `parallel`, `mix`) or a comma list; `--dcpa_band` one of `collision`, `close`, `passing` (a safe passage if nobody manoeuvres, so the attacker has to create the danger), `wide`, `v2`; `--scale arena` (the standard scaled to the 2 km arena, ship length 35 m) or `maritime` (245 m); `--control full` (speed, acceleration, turn rate, course change and position area, each with a limit) or `turn`; `--trace` shapes the target routes with trace arrays. The reward `final_attack_reward` is terminal only, `dense_attack_reward` adds shaping. `main_attack_ppo_scen.py` is the earlier three-family variant in the hand-scaled arena (`--scenario head_on|crossing_starboard|crossing_port|mix`, `--automation fixed|manual|autonomous`, `--trace`, `--trace_scale`).

**Matrices.** `run_matrix.py` runs scenario x reward x seed matrices in parallel and aggregates them (`matrix_runs.csv`, `matrix_summary.csv`, curves):

```
python run_matrix.py --out runs/enc_full --tasks enc --control full --enc_scenarios mix --bands passing \
       --seeds 1 2 3 --episodes 3000 --eval_every 250 --num_eval 24
python run_matrix.py --out runs/enc_full --summary-only
```

**Geometry and baselines.** `check_attack_scenarios.py --out runs/enc_check --bands passing close v2 --episodes 12 --grid` runs the geometry self-test at both scales (the initial DCPA is the drawn value, the target passes on the named side), the scripted baselines hold course, intercept and pursuit per scenario and band, and the catalogue figure.

**Visualisation.** `viz_tool.py all <run> --episodes 20` writes training and evaluation curves, the evaluation records, a replay fan, a chart-style GIF and storyboard of the first episode and an HTML report under `<run>/viz/`; `viz_tool.py chart <run> --episodes N --grid M` renders one GIF per episode and a grid; `viz_tool.py compare <run> <run2> ...` overlays runs. `chart_viz.py` is the renderer (full-scope map, navigation display with hulls, COLREG sectors, CPA prediction and data box, side panels).

## Checks

```
python check_mending_plan.py fleet                                        # synthetic CSV fleets with a known truth
<python with scipy> legacy_stage.py check_out/csv check_out/legacy.npz    # the removed row-index resampling, for comparison
python check_mending_plan.py tests                                        # T0 to T12, about 10 min, writes check_out/results.md
python check_inputs.py
python check_long_block_holdout.py
python check_forced_aiding.py
python check_env_wiring.py                                                # W1 to W6, exits with 1 when a check fails
```

`tests` writes `check_out/results.md` and `results.json`; the file opens with the time stamps of the code that produced it. `check_inputs.py` runs the chain on input variants (no SOG / COG, other units, other column names, several vessels per file). `check_long_block_holdout.py` holds the evidence against tuning on long hidden blocks, `check_forced_aiding.py` shows what a wrongly trusted COG does and how the third COG test catches it. `check_env_wiring.py` checks the wired environments against the standard (`check_out/results_env_wiring.md`).

## Files

| File | Role |
|---|---|
| `ais_prep.py` | extraction chain (clean, split, outliers, gaps, Kalman / RTS smoothing, 20 s grid, windows, report) and route chain (spline table, turn-rate check, placement, playback) |
| `synth_ais.py` | synthetic AIS-like voyages with a known truth for the checks |
| `legacy_stage.py` | the original row-index resampling of the removed `data_csv2npy.py`, run on the synthetic fleets for the comparison |
| `check_mending_plan.py`, `check_inputs.py`, `check_long_block_holdout.py`, `check_forced_aiding.py` | checks of the data step; results in `check_out/` |
| `check_env_wiring.py` | checks of the route chain inside the environments |
| `env_moving_obj.py` | world model and base `MassTestingEnv`: own ship with rate limits, obstacles, nine discrete actions, 1 s sub-steps |
| `env_moving_attack.py` | the attack task on that world model |
| `encounter_standard.py` | collision standard: CPA warning, ship domain, collision, CRI, COLREG type, event grading, lifecycle, `CollisionStandard(scale)` |
| `scenario_targets.py` | scenario geometry, `TargetShip` (fixed track or reactive presets), trace routes through `ais_prep.py`, `ScenarioAttackEnv` |
| `attack_scenarios.py` | the eight attack scenarios, the DCPA bands, `EncounterAttackEnv`, the scripted baselines |
| `PPO.py` | the PPO agent (actor-critic, clipped surrogate); CPU by default, `PPO_DEVICE=cuda` for the GPU |
| `gym.py` | shim so that `import gym` in the trainers resolves to gymnasium |
| `main_attack_ppo_enc.py` | attack trainer on the encounter scenarios (the main trainer) |
| `main_attack_ppo_scen.py` | attack trainer of the earlier three-family arena variant |
| `run_matrix.py` | seed and scenario matrices with aggregation |
| `check_attack_scenarios.py` | geometry self-test, scripted baselines, catalogue figure |
| `chart_viz.py`, `viz_tool.py` | chart-style renderer; curves, evaluation records, replays and reports |
| `generators/` | the trace generators (below) |
| `MASS_TESTING_ENV_REBUILD.md` | design and verification log of the environments and of the training runs |

## Trace generators (`generators/`)

Standalone scripts written for Python 3.10 with the TSGM stack (`generators/requirements.txt`). They carry the absolute data paths of the machine they were run on and are not called by the pipeline. Input and output are `(n, 100, 2)` lon / lat arrays, the layout that `ais_prep.py prepare` writes and the environments read.

| Script | Model |
|---|---|
| `train_vae_full.py` | TSGM `vae_conv5` BetaVAE with a reconstruction, KL and temporal loss; writes generated samples |
| `multi_training_test.py` | the same VAE over five epoch / beta configurations, with the TSGM feature-wise scaler |
| `train_cGAN_full.ipynb` | TSGM BetaVAE with a custom convolutional encoder and decoder (despite the name, no GAN) |
| `train_Unet_full.py` | DDPM-style diffusion with a 1-D convolutional U-Net and a time embedding; writes generated samples |
| `train_DiffTraj_full.ipynb` | the DiffTraj guided U-Net; needs the DiffTraj `utils/config_WD.py` configuration, which is not in the repository, and its data cell is a random placeholder |
| `plot_training.ipynb` | plots and animates generated samples |

The JSD notebook that used to sit next to them was removed on 2026-09-24: it compared each trace with a distribution fitted to that same trace and could not rank generators (`../review/jsd_probe.py`).

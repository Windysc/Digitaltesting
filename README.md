# Digitaltesting

Digital testing of maritime autonomous surface ships (MASS). A reinforcement-learning adversary generates safety-critical encounter scenarios for stress-testing simplified reactive collision-avoidance controllers, on target routes shaped by AIS trace data. The pipeline has two stages, extraction and generation, and all of its code is in the folder `general/`; `review/` holds the records of the external review that fixed the scope.

## Research scope

**Extraction.** AIS tracks (CSV reports with timestamps) become trace arrays with a valid time base. `general/ais_prep.py` cleans the reports, splits the voyages into train / val / test before any estimate, removes bad fixes by a distance rule, splits at reception gaps, smooths on the real timestamps (Kalman / RTS, strength chosen inside physical bounds), resamples on an absolute 20 s grid and writes windows of 100 points in the `(n, 100, 2)` lon / lat layout, together with speed and course per point and an aggregate report that can be shared while the data stay private. The same module turns a window into a scenario route: a C2 spline tabulated by arc length with heading and curvature, checked against the target ship's turn rate at the scale of the collision standard, so that what is checked is what is sailed. Synthetic fleets with a known truth (`synth_ais.py`) and twelve checks (`check_mending_plan.py`) establish the chain. The trace generators in `general/generators/` (TSGM VAE and diffusion models) trained on the windows produce further traces in the same layout.

**Generation.** The encounter environment (`env_moving_obj.py`, `env_moving_attack.py`, `scenario_targets.py`, `attack_scenarios.py`) places a target ship on a fixed track, a straight chord or a data-shaped route, under the collision standard of `encounter_standard.py`: CPA warning, ship domain and collision as severity levels, the collision-risk index and the COLREG encounter type, with the encounter lifecycle approach, action, passing and clear. Eight attack scenarios (head-on on either side, crossing from either side ahead or astern, parallel lanes) with an exact initial DCPA band form the catalogue. A PPO attacker (`PPO.py`, `main_attack_ppo_enc.py`) with full control of the six own-ship parameters inside set limits has to create the event; the scripted baselines hold course, intercept and pursuit are the sanity references. `run_matrix.py` runs seed and scenario matrices, `check_attack_scenarios.py` checks the geometry, `viz_tool.py` and `chart_viz.py` replay checkpoints and write reports. `main_attack_ppo_scen.py` is the earlier three-family variant in a hand-scaled 2 km arena, kept because it takes the same route chain and is covered by the same wiring check.

The study design agreed after the external review (systems under test, methods compared, outcomes and statistics) is `review/STUDY_PLAN_PEER_REVIEW.md`; the review record with the open items is `review/RESEARCH_REVIEW.md`.

## Layout

| Path | Content |
|---|---|
| `general/` | The pipeline: extraction, generation and their checks. `general/README.md` says how to run each part; `general/MASS_TESTING_ENV_REBUILD.md` is the design and verification log of the environments. |
| `general/generators/` | The trace generators and a plotting notebook (listed in `general/README.md`). |
| `general/check_out/` | Results of the checks of the mending plan and of the environment wiring on the synthetic fleets. |
| `review/` | Records: the external review (Codex, September 2026), the study plan, the mending plan of the data step and two probes. Paths inside these records refer to the layout at the time (`PPO_scenario_generate/` with `data_prep/`), which is now `general/`. |
| `.aris/` | The traces of the review rounds. |

## Data

The AIS data are private and not included. Everything in the repository runs on synthetic traces: `python check_mending_plan.py fleet` writes the synthetic fleets and `ais_prep.py prepare` turns them into windows. On the private data:

```
cd general
python ais_prep.py legacy-check --input <folder with the CSV files>
python ais_prep.py prepare --input <folder> --out <out folder> --pairs
python main_attack_ppo_enc.py --trace <out folder>/windows_lonlat_train.npy --scenario mix --dcpa_band passing --control full
```

`review/MENDING_PLAN_PART1.md` explains the checks and how to read the report. Arrays made with the old row-index resampling, where one array point stood for 13 to 218 s of real time, must not be mixed with the new ones.

## Requirements

One Python (3.10 or newer; 3.14 was used) with the packages in `requirements.txt`: numpy, pandas, matplotlib, pillow, tqdm, gymnasium and torch; scipy only for the legacy comparison. The generators were written for Python 3.10 with the TSGM stack listed in `general/generators/requirements.txt`. Run the scripts from a work directory, because logs and `runs/` land in the current directory; the CPU is faster than a GPU for the 64-unit policy network.

## Removed on 2026-09-24

The repository was reduced to the methods that the extraction and the generation use. The parts below are no longer in the tree; all of them are in the history up to commit 3b9feb0.

- `Ship_envre`: the first environment (Stable-Baselines3 PPO on the ShipAI hull model, Python 3.7). Its `viewer.py` was never committed, so it could not be imported, and nothing in the pipeline used it.
- `Ship_envre_v2` and `Ship_envre_v3`: a separate build with continuous ship dynamics. Its collision standard, grading and encounter lifecycle live on in `encounter_standard.py`; its own route path was never wired to the mended data step, and the study plan dropped the transfer to it.
- The first data step, `Train_VAE_full/data_csv2npy.py` and `Ship_envre/interpolation.py` (row-index resampling). The original resampling function is kept in `general/legacy_stage.py` for the comparison in the checks.
- The generator attempts that the old README marked as not usable: `train_diffusion_full.py`, `Others/train_GAN_custom_full.ipynb` and `Train_wGAN/train_wavegan_full.ipynb`.
- `Evaluation/scenario_creation.ipynb`, the random pairing of two traces that the scenario builder replaces, and `Evaluation/jsd_matrix.ipynb`, whose metric compares each trace with a distribution fitted to that same trace and cannot rank generators (`review/jsd_probe.py`).
- The two 2024 obstacle-scene reference trainers `main_ppo.py` and `main_attack_ppo.py`; the attack trainers carry their loop.

# Digitaltesting

Code and documentation for the digital testing of MASS (maritime autonomous surface ships), built on the ShipAI basis: generated AIS traces feed scenario environments in which a reinforcement-learning agent creates intentional safety-critical encounters.

The workflow has two parts.

1. **Trace following.** The adversarial ship enters at the start of a route and moves along it according to generated and recorded route data.
2. **Free chase.** Once both ships reach the selected area the agent is free to act and the catch-up race begins.

## Folders

| Folder | Content | Status |
|---|---|---|
| `Train_VAE_full`, `Train_nGAN_full`, `Train_Diff_Full` | Training scripts of the trace generators (VAE, cGAN / wGAN, diffusion, TrajDiffusion, U-Net) on TSGM built-in and self-built models. Scripts marked `(u)` are the usable ones. | original |
| `Evaluation` | Notebooks that turn data into importable traces, judge the generation with the JSD matrix, plot training and create scenarios. | original |
| `Ship_envre` | The first environment: `ppo_sb3_rl.py` (Stable-Baselines3 PPO, train and eval modes), `Ship_env.py` (rebuilt ShipAI environment with border reading, scope calculation and reward stages), `simulator.py` (ship simulation). | original, kept as is |
| `Ship_envre_v2` | Regenerated attack-scenario pipeline (2026-09-14): Fossen-form ship dynamics with yaw damping, danger criteria (DCPA / TCPA, ship domain, CRI, COLREG encounter type), a ship-like target with automation levels, COLREG scenario families, event grading and scoring, scorecards, self-play and chart-style animations. `README_v2.md`. | remade |
| `Ship_envre_v3` | The encounter-lifecycle definition (2026-09-14): an episode is one encounter with the phases approach, action, passing and clear; no destination point and no time limit. Steady course-holding target, live viewer and episode recorder. Imports v2 unchanged. `README_v3.md`. | remade |
| `PPO_scenario_generate` | Rebuilt modules for the reference scripts `main_ppo.py` / `main_attack_ppo.py` (world model, PPO interface, gym shim), the collision standard, eight attack scenarios under the v3 definition, full attacker control with limits, scripted baselines, training matrices and the visualisation tool. `MASS_TESTING_ENV_REBUILD.md`. | remade |
| `PPO_scenario_generate/data_prep` | Timestamp-correct sampling and smoothing of AIS tracks (`ais_prep.py`), the synthetic known-truth fleets and the checks of the mending plan. `README.md`. | remade |
| `PPO_scenario_generate/review` | The external review of the pipeline (Codex, September 2026), the study-plan review and the checked mending plan for the data step. | records |

## Data

The AIS data are not included; they are private. Everything in the repository runs on synthetic traces (`Ship_envre_v2/make_synthetic_data.py`, `PPO_scenario_generate/data_prep/synth_ais.py`).

The original data step (`Train_VAE_full/data_csv2npy.py` and `Ship_envre/interpolation.py`) resampled each CSV by row index, so one array point stood for 13 to 218 s of real time inside one trace. It is replaced by the mended chain:

```
cd PPO_scenario_generate/data_prep
python ais_prep.py legacy-check --input <folder with the CSV files>
python ais_prep.py prepare --input <folder> --out <out folder> --pairs
```

`prepare` cleans the reports, splits the voyages into train / val / test first, removes bad fixes by a distance rule, splits at reception gaps, smooths on the real timestamps (Kalman / RTS, strength chosen inside physical bounds), resamples on an absolute 20 s grid and writes `windows_lonlat*.npy` in the old `(n, 100, 2)` format, so every existing consumer loads it, together with speed and course per point and an aggregate report that can be shared. `review/MENDING_PLAN_PART1.md` explains the checks and how to read the report. Arrays made with the old step should not be mixed with the new ones.

## Running the remade parts

All remade parts run on one Python (3.14 was used) with numpy, pandas, matplotlib, torch, gymnasium and tqdm; scipy is needed only to rerun the legacy comparison. Run each script from a work directory, because logs land in the current directory.

**Ship_envre_v2 and v3.** `python main_attack_ppo_ship.py --task attack` (v2) and `python main_attack_ppo_v3.py` (v3) train; `evaluate_agent*.py` writes scorecards; `animate_*.py` renders episodes. With arrays from `data_prep` pass `--trace_dt 20 --smooth_sigma 0`.

**PPO_scenario_generate.** `main_attack_ppo_enc.py` trains the attacker on the eight encounter scenarios (`--scenario mix --dcpa_band passing --control full`), `check_attack_scenarios.py` runs the geometry self-test and the scripted baselines, `run_matrix.py` runs seed and scenario matrices, `viz_tool.py` replays checkpoints and writes reports. `--trace <windows_lonlat.npy>` shapes the target routes with generated or recorded traces through the mended chain: every trace is checked against the target's turn rate at the standard's scale before it is used, and the target plays the checked curve back.

**Checks.** `data_prep/check_mending_plan.py tests` (about 10 min) reproduces every table of the mending plan on the synthetic fleets; `data_prep/check_env_wiring.py` checks the environments against the collision standard.

## Requirements

The original generation scripts were written for Python 3.10 with the default TSGM requirements; the original `Ship_envre` for Python 3.7 with the packages in `requirements.txt`. The remade parts have no dependency on either.


# Review brief: data extraction, trace evaluation, and how the attack-scenario benchmark is filled

Date 2026-09-18. Executor: Claude Opus 5. Reviewer: please verify everything against the files; my notes are not evidence.

## The project in one paragraph

Digitaltesting (owner's project) builds a pipeline for testing maritime autonomous surface ships (MASS). The aim is to produce adversarial "attack" scenarios: an RL attacker (PPO) is trained to force a safety-critical encounter (CPA warning, ship-domain violation, collision) with a target ship.

The pipeline has three stages:

1. **Data extraction.** An AIS CSV track is turned into a fixed-length trace array.
2. **Generation and evaluation.** Generative models (VAE, "cGAN", which is really a BetaVAE, U-Net diffusion, and DiffTraj) are trained on those arrays to produce new traces. The traces are judged with a JSD metric.
3. **Benchmark.** The traces are paired into encounter scenarios, and the RL attacker is trained and scored on them.

The owner's question is: **review the first-step data extraction and its evaluation method, and think whether there are better methods to fill in the benchmark.**

## Files to inspect

Original project (read-only): `C:\Users\ASUS\Desktop\Digitaltesting-main\Digitaltesting-main\`

| Area | Files |
|---|---|
| Stage 1 (extraction) | `Train_VAE_full/data_csv2npy.py` |
| Stage 2 (generators) | `Train_VAE_full/train_vae_full.py`, `Train_nGAN_full/Train_cGAN/train_cGAN_full(u).ipynb`, `Train_Diff_Full/train_Unet_full(u).py`, `Train_Diff_Full/train_DiffTraj_full(u).ipynb` |
| Evaluation | `Evaluation/jsd_matrix.ipynb` (the "judge"), `Evaluation/plot_training.ipynb`, `Evaluation/scenario_creation.ipynb` (pairs two traces) |
| Original env | `Ship_envre/Ship_env.py` (loads `dataset_1.csv.npy`, `dataset_2.csv.npy` as guideline and merge line), `Ship_envre/interpolation.py` |
| Earlier audit | `Ship_envre_v2/SCENARIO_SETTINGS_CHECK.md` (section A). This is an earlier audit I wrote; verify it, don't trust it. |

The real AIS CSVs and `.npy` files are **not on this machine**, because they are sensitive. Only the code can be checked. The CSV path in `data_csv2npy.py` is `/home/junze/.jupyter/head-on/1/2.csv`.

The current benchmark lives in `C:\Users\ASUS\Desktop\PPO_scenario_generate\`:

| Role | Files |
|---|---|
| Collision standard (v2 severity, CRI, grading, v3 lifecycle) | `encounter_standard.py` |
| 8 parametric attack scenarios × DCPA bands, env, scripted baselines | `attack_scenarios.py` |
| How generated traces enter (`load_trace_shapes`, `rotate_translate`) | `scenario_targets.py` |
| Baseline tables | `runs/enc_check/baseline_summary.md`, `runs/enc_check_full/baseline_summary.md` |
| Notes | `C:\Users\ASUS\Desktop\MASS_TESTING_ENV_REBUILD.md` (sections on the collision standard, scenarios, full control) |
| Metric probe | `review/jsd_probe.py` |

## What I believe (verify, then refute or confirm)

### Stage 1: data extraction (`data_csv2npy.py`)

- **E1.** One CSV path is hard-coded, so one vessel track becomes the whole dataset. Only `longitude_degrees` and `latitude_degrees` are kept. Speed, time, COG, heading, MMSI and ship type are dropped.
- **E2.** The track is resampled to 100 points by a cubic spline over the **row index**, not the timestamp. The time per point is therefore unknown downstream. There is no cleaning: no MMSI or voyage segmentation, no gap or outlier removal, no handling of duplicate timestamps.
- **E3.** Augmentation uses `tsgm GaussianNoise(variance=1e-4)` on raw degrees, giving sigma 0.01°, about 1.1 km of i.i.d. jitter per point, to make 500 copies. So the generative models are trained on one trace plus white noise. There is no train/test split and no held-out real data.
- **E4.** `np.save('trajectory_data.npy', lat_lon+speed)` broadcasts the speed onto both coordinates. The output is unused.
- **E5.** The unit of a sample is one ship's track. A scenario, though, is a *pair* of ships in an encounter. Pairing happens later by random indexing (`scenario_creation.ipynb` draws a random index out of 200 from each of two files, with empty paths). The folder name `head-on/1/2.csv` suggests encounters were hand-labelled somewhere, but the extraction does not keep the partner ship or the encounter geometry.

### Evaluation (`jsd_matrix.ipynb`)

- **J1.** The metric does **not** compare generated data with real data. Each trajectory's segment lengths are compared with a Normal distribution fitted by MLE to those same segment lengths.
- **J2.** JSD-TD is computed on a single number (the total distance), with 3 bins over ±1e-10. It is degenerate.
- **J3.** The JSD is computed on CDFs plugged into a KL formula, not on probability masses.
- **J4.** The loop over "20 trajectories" loads the same file 20 times.
- **J5.** The metric only sees step lengths, so it ignores:
  - shape, heading and turn rate;
  - position, time and speed;
  - diversity, novelty and memorisation;
  - encounter properties.

**Probe result.** `review/jsd_probe.py` executes the notebook's own functions on controlled traces:

| Case | JSD-TD | JSD-SD |
|---|---|---|
| Clean real-like trace | 0.4472 | 0.0673 |
| + 11 m noise | 0.4472 | 0.0267 |
| + 1.1 km noise (the data_csv2npy setting) | 0.2765 | 0.0552 |
| Pure random walk (not a ship) | 0.5437 | 0.0345 |
| Reversed trace | 0.4472 | 0.0673 |
| Trace moved 1° away | 0.4472 | 0.0673 |
| Uniform random points in a box | 0.4472 | 0.0519 |

My reading: JSD-TD is constant at sqrt(0.2) except when the MLE drifts, and JSD-SD ranks a random walk and random points as *better* than the clean trace. The metric cannot rank generators.

### Benchmark: how the generated traces are used now (PPO_scenario_generate)

- **B1.** Generated traces enter only as optional route *shapes*: `--trace` → `load_trace_shapes`, smoothed, rotated onto a parametric scenario course. Speed and timing come from the parametric generator, which uses equal constant speeds and a fixed-track target. The encounter geometry (8 families × DCPA bands, exact initial DCPA) is fully synthetic. So the data currently barely matters to scenario difficulty.
- **B2.** The benchmark saturates. A scripted intercept succeeds 1.0 in every head-on and crossing cell. The learned attacker gets 1152/1152 on the scorecard under full control. Holding course succeeds 83–100% in the v2 DCPA band. Only the parallel lanes discriminate between the scripted rules.
- **B3.** There is no measure of coverage, realism, or criticality distribution over the scenario space. The success rate is the only benchmark output.

## Questions for the reviewer

1. Are E1–E5 and J1–J5 correct? Is anything I call a defect actually fine, or did I miss one?
2. **Better extraction.** What is the right first step for AIS-based encounter scenarios?
   - Segmentation, time-based resampling, and cleaning thresholds.
   - Encounter mining from pairs (CPA filters, COLREG classification).
   - What to record per encounter.
   - Public AIS sources usable when the owner's data are sensitive.
3. **Better evaluation of generated traces.** Which metrics are actually valid? For example:
   - Kinematic plausibility: speed, acceleration and turn-rate distributions.
   - Distribution distances between real and generated data (MMD, Wasserstein on features, Fréchet/DTW-based).
   - Discriminative and predictive scores (TSTR).
   - Memorisation and novelty checks.
   - Encounter-level parameter distributions (DCPA, TCPA, crossing angle, speed ratio).

   Which subset is the minimum honest set?
4. **Filling the benchmark.** Given B1–B3, is a deep generative trace model even the right tool? Alternatives to judge:
   - (a) Sample encounter *parameters* from empirical AIS encounter distributions (KDE or copula), then instantiate them parametrically.
   - (b) Replay mined real encounters directly.
   - (c) Importance sampling or adaptive stress testing toward rare critical regions.
   - (d) Coverage-driven combinatorial or search-based scenario selection.
   - (e) A generative model conditioned on encounter parameters.

   Which gives a defensible benchmark, and how should coverage and difficulty be measured so it stops saturating?
5. **Minimum package.** What is the smallest concrete work package, in priority order, that turns this into a defensible pipeline? A laptop CPU is enough for the RL; GPU is available for generators.

Scope limits: this is a research-workflow review. Do not propose hashing or fingerprinting, speculative infrastructure, or corner cases that do not arise here. Say plainly when something is correct.

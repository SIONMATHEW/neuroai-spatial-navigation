# NeuroAI Spatial Navigation Experiments

This repository contains the code and reviewed result artifacts for a UROP
study of how training objectives and environmental structure affect recurrent
neural-network representations during spatial navigation.

The repository preserves the actual experimental protocols used for the
report. It does not combine results from incompatible training budgets or
silently replace historical analyses with a different method.

## Experiments

### 1. Continuous-coordinate baseline

The exploratory baseline predicts continuous position and gaze-intersection
coordinates. It motivated the later use of categorical place and gaze labels.
The original controlled baseline is retained in `continuous_baseline/`.

### 2. Factorized and conjunction objectives

The primary objective comparison uses clean inputs, a 256-unit tanh RNN,
8,000 updates, batches of 256 trajectories and five paired seeds. Within each
seed, both objectives receive the same training batches and start from the
same recurrent weights. Validation data select checkpoints; the 4,096-trial
test set is used only after selection.

Two comparisons are reported:

1. the best validation-selected checkpoint within the common training budget;
2. checkpoints matched near 83% validation exact-joint accuracy.

The matched comparison contains seeds 2-5 because seed 1 did not satisfy the
predefined validation tolerances. Test accuracy was not forced to match.

### 3. Frozen hidden-state analysis

The hidden-state script loads the saved objective checkpoints without
retraining. It evaluates post-tanh activity on new probe trajectories using
trajectory-grouped nested cross-validation. The primary measure is the held-out
gain from adding a place-by-gaze interaction to an additive place-and-gaze
model. Unit categories are secondary descriptive summaries. Network seed, not
hidden unit or timestep, is the inferential replicate.

### 4. Open and occluded environments

The environmental comparison uses the factorized objective only. Five paired
seeds are trained for 4,000 updates in either an open arena or an arena with
four textured pillars. Sensory noise is 0.20 during training and 0.25 during
test and probe evaluation. The environments have different label-space sizes,
so raw exact-joint accuracy is contextual rather than a class-matched causal
measure of occlusion.

## Repository layout

```text
continuous_baseline/       Historical continuous-coordinate comparison
objective_comparison/      Clean 8,000-update objective experiment
environment_comparison/    Noisy open-versus-occluded experiment
configs/                   Machine-readable experimental specifications
results/                   Reviewed CSV, JSON and figure artifacts
tests/                     Fast structural and numerical tests
docs/                      Methods, provenance and reproduction notes
figures/                   Script and outputs for compact report figures
```

## Installation

Python 3.12 and Julia 1.11 are recommended. PyTorch must be installed for the
target platform before installing the remaining dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On the NUS cluster, the reported runs used
`$HOME/kernel-env-h100/bin/python`, PyTorch 2.11.0+cu128 and H100 GPUs. See
`docs/REPRODUCIBILITY.md` for the exact commands and important limitations.

## Fast checks

```bash
python -m compileall objective_comparison environment_comparison continuous_baseline
pytest
```

Each full experiment also has a `--quick` mode or a documented smoke command.
Quick runs validate execution only; their numerical values are not scientific
results.

Regenerate the compact report figures without retraining:

```bash
python figures/make_summary_figures.py
```

## Reproducing the reported objective comparison

From `objective_comparison/`:

```bash
bash submit_tuned_comparison.sh
```

After the five training jobs and dependent aggregation job finish, run the
frozen analysis with:

```bash
sbatch run_frozen_hidden_h100.slurm
```

## Reproducing the environmental comparison

From `environment_comparison/`:

```bash
bash submit_environment_comparison.sh
```

## Results and checkpoints

The repository versions compact tables and figures so every report figure can
be regenerated without retraining. Model checkpoints and raw per-timestep
arrays are intentionally excluded from ordinary Git history; they should be
distributed as a tagged release asset with SHA-256 checksums.

Numerical summaries and interpretation limits are recorded in
`docs/RESULTS.md`; ready-to-use captions are in `docs/FIGURE_CAPTIONS.md`.

## Dependency acknowledgement

Trial generation uses the Python and Julia `RNNTrialStructures` packages
developed by Roger Herikstad and extended for reproducible navigation presets.
The final environment workflow pins the following revisions:

- Python wrapper: `a9d4b3b6f38dec6abf5f67f4a5cc6e4c0faf2b65`
- Julia package: `0fb8c74710d8e129b3f6918daeb2bf16759282c2`

See `DEPENDENCIES.md` for the repository links and historical-version caveat.

The clean objective checkpoints were trained with an earlier installed Julia
package snapshot identified in the cluster logs as `0Vqa9`. The corresponding
tree identifier must be archived before claiming bitwise retraining
reproduction; the saved checkpoints and result tables remain the source for
exact figure reproduction.

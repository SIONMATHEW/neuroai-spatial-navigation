# Objective comparison

This directory contains the clean, paired comparison between factorized and
conjunction training objectives. Both networks have the same recurrent
architecture, receive the same batch of trajectories at each update and start
from the same recurrent initialization within a seed. Only the output heads
and supervised loss differ.

The full run uses five seeds, 8,000 updates per model and clean sensory input.
Validation data select checkpoints. Test data are evaluated after checkpoint
selection and never tune training settings.

Run the complete comparison on Slurm with:

```bash
bash submit_tuned_comparison.sh
```

After aggregation completes, run the frozen hidden-state analysis with:

```bash
sbatch run_frozen_hidden_h100.slurm
```

The machine-readable protocol is in `../configs/objective_comparison.json`.
The compact reviewed outputs are in `../results/objective_comparison/`.

## Per-bin accuracy and SIC

`evaluate_sic_bins.py` reuses the saved held-out predictions from the
accuracy-matched comparison to report accuracy separately for all 21 place
bins and all 24 gaze-surface bins. It then runs the frozen checkpoints on a
new shared probe bank and computes Skaggs information for each post-tanh
hidden unit.

Significance is defined against the 99th percentile of pooled SIC values from
20 untrained recurrent networks evaluated on the same inputs. The resulting
place-only, surface-only and dual-selective labels describe information about
the two variables; they do not imply a nonlinear interaction. Run with:

```bash
sbatch run_sic_bins_h100.slurm
```

Set `OBJECTIVE_RESULTS_ROOT` if the frozen result directory is stored at a
different location.

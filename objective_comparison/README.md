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

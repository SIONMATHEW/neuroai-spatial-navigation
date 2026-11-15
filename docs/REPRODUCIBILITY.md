# Reproducibility guide

## Reproduction levels

The repository distinguishes two claims:

1. **Figure reproduction** uses the reviewed CSV and JSON files under
   `results/` and should regenerate the reported summaries exactly.
2. **Retraining reproduction** reruns stochastic GPU optimization from the
   recorded configuration. It should reproduce the qualitative and seed-level
   effects, but bitwise equality is not promised across CUDA, PyTorch or GPU
   versions.

## Data separation

Training, validation and test generators use disjoint RNG seed ranges. The
clean objective comparison materializes one validation set per seed before
training and generates the final test trials only after checkpoint selection.
The hidden analyses generate separate probe trajectories that never choose a
checkpoint.

Online random generation does not by itself prove that no two trajectories
share a partial path. The overlap audit records exact and approximate matches
where historical regeneration is possible. This limitation is documented
rather than treated as a guarantee.

## Cluster commands

Set the project path if the repository is not located at the default:

```bash
export NEUROAI_PROJECT="$HOME/research/neuroai-spatial-navigation"
```

Objective comparison:

```bash
cd "$NEUROAI_PROJECT/objective_comparison"
bash submit_tuned_comparison.sh
```

Frozen hidden-state analysis after checkpoints exist:

```bash
cd "$NEUROAI_PROJECT/objective_comparison"
sbatch run_frozen_hidden_h100.slurm
```

Environmental comparison:

```bash
cd "$NEUROAI_PROJECT/environment_comparison"
bash submit_environment_comparison.sh
```

## Required archival metadata

Before the public release, add:

- the commit hash of this repository;
- checksums for released checkpoints;
- the exact historical Julia tree used for the clean objective training;
- the completed frozen hidden-analysis tables;
- the cluster Python, Julia, PyTorch and CUDA versions recorded by each run.

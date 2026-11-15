#!/usr/bin/env bash
# Submit the five paired-seed training jobs and the dependent evaluation job.
set -euo pipefail

PROJECT=${NEUROAI_PROJECT:-"$HOME/research/neuroai-spatial-navigation"}
cd "$PROJECT/environment_comparison"

TRAIN_JOB=$(sbatch --parsable run_environment_training_h100.slurm)
TRAIN_JOB=${TRAIN_JOB%%;*}
echo "Environmental training array: $TRAIN_JOB"

EVAL_JOB=$(sbatch --parsable \
  --dependency="afterok:$TRAIN_JOB" \
  run_environment_evaluation_h100.slurm)
EVAL_JOB=${EVAL_JOB%%;*}
echo "Dependent evaluation job: $EVAL_JOB"

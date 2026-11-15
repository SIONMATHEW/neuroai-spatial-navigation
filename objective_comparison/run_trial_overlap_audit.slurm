#!/bin/bash
#SBATCH --job-name=rnn-overlap
#SBATCH --partition=long
#SBATCH --array=1-5%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-overlap-%A_%a.log

set -euo pipefail
PROJECT=${NEUROAI_PROJECT:-"$HOME/research/neuroai-spatial-navigation"}
cd "$PROJECT/objective_comparison"
export PYTHONUNBUFFERED=1
export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
# CPU trial regeneration only: no model forward/backward or GPU allocation.
srun "$HOME/kernel-env-h100/bin/python" audit_trial_overlap.py \
  --results-dir objective_results \
  --output-dir trial_overlap_audit \
  --seeds "$SLURM_ARRAY_TASK_ID"

#!/usr/bin/env bash
#SBATCH --job-name=env-eval
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-environment-eval-%j.log

set -euo pipefail

PROJECT=${NEUROAI_PROJECT:-"$HOME/research/neuroai-spatial-navigation"}
cd "$PROJECT/environment_comparison"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

"$HOME/kernel-env-h100/bin/python" evaluate_environments.py \
  --seeds 1 2 3 4 5 \
  --checkpoint final \
  --device cuda \
  --results-root environment_results \
  --output-dir environment_evaluation

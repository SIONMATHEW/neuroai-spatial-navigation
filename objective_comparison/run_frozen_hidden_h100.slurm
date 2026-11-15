#!/usr/bin/env bash
#SBATCH --job-name=frozen-hidden
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-frozen-hidden-%j.log

set -euo pipefail

PROJECT=${NEUROAI_PROJECT:-"$HOME/research/neuroai-spatial-navigation"}
cd "$PROJECT/objective_comparison"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$HOME/kernel-env-h100/bin/python" evaluate_frozen_hidden.py \
  --results-root objective_results \
  --output-dir frozen_hidden_analysis \
  --analysis both \
  --probe-trajectories 1024 \
  --device cuda

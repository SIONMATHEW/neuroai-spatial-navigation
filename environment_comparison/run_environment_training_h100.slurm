#!/usr/bin/env bash
#SBATCH --job-name=env-train
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=1-5%2
#SBATCH --output=slurm-environment-train-%A_%a.log

set -euo pipefail

PROJECT=${NEUROAI_PROJECT:-"$HOME/research/neuroai-spatial-navigation"}
cd "$PROJECT/environment_comparison"
export PYTHONUNBUFFERED=1

echo "seed=$SLURM_ARRAY_TASK_ID node=$(hostname)"
"$HOME/kernel-env-h100/bin/python" train_environments.py \
  --seeds "$SLURM_ARRAY_TASK_ID" \
  --device cuda \
  --output-root environment_results

#!/bin/bash
#SBATCH --job-name=rnn-tuned-compare
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=1-5%1
#SBATCH --output=slurm-tuned-comparison-%A_%a.log

set -euo pipefail
PROJECT=${NEUROAI_PROJECT:-"$HOME/research/neuroai-spatial-navigation"}
cd "$PROJECT/objective_comparison"
export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID; paired training seed: $SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun "$HOME/kernel-env-h100/bin/python" compare_tuned_models.py \
  --seeds "$SLURM_ARRAY_TASK_ID" --train-only --device cuda \
  --also-accuracy-matched --output-dir objective_results

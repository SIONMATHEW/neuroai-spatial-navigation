#!/bin/bash
#SBATCH --job-name=rnn-factorized-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-factorized-sweep-%j.log

set -euo pipefail

PROJECT=${NEUROAI_PROJECT:-"$HOME/research/neuroai-spatial-navigation"}
cd "$PROJECT/objective_comparison"

PYTHON="$HOME/kernel-env-h100/bin/python"
export PYTHONUNBUFFERED=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PYTHON" -c "import torch; assert torch.cuda.is_available(); print('PyTorch:', torch.__version__); print('CUDA GPU:', torch.cuda.get_device_name(0))"

# Coarse search: one training seed, fixed validation data, 12 configurations.
"$PYTHON" sweep_factorized.py \
  --learning-rates 0.001 0.0005 \
  --batch-sizes 64 128 256 \
  --gradient-clips 1.0 3.0 \
  --schedulers none \
  --max-updates 8000 \
  --eval-every 250 \
  --val-trials 1024 \
  --seeds 1 \
  --target-accuracy 0.83 \
  --device cuda \
  --output-dir factorized_sweep

echo "Factorized-discrete coarse sweep completed."

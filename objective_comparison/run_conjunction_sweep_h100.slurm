#!/bin/bash
#SBATCH --job-name=rnn-conjunction-sweep
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-9%1
#SBATCH --output=slurm-conjunction-sweep-%A_%a.log

set -euo pipefail

PROJECT=${NEUROAI_PROJECT:-"$HOME/research/neuroai-spatial-navigation"}
cd "$PROJECT/objective_comparison"
RNN_PYTHON="$HOME/kernel-env-h100/bin/python"
export PYTHONUNBUFFERED=1

echo "Job ID: $SLURM_JOB_ID; array configuration: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$RNN_PYTHON" -c "import torch; assert torch.cuda.is_available(); print('PyTorch:', torch.__version__); print('CUDA GPU:', torch.cuda.get_device_name(0))"

# Indices 0..9 match the ten fully completed factorized configurations:
# lr=0.001, batches 64/128/256, clips 1/3 (six configurations), then
# lr=0.0005, batches 64/128, clips 1/3 (four configurations).
# For all twelve grid configurations use: sbatch --array=0-11%1 this_script
# Keep concurrency at %1: the sweep shares ranking/manifest files.
# Each configuration gets its own 8-hour limit, rather than sharing one limit.
srun "$RNN_PYTHON" sweep_conjunction.py \
  --config-index "$SLURM_ARRAY_TASK_ID" \
  --learning-rates 0.001 0.0005 \
  --batch-sizes 64 128 256 \
  --gradient-clips 1.0 3.0 \
  --schedulers none \
  --max-updates 8000 \
  --patience-evals 0 \
  --eval-every 250 \
  --val-trials 1024 \
  --eval-batch-size 256 \
  --seeds 1 \
  --target-accuracy 0.83 \
  --device cuda \
  --output-dir conjunction_sweep

echo "Conjunction configuration $SLURM_ARRAY_TASK_ID completed."

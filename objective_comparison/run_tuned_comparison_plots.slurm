#!/bin/bash
#SBATCH --job-name=rnn-tuned-plots
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-tuned-plots-%j.log

set -euo pipefail
PROJECT=${NEUROAI_PROJECT:-"$HOME/research/neuroai-spatial-navigation"}
cd "$PROJECT/objective_comparison"
export PYTHONUNBUFFERED=1
srun "$HOME/kernel-env-h100/bin/python" compare_tuned_models.py \
  --aggregate-only --device cpu --seeds 1 2 3 4 5 \
  --also-accuracy-matched --output-dir objective_results

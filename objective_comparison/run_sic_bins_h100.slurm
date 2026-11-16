#!/usr/bin/env bash
#SBATCH --job-name=obj-sic
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-objective-sic-%j.log

set -euo pipefail

PROJECT="${NEUROAI_PROJECT:-$HOME/research/neuroai-spatial-navigation}"
RESULTS_ROOT="${OBJECTIVE_RESULTS_ROOT:-$PROJECT/objective_comparison/tuned_dual_comparison_results}"

cd "$PROJECT/objective_comparison"
"$HOME/kernel-env-h100/bin/python" evaluate_sic_bins.py \
  --results-root "$RESULTS_ROOT" \
  --output-dir objective_sic_bin_analysis \
  --device cuda


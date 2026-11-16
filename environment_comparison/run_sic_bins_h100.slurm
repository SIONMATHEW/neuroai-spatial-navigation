#!/usr/bin/env bash
#SBATCH --job-name=env-sic
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-environment-sic-%j.log

set -euo pipefail

PROJECT="${NEUROAI_PROJECT:-$HOME/research/neuroai-spatial-navigation}"
RESULTS_ROOT="${ENVIRONMENT_RESULTS_ROOT:-$PROJECT/environment_comparison/environment_results_full}"

cd "$PROJECT/environment_comparison"
"$HOME/kernel-env-h100/bin/python" evaluate_sic_bins.py \
  --results-root "$RESULTS_ROOT" \
  --output-dir environment_sic_bin_analysis \
  --checkpoint final \
  --device cuda


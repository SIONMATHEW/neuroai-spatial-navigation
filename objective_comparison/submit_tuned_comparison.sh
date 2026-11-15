#!/bin/bash
# Submit only after the GPU smoke test passes. The CPU plot job waits for every
# array element to succeed. Running this script submits two jobs to Slurm.
set -euo pipefail
PROJECT=${NEUROAI_PROJECT:-"$HOME/research/neuroai-spatial-navigation"}
cd "$PROJECT/objective_comparison"
RNN_ARRAY_ID=$(sbatch --parsable run_tuned_comparison_h100.slurm)
RNN_ARRAY_ID=${RNN_ARRAY_ID%%;*}
echo "Paired comparison array: $RNN_ARRAY_ID"
RNN_PLOT_ID=$(sbatch --parsable --dependency="afterok:$RNN_ARRAY_ID" run_tuned_comparison_plots.slurm)
echo "Dependent CPU plot job: $RNN_PLOT_ID"

# Source provenance

The repository was assembled from the experiment scripts and result artifacts
used during the UROP project. Cleaning is intentionally conservative: command
paths and documentation are normalized, while scientific defaults are kept
unchanged.

## Objective experiment

The source files in `objective_comparison/` were copied from the controlled
objective-comparison working directory used for the clean 8,000-update runs.
The authoritative run configuration is
`configs/objective_comparison.json`. Reviewed summaries and plots were imported
from `tuned_dual_comparison_results` and its `accuracy_matched` subdirectory.

## Environmental experiment

`environment_comparison/train_environments.py` and
`environment_comparison/evaluate_environments.py` are the scripts used for the
five-seed 4,000-update open-versus-occluded run. The corresponding reviewed
evaluation artifacts are stored under `results/environment_comparison/`.

## Continuous baseline

`continuous_baseline/continuous_vs_conjunction.py` preserves the historical
controlled continuous-coordinate baseline. It is retained to reproduce the
developmental comparison, not as the primary objective experiment.

## Excluded work

The noisy 4,000-update factorized-versus-conjunction experiment, exploratory
SIC analyses, quick runs and obsolete hidden-state pipelines are not part of
the report's core reproducible workflow. They remain in the private working
archive rather than being presented as confirmatory evidence.

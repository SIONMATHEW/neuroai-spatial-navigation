# Versioned results

This directory contains compact, reviewed outputs used to build the report.
CSV and JSON files are the numerical source of truth; PNG files are convenient
renderings. Checkpoints, raw timestep arrays and generated training batches are
excluded from Git because of size.

- `objective_comparison/best_within_budget/`: five validation-best seed pairs.
- `objective_comparison/accuracy_matched/`: four matched seed pairs (2-5).
- `environment_comparison/`: five paired open and occluded networks.
- `continuous_baseline/`: historical continuous-coordinate comparison.

The frozen objective hidden-analysis outputs will be added after the full
analysis job completes and passes consistency checks.

`SHA256SUMS.txt` records the content hash of every versioned result artifact.

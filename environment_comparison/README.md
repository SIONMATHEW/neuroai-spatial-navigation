# Environmental comparison

This experiment asks how environmental structure changes the behavior and
hidden representation of a factorized RNN. It compares an open arena with an
occluded arena containing four textured pillars. The training objective,
architecture, optimizer settings and random-seed schedule are held constant.

The models are trained with sensory noise standard deviation 0.20 and tested
with noise standard deviation 0.25. Five paired seeds are run for 4,000
updates. Open and occluded trials are generated independently because the
geometries and valid trajectories differ.

Submit the five training jobs and the dependent evaluation job with:

```bash
bash submit_environment_comparison.sh
```

The machine-readable protocol is in `../configs/environment_comparison.json`.
The compact reviewed outputs are in `../results/environment_comparison/`.

## Per-bin accuracy and SIC

`evaluate_sic_bins.py` evaluates the frozen factorized networks on independent
noisy test trajectories and reports the open arena's 25/8 place/surface bins
and the occluded arena's 21/24 bins separately. It also computes place and
surface Skaggs information for every post-tanh hidden unit using a separate
probe bank.

Each environment has its own untrained-network SIC null because its geometry,
input distribution and label space differ. Consequently, the main population
comparison is the fraction of units exceeding the environment-specific null,
not a direct correspondence between unrelated bin identifiers. Run with:

```bash
sbatch run_sic_bins_h100.slurm
```

Set `ENVIRONMENT_RESULTS_ROOT` if the frozen result directory is stored at a
different location.

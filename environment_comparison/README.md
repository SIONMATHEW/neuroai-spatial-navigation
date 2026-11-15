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

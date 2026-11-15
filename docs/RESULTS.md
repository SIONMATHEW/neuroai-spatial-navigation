# Reviewed results

All summaries below are computed from paired network seeds. Hidden units and
timesteps are not treated as independent biological or statistical replicates.

## Clean objective comparison

At the best validation-selected checkpoints within the common 8,000-update
budget, conjunction training achieved higher mean held-out accuracy than
factorized training:

| Metric | Factorized | Conjunction | Conjunction minus factorized |
|---|---:|---:|---:|
| Place-bin accuracy | 96.76% | 98.46% | +1.70 percentage points |
| Gaze-surface accuracy | 94.65% | 97.83% | +3.18 percentage points |
| Exact joint accuracy | 92.73% | 97.44% | +4.71 percentage points |

These are five-seed means. The conjunction output head also contains more
parameters (204,792 total versus 86,829), so this comparison measures the full
training formulations rather than a parameter-count-matched architecture.

Four seeds could be matched near 83% validation exact-joint accuracy. Their
mean test exact-joint accuracies were 83.08% for factorized and 82.97% for
conjunction training. The factorized checkpoints required a mean of 928,000
training trajectories, compared with 512,000 for conjunction checkpoints.
Seed 1 was excluded from this matched analysis because no checkpoint pair met
both prespecified validation tolerances; it was not removed based on test data.

## Environmental comparison

The factorized networks were trained with noise standard deviation 0.20 and
evaluated with noise standard deviation 0.25. Across five paired seeds:

| Metric | Open mean ± seed SD | Occluded mean ± seed SD | Paired change |
|---|---:|---:|---:|
| Place-bin accuracy | 92.69% ± 1.80% | 91.32% ± 0.75% | −1.36 pp |
| Gaze-surface accuracy | 94.54% ± 0.15% | 90.43% ± 0.34% | −4.11 pp |
| Exact joint accuracy | 88.21% ± 1.82% | 84.29% ± 0.99% | −3.93 pp |
| Interaction-dependent units | 208.8 ± 4.3 | 205.4 ± 1.5 | −3.4 units |
| Median interaction gain | 0.0491 ± 0.0034 | 0.0322 ± 0.0017 | −0.0169 ΔR² |
| Minimum place–gaze angle | 79.64° ± 1.32° | 63.22° ± 3.25° | −16.42° |

The smaller principal angle and larger alignment index in the occluded model
indicate greater overlap between the fitted place and gaze effect subspaces.
This is a representational description, not evidence that individual units
map directly onto biological cell types.

The open and occluded environments have different numbers and meanings of
gaze-surface classes. Their trajectories are also generated independently
because pillars alter valid motion. The environmental results therefore show
the effect of the complete environment manipulation, not an isolated causal
effect of occlusion at fixed class cardinality.

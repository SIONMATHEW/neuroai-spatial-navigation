# Figure captions

## Objective performance

**Best validation-selected performance.** Held-out test accuracy for five
paired training seeds. Thin lines connect networks that began from the same
recurrent initialization and received the same sequence of training batches.
Large points and dark lines show seed means. Checkpoints were selected using
validation exact-joint accuracy within a common 8,000-update budget.

## Accuracy-matched objective comparison

**Validation-accuracy-matched checkpoints.** Exact-joint test accuracy and
training trajectories seen for four paired seeds whose checkpoints met the
prespecified 83% ± 0.5 percentage-point validation band and a maximum
between-model validation gap of 0.5 percentage points. Test outcomes did not
participate in matching. Seed 1 had no eligible pair.

## Environmental behavior

**Behavior under different environmental structures.** Held-out performance
of factorized RNNs trained in an open arena or a four-pillar occluded arena.
Models were trained with sensory noise SD 0.20 and tested at SD 0.25. Thin
lines connect common initialization seeds; trajectories are not literally
replayed between geometries. The two environments have different label spaces.

## Environmental hidden encoding

**Hidden representations under different environmental structures.** The
percentage of units classified as interaction-dependent, median held-out gain
from the saturated place-by-gaze model over the additive model, and minimum
principal angle between fitted place and gaze effect subspaces. Activity was
post-tanh. Probe trajectories were separated by whole trajectory into fitting,
ridge-selection and held-out sets. Thresholded cell categories are secondary
to the continuous held-out measures.

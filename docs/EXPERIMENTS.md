# Experimental protocols

## Clean objective comparison

The two models share a 36-channel input and a 256-unit tanh recurrent core.
The factorized model has separate 21-way place and 24-way gaze-surface heads.
The conjunction model has one 504-way joint head.

Both models are trained for the complete 8,000-update budget. A selected
checkpoint is therefore a retrospective validation choice, not an instruction
to stop the training run early. The best-checkpoint rule is strict maximum
validation exact-joint accuracy with the earliest checkpoint retained on a
tie.

The accuracy-matching rule is also validation-only. Each checkpoint must be
within 0.5 percentage points of the 83% target, and the two checkpoints must be
within 0.5 percentage points of one another. Eligible pairs are ranked by
pairwise gap, distance from the target and then number of updates. The held-out
test set never participates in matching.

## Frozen hidden-state analysis

The saved objective checkpoints are evaluated on new clean probe trajectories.
Post-tanh recurrent activity is predicted using categorical place, categorical
gaze, additive place-plus-gaze and saturated joint models. Ridge strength is
selected inside trajectory-grouped cross-validation. Reported predictions are
out-of-fold, and uncertainty is obtained by resampling whole trajectories.

The primary comparison uses the four validation-matched seed pairs. The five
validation-best pairs form a sensitivity analysis. Continuous interaction gain
is the primary hidden-unit outcome; thresholded categories are secondary.

## Environmental comparison

The open and occluded networks have the same input dimension, recurrent size,
optimizer and training budget. Their recurrent cores are initialized
identically within each seed. Trajectories are generated separately because
the pillars change collisions and valid paths; paired seed identifiers do not
imply literal replay of trajectories.

Noise is independent zero-mean Gaussian noise scaled by the calibration
standard deviation of each channel. It is applied to distance and texture
channels only. Movement channels remain clean. Validation uses the training
noise level of 0.20. Final behavior and hidden-state probes use noise 0.25.

The open label space has 25 place bins and 8 gaze-surface bins. The occluded
label space has 21 accessible place bins and 24 gaze-surface bins. Place and
gaze accuracy are therefore emphasized; joint accuracy is reported with this
class-cardinality limitation.

For hidden-state evaluation, 512 new trajectories per network are divided by
whole trajectory into 70% fitting, 15% ridge-selection and 15% held-out test
sets. Four categorical ridge models predict each post-tanh hidden unit: place
only, gaze only, additive place plus gaze, and a saturated place-by-gaze model.
Interaction gain is held-out `R²(interaction) - R²(additive)`. Confidence bounds
come from 200 resamples of complete held-out trajectories.

The thresholded unit labels are descriptive. A unit is called
interaction-dependent only when its interaction model explains at least 1%
of held-out variance and the lower 95% bootstrap bound on interaction gain is
greater than 0.01. Place-dominant, gaze-dominant and additive-mixed labels use
the corresponding unique held-out gains and require the interaction upper
bound to be no greater than 0.01. Continuous held-out scores should be
interpreted alongside these counts.

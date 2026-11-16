"""Synthetic checks for the SIC and per-bin performance helpers."""

from __future__ import annotations

import numpy as np

from analysis.sic import classify_units, min_shift, per_bin_accuracy, sic_score


def test_known_place_field_has_more_information_than_noise() -> None:
    rng = np.random.default_rng(7)
    samples = 20_000
    labels = rng.integers(0, 25, size=samples)
    activity = rng.uniform(0.0, 1.0, size=(samples, 5))
    activity[:, 0] = (labels == 3).astype(float)
    scores = sic_score(min_shift(activity), labels, min_visits=50)
    assert scores[0] > 10 * np.max(scores[1:])


def test_per_bin_accuracy_uses_the_true_bin_as_denominator() -> None:
    truth = np.array([0, 0, 1, 1, 1, 2])
    prediction = np.array([0, 1, 1, 1, 0, 2])
    correct, total, accuracy = per_bin_accuracy(truth, prediction, 3)
    np.testing.assert_array_equal(correct, [1, 2, 1])
    np.testing.assert_array_equal(total, [2, 3, 1])
    np.testing.assert_allclose(accuracy, [0.5, 2 / 3, 1.0])


def test_sic_categories_do_not_claim_interaction_selectivity() -> None:
    labels = classify_units(
        np.array([2.0, 0.2, 2.0, 0.2]),
        np.array([0.2, 2.0, 2.0, 0.2]),
        1.0,
        1.0,
    )
    np.testing.assert_array_equal(
        labels, ["place-only", "surface-only", "dual-selective", "not-significant"]
    )

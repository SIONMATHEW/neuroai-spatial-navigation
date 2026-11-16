"""Skaggs information and small helpers used by the frozen-model analyses.

The SIC implementation intentionally matches the standalone ``sic.py`` used
during the project.  Tanh activity is shifted to be non-negative separately
for every unit, and significance is assessed against untrained networks run on
the same inputs.  A label-shuffle null is not used because wall and landmark
inputs already contain position-dependent structure before training.
"""

from __future__ import annotations

import numpy as np


SIC_CATEGORIES = ("place-only", "surface-only", "dual-selective", "not-significant")


def min_shift(activity: np.ndarray) -> np.ndarray:
    """Shift each unit independently so its minimum observed activity is zero."""

    activity = np.asarray(activity, dtype=np.float64)
    if activity.ndim != 2:
        raise ValueError("activity must have shape (samples, units)")
    if len(activity) == 0:
        raise ValueError("activity cannot be empty")
    return activity - activity.min(axis=0, keepdims=True)


def sic_score(
    nonnegative_activity: np.ndarray,
    bin_index: np.ndarray,
    min_visits: int = 50,
) -> np.ndarray:
    """Return occupancy-weighted Skaggs information for every hidden unit.

    Bins with fewer than ``min_visits`` samples are excluded.  As in the
    original project implementation, occupancy remains divided by the total
    number of valid samples rather than being renormalised after exclusion.
    """

    activity = np.asarray(nonnegative_activity, dtype=np.float64)
    labels = np.asarray(bin_index)
    if activity.ndim != 2:
        raise ValueError("activity must have shape (samples, units)")
    if labels.ndim != 1 or len(labels) != len(activity):
        raise ValueError("bin_index must contain one label per activity sample")
    if min_visits < 1:
        raise ValueError("min_visits must be positive")
    if np.any(activity < -1e-12):
        raise ValueError("SIC requires non-negative activity; apply min_shift first")

    scores = np.zeros(activity.shape[1], dtype=np.float64)
    included: list[tuple[float, np.ndarray]] = []
    for label in np.unique(labels):
        selected = labels == label
        count = int(selected.sum())
        if count >= min_visits:
            included.append((count / len(labels), activity[selected].mean(axis=0)))
    if not included:
        return scores

    overall_rate = sum(occupancy * rate for occupancy, rate in included)
    for occupancy, rate in included:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(overall_rate > 0, rate / overall_rate, 0.0)
            log_ratio = np.where(ratio > 0, np.log2(ratio), 0.0)
        scores += occupancy * ratio * log_ratio
    return scores


def null_threshold(null_scores: np.ndarray, percentile: float = 99.0) -> float:
    """Return a percentile threshold from pooled untrained-network SIC scores."""

    values = np.asarray(null_scores, dtype=np.float64).ravel()
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("null_scores must be a non-empty finite array")
    if not 0 < percentile < 100:
        raise ValueError("percentile must lie strictly between 0 and 100")
    return float(np.percentile(values, percentile))


def classify_units(
    place_sic: np.ndarray,
    surface_sic: np.ndarray,
    place_threshold: float,
    surface_threshold: float,
) -> np.ndarray:
    """Classify units by whether place and surface SIC exceed their nulls.

    ``dual-selective`` means significant information about both variables.  It
    does not by itself imply a nonlinear place-by-surface interaction.
    """

    place = np.asarray(place_sic) > place_threshold
    surface = np.asarray(surface_sic) > surface_threshold
    if place.shape != surface.shape:
        raise ValueError("place_sic and surface_sic must have the same shape")
    labels = np.full(place.shape, "not-significant", dtype=object)
    labels[place & ~surface] = "place-only"
    labels[~place & surface] = "surface-only"
    labels[place & surface] = "dual-selective"
    return labels


def per_bin_accuracy(
    truth: np.ndarray,
    prediction: np.ndarray,
    number_of_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return correct counts, total counts and accuracy for each true bin."""

    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    if truth.shape != prediction.shape:
        raise ValueError("truth and prediction must have the same shape")
    if truth.ndim != 1:
        raise ValueError("truth and prediction must be one-dimensional")
    if number_of_bins < 1:
        raise ValueError("number_of_bins must be positive")
    if np.any((truth < 0) | (truth >= number_of_bins)):
        raise ValueError("truth contains a label outside the declared bin range")

    total = np.bincount(truth, minlength=number_of_bins).astype(np.int64)
    correct = np.bincount(
        truth[truth == prediction], minlength=number_of_bins
    ).astype(np.int64)
    accuracy = np.full(number_of_bins, np.nan, dtype=np.float64)
    np.divide(correct, total, out=accuracy, where=total > 0)
    return correct, total, accuracy


def mean_activity_by_bin(
    activity: np.ndarray,
    labels: np.ndarray,
    number_of_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-bin mean activity and sample counts for every unit."""

    activity = np.asarray(activity, dtype=np.float64)
    labels = np.asarray(labels)
    if activity.ndim != 2 or labels.ndim != 1 or len(activity) != len(labels):
        raise ValueError("activity and labels have incompatible shapes")
    means = np.full((number_of_bins, activity.shape[1]), np.nan, dtype=np.float64)
    counts = np.bincount(labels, minlength=number_of_bins).astype(np.int64)
    for label in range(number_of_bins):
        selected = labels == label
        if np.any(selected):
            means[label] = activity[selected].mean(axis=0)
    return means, counts


def choose_representative_units(
    place_sic: np.ndarray,
    surface_sic: np.ndarray,
    categories: np.ndarray,
) -> dict[str, int]:
    """Choose examples with a fixed, auditable rule rather than by appearance."""

    place_sic = np.asarray(place_sic)
    surface_sic = np.asarray(surface_sic)
    categories = np.asarray(categories)
    selected: dict[str, int] = {}
    rules = {
        "place-only": place_sic,
        "surface-only": surface_sic,
        "dual-selective": np.minimum(place_sic, surface_sic),
    }
    for category, score in rules.items():
        candidates = np.flatnonzero(categories == category)
        if len(candidates):
            selected[category] = int(candidates[np.argmax(score[candidates])])
    return selected

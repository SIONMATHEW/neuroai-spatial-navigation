"""Regenerate the compact report figures from versioned CSV result tables.

This script does not load checkpoints or rerun an experiment. It only turns
the reviewed tables in ``results/`` into presentation-ready figures.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "figures" / "paper"

BLUE = "#377eb8"
RED = "#e4574f"
DARK = "#111827"
GRID = "#d9dee5"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finish_figure(fig: plt.Figure, filename: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def paired_axis(
    ax: plt.Axes,
    left: np.ndarray,
    right: np.ndarray,
    labels: tuple[str, str],
    ylabel: str,
    title: str,
) -> None:
    for a, b in zip(left, right, strict=True):
        ax.plot([0, 1], [a, b], color="#aeb7c4", linewidth=1.3, zorder=1)
        ax.scatter(0, a, color=BLUE, s=32, alpha=0.8, zorder=2)
        ax.scatter(1, b, color=RED, s=32, alpha=0.8, zorder=2)

    means = [left.mean(), right.mean()]
    ax.plot([0, 1], means, color=DARK, linewidth=3.0, zorder=3)
    ax.scatter([0, 1], means, c=[BLUE, RED], edgecolor=DARK, linewidth=1.5, s=125, zorder=4)
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def objective_best_figure() -> None:
    rows = read_csv(
        RESULTS / "objective_comparison" / "best_within_budget" / "metrics_by_seed.csv"
    )
    metrics = [
        ("place_accuracy", "Place-bin accuracy"),
        ("surface_accuracy", "Gaze-surface accuracy"),
        ("joint_accuracy", "Exact joint accuracy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        factorized = np.array([float(row[f"factorized_{metric}"]) for row in rows])
        conjunction = np.array([float(row[f"conjunction_{metric}"]) for row in rows])
        paired_axis(
            ax,
            factorized,
            conjunction,
            ("Factorized", "Conjunction"),
            "Held-out test accuracy" if ax is axes[0] else "",
            title,
        )
        ax.set_ylim(0.88, 1.0)
    fig.suptitle("Best validation-selected checkpoint within 8,000 updates", fontweight="bold")
    fig.tight_layout()
    finish_figure(fig, "objective_best_accuracy.png")


def objective_matched_figure() -> None:
    rows = read_csv(
        RESULTS / "objective_comparison" / "accuracy_matched" / "metrics_by_seed.csv"
    )
    factorized_accuracy = np.array([float(row["factorized_joint_accuracy"]) for row in rows])
    conjunction_accuracy = np.array([float(row["conjunction_joint_accuracy"]) for row in rows])
    factorized_trials = np.array(
        [float(row["factorized_selected_training_trials"]) for row in rows]
    )
    conjunction_trials = np.array(
        [float(row["conjunction_selected_training_trials"]) for row in rows]
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    paired_axis(
        axes[0],
        factorized_accuracy,
        conjunction_accuracy,
        ("Factorized", "Conjunction"),
        "Held-out test exact-joint accuracy",
        "Test performance",
    )
    axes[0].set_ylim(0.80, 0.86)
    paired_axis(
        axes[1],
        factorized_trials / 1_000,
        conjunction_trials / 1_000,
        ("Factorized", "Conjunction"),
        "Training trajectories (thousands)",
        "Data required to reach the match",
    )
    fig.suptitle("Validation-accuracy-matched checkpoints (four paired seeds)", fontweight="bold")
    fig.tight_layout()
    finish_figure(fig, "objective_accuracy_matched.png")


def environment_behavior_figure() -> None:
    rows = read_csv(RESULTS / "environment_comparison" / "behavior_metrics.csv")
    by_environment = {
        environment: {int(row["seed"]): row for row in rows if row["environment"] == environment}
        for environment in ("open", "occluded")
    }
    metrics = [
        ("place_accuracy", "Place-bin accuracy"),
        ("surface_accuracy", "Gaze-surface accuracy"),
        ("joint_accuracy", "Exact joint accuracy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        open_values = np.array(
            [float(by_environment["open"][seed][metric]) for seed in range(1, 6)]
        )
        occluded_values = np.array(
            [float(by_environment["occluded"][seed][metric]) for seed in range(1, 6)]
        )
        paired_axis(
            ax,
            open_values,
            occluded_values,
            ("Open", "Occluded"),
            "Held-out test accuracy" if ax is axes[0] else "",
            title,
        )
        ax.set_ylim(0.78, 1.0)
    fig.suptitle("Factorized RNNs trained with sensory noise (five paired seeds)", fontweight="bold")
    fig.tight_layout()
    finish_figure(fig, "environment_behavior.png")


def environment_hidden_figure() -> None:
    network_rows = read_csv(RESULTS / "environment_comparison" / "network_summary.csv")
    subspace_rows = read_csv(RESULTS / "environment_comparison" / "subspace_metrics.csv")
    network = {
        environment: {int(row["seed"]): row for row in network_rows if row["environment"] == environment}
        for environment in ("open", "occluded")
    }
    subspace = {
        environment: {int(row["seed"]): row for row in subspace_rows if row["environment"] == environment}
        for environment in ("open", "occluded")
    }

    interaction_open = np.array(
        [100 * int(network["open"][seed]["interaction-dependent_units"]) / 256 for seed in range(1, 6)]
    )
    interaction_occluded = np.array(
        [100 * int(network["occluded"][seed]["interaction-dependent_units"]) / 256 for seed in range(1, 6)]
    )
    gain_open = np.array(
        [float(network["open"][seed]["median_interaction_gain"]) for seed in range(1, 6)]
    )
    gain_occluded = np.array(
        [float(network["occluded"][seed]["median_interaction_gain"]) for seed in range(1, 6)]
    )
    angle_open = np.array(
        [float(subspace["open"][seed]["minimum_angle_degrees"]) for seed in range(1, 6)]
    )
    angle_occluded = np.array(
        [float(subspace["occluded"][seed]["minimum_angle_degrees"]) for seed in range(1, 6)]
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    paired_axis(
        axes[0], interaction_open, interaction_occluded, ("Open", "Occluded"),
        "Units (%)", "Interaction-dependent units",
    )
    paired_axis(
        axes[1], gain_open, gain_occluded, ("Open", "Occluded"),
        "Median held-out ΔR²", "Interaction gain",
    )
    paired_axis(
        axes[2], angle_open, angle_occluded, ("Open", "Occluded"),
        "Minimum principal angle (degrees)", "Place–gaze subspace angle",
    )
    fig.suptitle("Hidden representation under different environmental structures", fontweight="bold")
    fig.tight_layout()
    finish_figure(fig, "environment_hidden_encoding.png")


def main() -> None:
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    objective_best_figure()
    objective_matched_figure()
    environment_behavior_figure()
    environment_hidden_figure()
    print(f"Figures written to {OUTPUT}")


if __name__ == "__main__":
    main()

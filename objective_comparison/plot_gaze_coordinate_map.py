"""Plot gaze-coordinate error along the arena walls and pillar surfaces.

The script reads the per-seed ``raw_evaluation/seed_*.npz`` files produced by
``controlled_comparison.py``.  True gaze-intersection points are assigned to
short spatial subsegments of their true surface.  For each seed and subsegment,
the median coordinate error is calculated.  The displayed value is the mean of
those seed medians, so each training seed contributes equally.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np


SURFACE_BINS = 24


def restore_physical(normalized: np.ndarray) -> np.ndarray:
    """Convert normalized coordinates back to the 0--10 arena."""
    return 10.0 * (normalized - 0.05) / 0.8


def arena_surface_segments() -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the 24 surfaces in the order used by RNNTrialStructures."""
    segments: list[tuple[np.ndarray, np.ndarray]] = []

    # Two five-grid-square bins on each outer wall.
    corners = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    for start, end in zip(corners, corners[1:] + corners[:1]):
        p1 = np.asarray(start, dtype=float)
        p2 = np.asarray(end, dtype=float)
        midpoint = (p1 + p2) / 2.0
        segments.append((p1, midpoint))
        segments.append((midpoint, p2))

    # Four faces for each of the four pillars.
    pillar_bounds = [(2, 2, 4, 4), (6, 2, 8, 4), (6, 6, 8, 8), (2, 6, 4, 8)]
    for xmin, ymin, xmax, ymax in pillar_bounds:
        points = [
            np.asarray(point, dtype=float)
            for point in [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        ]
        segments.extend(zip(points, points[1:] + points[:1]))

    if len(segments) != SURFACE_BINS:
        raise AssertionError(len(segments))
    return segments


def make_spatial_subsegments(
    resolution: float,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Split every true surface into short colored line segments."""
    surfaces = arena_surface_segments()
    lines: list[np.ndarray] = []
    surface_ids: list[int] = []
    local_ids: list[int] = []
    counts: list[int] = []

    for surface_id, (start, end) in enumerate(surfaces):
        length = float(np.linalg.norm(end - start))
        count = max(1, int(round(length / resolution)))
        counts.append(count)
        for local_id in range(count):
            left = start + (end - start) * (local_id / count)
            right = start + (end - start) * ((local_id + 1) / count)
            lines.append(np.stack([left, right]))
            surface_ids.append(surface_id)
            local_ids.append(local_id)

    return (
        lines,
        np.asarray(surface_ids, dtype=int),
        np.asarray(local_ids, dtype=int),
        np.asarray(counts, dtype=int),
    )


def spatial_group_ids(
    gaze_xy: np.ndarray,
    true_surface: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Assign each true gaze point to a spatial subsegment of its surface."""
    surfaces = arena_surface_segments()
    starts = np.stack([item[0] for item in surfaces])
    deltas = np.stack([item[1] - item[0] for item in surfaces])
    squared_lengths = np.sum(deltas * deltas, axis=1)
    offsets = np.concatenate([[0], np.cumsum(counts[:-1])])

    start = starts[true_surface]
    delta = deltas[true_surface]
    fraction = np.sum((gaze_xy - start) * delta, axis=1) / squared_lengths[true_surface]
    fraction = np.clip(fraction, 0.0, np.nextafter(1.0, 0.0))
    local = np.floor(fraction * counts[true_surface]).astype(int)
    return offsets[true_surface] + local


def grouped_medians(
    values: np.ndarray,
    groups: np.ndarray,
    group_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    medians = np.full(group_count, np.nan, dtype=float)
    support = np.zeros(group_count, dtype=np.int64)
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    sorted_values = values[order]
    unique, starts, counts = np.unique(
        sorted_groups, return_index=True, return_counts=True
    )
    for group, start, count in zip(unique, starts, counts):
        medians[group] = float(np.median(sorted_values[start : start + count]))
        support[group] = int(count)
    return medians, support


def draw_arena_background(ax: plt.Axes) -> None:
    for start, end in arena_surface_segments():
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="#D4D4D4",
            linewidth=11,
            solid_capstyle="butt",
            zorder=0,
        )
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_yticks(np.arange(0, 11, 2))
    ax.set_xlabel("Arena x-coordinate")
    ax.set_ylabel("Arena y-coordinate")
    ax.grid(alpha=0.12)


def add_colored_segments(
    ax: plt.Axes,
    lines: list[np.ndarray],
    values: np.ndarray,
    cmap: str,
    norm: Normalize,
) -> LineCollection:
    collection = LineCollection(
        lines,
        cmap=cmap,
        norm=norm,
        linewidths=9,
        capstyle="butt",
        zorder=2,
    )
    collection.set_array(values)
    ax.add_collection(collection)
    return collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to the results directory.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.5,
        help="Length of each displayed surface subsegment in grid squares.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_paths = sorted((args.results_dir / "raw_evaluation").glob("seed_*.npz"))
    if not seed_paths:
        raise SystemExit("No raw_evaluation/seed_*.npz files found")

    lines, surface_ids, local_ids, counts = make_spatial_subsegments(args.resolution)
    group_count = len(lines)
    continuous_by_seed = []
    conjunction_by_seed = []
    support_by_seed = []

    for seed_path in seed_paths:
        with np.load(seed_path) as data:
            true_surface = data["true_class"] % SURFACE_BINS
            true_gaze = restore_physical(data["true_xy"][:, 2:])
            groups = spatial_group_ids(true_gaze, true_surface, counts)
            continuous, support = grouped_medians(
                data["continuous_gaze_error"], groups, group_count
            )
            conjunction, _ = grouped_medians(
                data["conjunction_gaze_error"], groups, group_count
            )
        continuous_by_seed.append(continuous)
        conjunction_by_seed.append(conjunction)
        support_by_seed.append(support)

    continuous_by_seed = np.stack(continuous_by_seed)
    conjunction_by_seed = np.stack(conjunction_by_seed)
    support_by_seed = np.stack(support_by_seed)
    continuous_mean = np.nanmean(continuous_by_seed, axis=0)
    conjunction_mean = np.nanmean(conjunction_by_seed, axis=0)
    difference = continuous_mean - conjunction_mean

    shared_max = float(np.nanmax(np.concatenate([continuous_mean, conjunction_mean])))
    shared_norm = Normalize(vmin=0.0, vmax=shared_max)
    difference_max = float(np.nanmax(np.abs(difference)))
    difference_norm = TwoSlopeNorm(
        vmin=-difference_max, vcenter=0.0, vmax=difference_max
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7), dpi=180)
    titles = [
        "Continuous-output model",
        "Conjunction-trained model",
        "Error difference\n(continuous minus conjunction)",
    ]
    for ax, title in zip(axes, titles):
        draw_arena_background(ax)
        ax.set_title(title)

    first = add_colored_segments(
        axes[0], lines, continuous_mean, "magma", shared_norm
    )
    add_colored_segments(axes[1], lines, conjunction_mean, "magma", shared_norm)
    third = add_colored_segments(
        axes[2], lines, difference, "RdBu_r", difference_norm
    )

    shared_bar = fig.colorbar(first, ax=axes[:2], fraction=0.025, pad=0.025)
    shared_bar.set_label("Median gaze-coordinate error (grid squares)")
    difference_bar = fig.colorbar(third, ax=axes[2], fraction=0.046, pad=0.04)
    difference_bar.set_label("Positive = conjunction has lower error")
    fig.suptitle(
        f"Gaze-coordinate error across arena surfaces (mean of {len(seed_paths)} seed medians)",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.12, top=0.84, wspace=0.28)
    figure_path = output_dir / "gaze_coordinate_error_spatial_map.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    offsets = np.concatenate([[0], np.cumsum(counts[:-1])])
    csv_path = output_dir / "gaze_coordinate_error_by_segment.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "surface_bin",
                "subsegment",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "continuous_seed_mean_median_error",
                "continuous_seed_sd",
                "conjunction_seed_mean_median_error",
                "conjunction_seed_sd",
                "continuous_minus_conjunction",
                "total_support",
                "n_seeds",
            ]
        )
        for group, line in enumerate(lines):
            writer.writerow(
                [
                    int(surface_ids[group] + 1),
                    int(local_ids[group] + 1),
                    float(line[0, 0]),
                    float(line[0, 1]),
                    float(line[1, 0]),
                    float(line[1, 1]),
                    float(continuous_mean[group]),
                    float(np.nanstd(continuous_by_seed[:, group], ddof=1)),
                    float(conjunction_mean[group]),
                    float(np.nanstd(conjunction_by_seed[:, group], ddof=1)),
                    float(difference[group]),
                    int(np.sum(support_by_seed[:, group])),
                    len(seed_paths),
                ]
            )

    print(f"Wrote {figure_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

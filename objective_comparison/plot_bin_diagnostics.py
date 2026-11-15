"""Create per-bin error maps and a paired overall-accuracy plot.

This is a post-processing companion to controlled_comparison.py. It can run
once after results exist or watch the results directory and refresh whenever a
new seed finishes.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLACE_BINS = 21
SURFACE_BINS = 24
PILLARS = {(1, 1), (3, 1), (1, 3), (3, 3)}


def mean_available(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    total = np.nansum(values, axis=0)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def load_raw(raw_dir: Path):
    files = sorted(raw_dir.glob("seed_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
    seeds = []
    for path in files:
        with np.load(path) as data:
            seeds.append(
                {
                    "seed": int(path.stem.split("_")[-1]),
                    "true": data["true_class"].astype(np.int64),
                    "continuous": data["continuous_class"].astype(np.int64),
                    "conjunction": data["conjunction_class"].astype(np.int64),
                }
            )
    return seeds


def per_bin_error(true: np.ndarray, prediction: np.ndarray, component: str):
    if component == "place":
        true_component = true // SURFACE_BINS
        prediction_component = np.where(prediction >= 0, prediction // SURFACE_BINS, -1)
        count = PLACE_BINS
    elif component == "surface":
        true_component = true % SURFACE_BINS
        prediction_component = np.where(prediction >= 0, prediction % SURFACE_BINS, -1)
        count = SURFACE_BINS
    else:
        raise ValueError(component)

    errors = np.full(count, np.nan)
    support = np.zeros(count, dtype=np.int64)
    for index in range(count):
        selected = true_component == index
        support[index] = int(selected.sum())
        if support[index]:
            errors[index] = np.mean(prediction_component[selected] != index)
    return errors, support


def collect_bin_results(seed_data):
    output = {
        "continuous_place": [],
        "conjunction_place": [],
        "continuous_surface": [],
        "conjunction_surface": [],
    }
    supports = {"place": [], "surface": []}

    for data in seed_data:
        for model in ("continuous", "conjunction"):
            place, place_support = per_bin_error(data["true"], data[model], "place")
            surface, surface_support = per_bin_error(data["true"], data[model], "surface")
            output[f"{model}_place"].append(place)
            output[f"{model}_surface"].append(surface)
        supports["place"].append(place_support)
        supports["surface"].append(surface_support)

    return (
        {key: np.asarray(value) for key, value in output.items()},
        {key: np.asarray(value) for key, value in supports.items()},
    )


def floor_grid(values: np.ndarray) -> np.ndarray:
    cells = [
        (x, y)
        for x in range(5)
        for y in range(5)
        if (x, y) not in PILLARS
    ]
    grid = np.full((5, 5), np.nan)
    for index, (x, y) in enumerate(cells):
        grid[y, x] = values[index]
    return grid


def plot_place_map(results, output_dir: Path, n_seeds: int):
    continuous = mean_available(results["continuous_place"])
    conjunction = mean_available(results["conjunction_place"])
    values = [continuous, conjunction]
    titles = ["Continuous-output model", "Conjunction-trained model"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=160, constrained_layout=True)
    cmap = plt.cm.magma.copy()
    cmap.set_bad("black")
    image = None
    for ax, vector, title in zip(axes, values, titles):
        grid = floor_grid(vector)
        image = ax.imshow(grid, origin="lower", vmin=0, vmax=1, cmap=cmap)
        for y in range(5):
            for x in range(5):
                if (x, y) in PILLARS:
                    ax.text(x, y, "pillar", ha="center", va="center", color="white", fontsize=8)
                else:
                    color = "black" if grid[y, x] > 0.55 else "white"
                    ax.text(x, y, f"{100 * grid[y, x]:.1f}%", ha="center", va="center", color=color, fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Floor x-bin")
        ax.set_ylabel("Floor y-bin")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(image, ax=axes, label="Wrong place-bin rate", shrink=0.83)
    fig.suptitle(f"Place-bin error by true floor bin (mean across {n_seeds} seed{'s' if n_seeds != 1 else ''})")
    fig.savefig(output_dir / "place_bin_error_map.png", dpi=300)
    plt.close(fig)


def plot_surface_map(results, output_dir: Path, n_seeds: int):
    matrix = np.vstack(
        [
            mean_available(results["continuous_surface"]),
            mean_available(results["conjunction_surface"]),
        ]
    )
    fig, ax = plt.subplots(figsize=(16, 3.3), dpi=160, constrained_layout=True)
    cmap = plt.cm.magma.copy()
    cmap.set_bad("lightgray")
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", vmin=0, vmax=1, cmap=cmap)
    for row in range(2):
        for column in range(SURFACE_BINS):
            value = matrix[row, column]
            if np.isfinite(value):
                color = "black" if value > 0.55 else "white"
                label = f"{100 * value:.0f}%"
            else:
                color = "#555555"
                label = "no data"
            ax.text(column, row, label, ha="center", va="center", color=color, fontsize=7)
    ax.set_xticks(range(SURFACE_BINS))
    ax.set_xticklabels(range(1, SURFACE_BINS + 1))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Continuous-output", "Conjunction-trained"])
    ax.set_xlabel("True surface bin")
    ax.set_title(f"Surface-bin error by true gaze surface (mean across {n_seeds} seed{'s' if n_seeds != 1 else ''})")
    ax.set_xticks(np.arange(-0.5, SURFACE_BINS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(image, ax=ax, label="Wrong surface-bin rate", shrink=0.9)
    fig.savefig(output_dir / "surface_bin_error_map.png", dpi=300)
    plt.close(fig)


def read_metrics(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plot_accuracy_dumbbell(metric_rows, output_dir: Path):
    if not metric_rows:
        return
    keys = [
        ("Position bin", "continuous_position_accuracy", "conjunction_position_accuracy"),
        ("Surface bin", "continuous_surface_accuracy", "conjunction_surface_accuracy"),
        ("Exact conjunction", "continuous_conjunction_accuracy", "conjunction_conjunction_accuracy"),
    ]
    continuous = np.asarray(
        [[float(row[old]) for _, old, _ in keys] for row in metric_rows]
    )
    conjunction = np.asarray(
        [[float(row[new]) for _, _, new in keys] for row in metric_rows]
    )
    y = np.arange(len(keys))[::-1]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)

    # Thin paired lines show each seed; the thick line joins across-seed means.
    for seed_index in range(len(metric_rows)):
        for metric_index, yvalue in enumerate(y):
            ax.plot(
                [continuous[seed_index, metric_index], conjunction[seed_index, metric_index]],
                [yvalue, yvalue],
                color="#AAB1B7",
                linewidth=1,
                alpha=0.55,
            )
            ax.scatter(continuous[seed_index, metric_index], yvalue, color="#3976AF", s=18, alpha=0.45)
            ax.scatter(conjunction[seed_index, metric_index], yvalue, color="#D64B40", s=18, alpha=0.45)

    old_mean = continuous.mean(axis=0)
    new_mean = conjunction.mean(axis=0)
    for metric_index, yvalue in enumerate(y):
        ax.plot([old_mean[metric_index], new_mean[metric_index]], [yvalue, yvalue], color="#444444", linewidth=2.5)
    ax.scatter(old_mean, y, color="#3976AF", s=95, label="Continuous-output model", zorder=4)
    ax.scatter(new_mean, y, color="#D64B40", s=95, label="Conjunction-trained model", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _ in keys])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Held-out accuracy")
    ax.set_title("Overall controlled comparison (paired by training seed)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "overall_accuracy_dumbbell.png", dpi=300)
    plt.close(fig)


def write_per_bin_csv(results, supports, output_dir: Path):
    path = output_dir / "per_bin_error_rates.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["component", "bin", "model", "mean_error_rate", "seed_sd", "total_support", "n_seeds"])
        for component, count in (("place", PLACE_BINS), ("surface", SURFACE_BINS)):
            for model in ("continuous", "conjunction"):
                values = results[f"{model}_{component}"]
                total_support = supports[component].sum(axis=0)
                for index in range(count):
                    finite = values[:, index][np.isfinite(values[:, index])]
                    writer.writerow(
                        [
                            component,
                            index + 1,
                            model,
                            float(finite.mean()) if len(finite) else "",
                            float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
                            int(total_support[index]),
                            len(values),
                        ]
                    )


def render_once(results_dir: Path) -> int:
    seed_data = load_raw(results_dir / "raw_evaluation")
    if not seed_data:
        return 0
    results, supports = collect_bin_results(seed_data)
    plot_place_map(results, results_dir, len(seed_data))
    plot_surface_map(results, results_dir, len(seed_data))
    plot_accuracy_dumbbell(read_metrics(results_dir / "metrics_by_seed.csv"), results_dir)
    write_per_bin_csv(results, supports, results_dir)
    return len(seed_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-hours", type=float, default=24)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if not args.watch:
        count = render_once(args.results_dir)
        if count == 0:
            raise SystemExit("No raw seed files are available yet")
        print(f"Rendered diagnostics from {count} seed(s)", flush=True)
        return

    deadline = time.time() + args.timeout_hours * 3600
    previous = -1
    while time.time() < deadline:
        count = render_once(args.results_dir)
        if count != previous:
            print(f"Rendered diagnostics from {count} seed(s)", flush=True)
            previous = count
        if count >= args.expected_seeds:
            return
        time.sleep(args.poll_seconds)
    raise SystemExit("Timed out while waiting for all seed results")


if __name__ == "__main__":
    main()

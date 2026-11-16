"""Per-bin performance and SIC for frozen factorized/conjunction RNNs.

The primary comparison uses the validation-accuracy-matched checkpoints.  The
saved held-out test predictions are reused for per-bin accuracy, so this script
does not alter model selection or test the models on a different data set.

SIC is computed from new clean probe trajectories that were not used for
training, validation, checkpoint matching or behavioral testing.  Factorized
and conjunction networks receive the same probe inputs within a seed.  The
significance threshold is the 99th percentile of SIC pooled over untrained
RNNs of the same recurrent architecture run on those same inputs.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep the Julia bridge import before torch on the cluster.
import factorized_vs_conjunction as experiment
import controlled_comparison as base
import evaluate_frozen_hidden as frozen

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.sic import (
    SIC_CATEGORIES,
    choose_representative_units,
    classify_units,
    mean_activity_by_bin,
    min_shift,
    null_threshold,
    per_bin_accuracy,
    sic_score,
)


MODELS = ("factorized", "conjunction")
MODEL_LABELS = {"factorized": "Factorized", "conjunction": "Conjunction"}
MODEL_COLORS = {"factorized": "#3976AF", "conjunction": "#D95F55"}
CATEGORY_COLORS = {
    "place-only": "#4C78A8",
    "surface-only": "#F2A541",
    "dual-selective": "#8F5DA2",
    "not-significant": "#A9A9A9",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def mean_and_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    if values.shape[0] < 2:
        return mean, np.zeros_like(mean)
    return mean, np.nanstd(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def floor_grid(values: np.ndarray) -> np.ndarray:
    return experiment.floor_grid(np.asarray(values, dtype=float))


def saved_test_arrays(results_root: Path, seed: int) -> dict[str, np.ndarray]:
    path = (
        results_root
        / "seed_runs"
        / f"seed_{seed}"
        / "accuracy_matched_raw_evaluation.npz"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. This analysis requires the saved held-out predictions "
            "from the accuracy-matched comparison."
        )
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = {
        "true_place",
        "true_surface",
        "factorized_place",
        "factorized_surface",
        "conjunction_place",
        "conjunction_surface",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise RuntimeError(f"{path} is missing arrays: {missing}")
    return arrays


def bin_accuracy_rows(results_root: Path, seeds: list[int]) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        arrays = saved_test_arrays(results_root, seed)
        for variable, number_of_bins in (
            ("place", experiment.PLACE_BINS),
            ("surface", experiment.SURFACE_BINS),
        ):
            truth = arrays[f"true_{variable}"]
            for model in MODELS:
                correct, total, accuracy = per_bin_accuracy(
                    truth, arrays[f"{model}_{variable}"], number_of_bins
                )
                for label in range(number_of_bins):
                    rows.append(
                        {
                            "seed": seed,
                            "model": model,
                            "variable": variable,
                            "bin": label + 1,
                            "correct": int(correct[label]),
                            "total": int(total[label]),
                            "accuracy": float(accuracy[label]),
                        }
                    )
    return rows


def pooled_null_scores(
    probe: dict,
    device: torch.device,
    inference_batch_size: int,
    number_of_networks: int,
    seed: int,
    min_visits: int,
) -> tuple[np.ndarray, np.ndarray]:
    place_scores, surface_scores = [], []
    for index in range(number_of_networks):
        torch.manual_seed(90_000_000 + 10_000 * seed + index)
        model = experiment.FactorizedDiscreteRNN().to(device).eval()
        hidden = frozen.extract_hidden(model, probe, device, inference_batch_size)
        shifted = min_shift(hidden)
        place_scores.append(sic_score(shifted, probe["place"], min_visits))
        surface_scores.append(sic_score(shifted, probe["gaze"], min_visits))
        del model, hidden, shifted
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(place_scores), np.concatenate(surface_scores)


def network_sic(
    model: torch.nn.Module,
    probe: dict,
    device: torch.device,
    inference_batch_size: int,
    min_visits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = frozen.extract_hidden(model, probe, device, inference_batch_size)
    shifted = min_shift(hidden)
    place = sic_score(shifted, probe["place"], min_visits)
    surface = sic_score(shifted, probe["gaze"], min_visits)
    return hidden, place, surface


def sic_rows_for_seed(
    seed: int,
    paths: dict[str, Path],
    probe: dict,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], list[dict], dict[str, dict]]:
    null_place, null_surface = pooled_null_scores(
        probe,
        device,
        args.inference_batch_size,
        args.null_networks,
        seed,
        args.min_visits,
    )
    place_threshold = null_threshold(null_place, args.null_percentile)
    surface_threshold = null_threshold(null_surface, args.null_percentile)
    unit_rows: list[dict] = []
    summary_rows: list[dict] = []
    support_rows: list[dict] = []
    tuning: dict[str, dict] = {}

    for variable, labels, number_of_bins in (
        ("place", probe["place"], experiment.PLACE_BINS),
        ("surface", probe["gaze"], experiment.SURFACE_BINS),
    ):
        counts = np.bincount(labels, minlength=number_of_bins)
        support_rows.extend(
            {
                "seed": seed,
                "variable": variable,
                "bin": label + 1,
                "valid_timesteps": int(counts[label]),
                "included_in_sic": int(counts[label] >= args.min_visits),
            }
            for label in range(number_of_bins)
        )

    for model_name, checkpoint_path in paths.items():
        model, checkpoint = frozen.load_model(checkpoint_path, model_name, device)
        hidden, place_sic, surface_sic = network_sic(
            model,
            probe,
            device,
            args.inference_batch_size,
            args.min_visits,
        )
        categories = classify_units(
            place_sic, surface_sic, place_threshold, surface_threshold
        )
        counts = Counter(categories)
        for unit, category in enumerate(categories):
            unit_rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "unit": unit,
                    "place_sic": float(place_sic[unit]),
                    "surface_sic": float(surface_sic[unit]),
                    "place_null_threshold": place_threshold,
                    "surface_null_threshold": surface_threshold,
                    "place_sic_over_threshold": float(place_sic[unit] / place_threshold)
                    if place_threshold > 0
                    else float("nan"),
                    "surface_sic_over_threshold": float(surface_sic[unit] / surface_threshold)
                    if surface_threshold > 0
                    else float("nan"),
                    "place_significant": int(place_sic[unit] > place_threshold),
                    "surface_significant": int(surface_sic[unit] > surface_threshold),
                    "sic_category": str(category),
                }
            )
        summary_rows.append(
            {
                "seed": seed,
                "model": model_name,
                "selected_update": int(checkpoint["best_update"]),
                "probe_trajectories": int(len(np.unique(probe["trajectory"]))),
                "probe_valid_timesteps": int(len(hidden)),
                "place_bins_meeting_min_visits": int(
                    np.sum(np.bincount(probe["place"], minlength=experiment.PLACE_BINS) >= args.min_visits)
                ),
                "surface_bins_meeting_min_visits": int(
                    np.sum(np.bincount(probe["gaze"], minlength=experiment.SURFACE_BINS) >= args.min_visits)
                ),
                "median_place_sic": float(np.median(place_sic)),
                "median_surface_sic": float(np.median(surface_sic)),
                "place_null_threshold": place_threshold,
                "surface_null_threshold": surface_threshold,
                **{f"{category}_units": int(counts[category]) for category in SIC_CATEGORIES},
                **{
                    f"{category}_percent": 100.0 * counts[category] / len(categories)
                    for category in SIC_CATEGORIES
                },
            }
        )
        shifted = min_shift(hidden)
        place_tuning, place_counts = mean_activity_by_bin(
            shifted, probe["place"], experiment.PLACE_BINS
        )
        surface_tuning, surface_counts = mean_activity_by_bin(
            shifted, probe["gaze"], experiment.SURFACE_BINS
        )
        tuning[model_name] = {
            "place_sic": place_sic,
            "surface_sic": surface_sic,
            "categories": categories,
            "place_tuning": place_tuning,
            "surface_tuning": surface_tuning,
            "place_counts": place_counts,
            "surface_counts": surface_counts,
        }
        del model, hidden, shifted
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return unit_rows, summary_rows, support_rows, tuning


def plot_bin_accuracy(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    place_values = []
    for model in MODELS:
        selected = [r for r in rows if r["model"] == model and r["variable"] == "place"]
        matrix = np.array(
            [[r["accuracy"] for r in selected if r["seed"] == seed]
             for seed in sorted({r["seed"] for r in selected})]
        )
        place_values.append(mean_and_sem(matrix)[0])
    lower = max(0.0, np.floor(10 * np.nanmin(place_values)) / 10)

    image = None
    for column, model in enumerate(MODELS):
        place = [r for r in rows if r["model"] == model and r["variable"] == "place"]
        seeds = sorted({r["seed"] for r in place})
        matrix = np.array(
            [[next(r["accuracy"] for r in place if r["seed"] == seed and r["bin"] == label + 1)
              for label in range(experiment.PLACE_BINS)] for seed in seeds]
        )
        mean, _ = mean_and_sem(matrix)
        grid = floor_grid(mean)
        axis = axes[0, column]
        image = axis.imshow(grid, origin="lower", vmin=lower, vmax=1.0, cmap="viridis")
        for y, x in np.argwhere(np.isfinite(grid)):
            axis.text(x, y, f"{100 * grid[y, x]:.0f}", ha="center", va="center",
                      fontsize=8, color="white" if grid[y, x] < (lower + 1) / 2 else "black")
        axis.set_title(f"{MODEL_LABELS[model]}: place bins")
        axis.set_xlabel("Floor x-bin")
        axis.set_ylabel("Floor y-bin")

        surface = [r for r in rows if r["model"] == model and r["variable"] == "surface"]
        surface_matrix = np.array(
            [[next(r["accuracy"] for r in surface if r["seed"] == seed and r["bin"] == label + 1)
              for label in range(experiment.SURFACE_BINS)] for seed in seeds]
        )
        surface_mean, surface_sem = mean_and_sem(surface_matrix)
        x = np.arange(1, experiment.SURFACE_BINS + 1)
        axis = axes[1, column]
        axis.errorbar(x, surface_mean, yerr=surface_sem, color=MODEL_COLORS[model],
                      marker="o", markersize=3.5, linewidth=1.4, capsize=2)
        axis.set(xlabel="True gaze-surface bin", ylabel="Held-out accuracy", ylim=(0, 1.02))
        axis.set_title(f"{MODEL_LABELS[model]}: gaze-surface bins")
        axis.grid(axis="y", alpha=0.25)
    if image is not None:
        fig.colorbar(image, ax=axes[0].tolist(), label="Held-out place accuracy", shrink=0.8)
    fig.suptitle("Accuracy-matched models: performance within each true bin", fontsize=15)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_sic_summary(rows: list[dict], path: Path) -> None:
    metrics = (
        ("place", "Place-significant units (%)"),
        ("surface", "Surface-significant units (%)"),
        ("dual-selective", "Dual-selective units (%)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    seeds = sorted({int(row["seed"]) for row in rows})
    for axis, (metric, ylabel) in zip(axes, metrics):
        key = f"{metric}_percent" if metric == "dual-selective" else None
        for seed in seeds:
            pair = {row["model"]: row for row in rows if row["seed"] == seed}
            if metric == "place":
                values = [pair[m]["place-only_percent"] + pair[m]["dual-selective_percent"] for m in MODELS]
            elif metric == "surface":
                values = [pair[m]["surface-only_percent"] + pair[m]["dual-selective_percent"] for m in MODELS]
            else:
                values = [pair[m][key] for m in MODELS]
            axis.plot((0, 1), values, color="#B7BEC8", linewidth=1.2, zorder=1)
            axis.scatter((0, 1), values, color=[MODEL_COLORS[m] for m in MODELS], s=28, zorder=2)
        for x, model in enumerate(MODELS):
            model_rows = [r for r in rows if r["model"] == model]
            if metric == "place":
                values = np.array([r["place-only_percent"] + r["dual-selective_percent"] for r in model_rows])
            elif metric == "surface":
                values = np.array([r["surface-only_percent"] + r["dual-selective_percent"] for r in model_rows])
            else:
                values = np.array([r[key] for r in model_rows])
            mean = values.mean()
            sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            axis.errorbar(x, mean, yerr=sem, fmt="o", color=MODEL_COLORS[model],
                          markersize=9, capsize=4, zorder=3)
        axis.set_xticks((0, 1), [MODEL_LABELS[m] for m in MODELS])
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("SIC significance relative to untrained recurrent networks")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_sic_scatter(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    for axis, model in zip(axes, MODELS):
        selected = [row for row in rows if row["model"] == model]
        for category in SIC_CATEGORIES:
            members = [row for row in selected if row["sic_category"] == category]
            if not members:
                continue
            axis.scatter(
                [row["place_sic_over_threshold"] for row in members],
                [row["surface_sic_over_threshold"] for row in members],
                s=13,
                alpha=0.42,
                color=CATEGORY_COLORS[category],
                label=category,
                edgecolors="none",
            )
        axis.axvline(1.0, color="black", linestyle="--", linewidth=1)
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set(xlabel="Place SIC / null threshold", ylabel="Surface SIC / null threshold",
                 title=MODEL_LABELS[model])
        axis.grid(alpha=0.15)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Individual hidden-unit selectivity across paired seeds")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_representatives(tuning: dict[str, dict], seed: int, path: Path) -> list[dict]:
    category_order = ("place-only", "surface-only", "dual-selective")
    fig, axes = plt.subplots(4, 3, figsize=(13, 13), constrained_layout=True)
    records: list[dict] = []
    for model_index, model in enumerate(MODELS):
        values = tuning[model]
        representatives = choose_representative_units(
            values["place_sic"], values["surface_sic"], values["categories"]
        )
        for column, category in enumerate(category_order):
            place_axis = axes[2 * model_index, column]
            surface_axis = axes[2 * model_index + 1, column]
            if category not in representatives:
                place_axis.axis("off")
                surface_axis.axis("off")
                place_axis.text(0.5, 0.5, f"No {category} unit", ha="center", va="center")
                continue
            unit = representatives[category]
            records.append(
                {
                    "seed": seed,
                    "model": model,
                    "selection_rule": category,
                    "unit": unit,
                    "place_sic": float(values["place_sic"][unit]),
                    "surface_sic": float(values["surface_sic"][unit]),
                }
            )
            grid = floor_grid(values["place_tuning"][:, unit])
            image = place_axis.imshow(grid, origin="lower", cmap="magma")
            place_axis.set_title(
                f"{MODEL_LABELS[model]} unit {unit}: {category}\nplace SIC={values['place_sic'][unit]:.3f}"
            )
            place_axis.set_xlabel("Floor x-bin")
            place_axis.set_ylabel("Floor y-bin")
            fig.colorbar(image, ax=place_axis, fraction=0.045, pad=0.03)
            surface_axis.bar(
                np.arange(1, experiment.SURFACE_BINS + 1),
                values["surface_tuning"][:, unit],
                color=MODEL_COLORS[model],
            )
            surface_axis.set_title(f"surface SIC={values['surface_sic'][unit]:.3f}")
            surface_axis.set_xlabel("Gaze-surface bin")
            surface_axis.set_ylabel("Mean shifted activity")
    fig.suptitle(
        f"Representative post-tanh hidden units (predefined SIC rules; seed {seed})",
        fontsize=15,
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("tuned_dual_comparison_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("objective_sic_bin_analysis"))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--probe-trajectories", type=int, default=1_024)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--min-visits", type=int, default=50)
    parser.add_argument("--null-networks", type=int, default=20)
    parser.add_argument("--null-percentile", type=float, default=99.0)
    parser.add_argument("--representative-seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.probe_trajectories = 40
        args.inference_batch_size = 40
        args.min_visits = 5
        args.null_networks = 2
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = base.choose_device(args.device)
    bridge = base.JuliaTrialBridge()
    checkpoint_map = frozen.matched_checkpoint_map(args.results_root)
    seeds = sorted(checkpoint_map)
    if args.seeds is not None:
        missing = sorted(set(args.seeds).difference(checkpoint_map))
        if missing:
            raise ValueError(f"Seeds are not accuracy matched: {missing}")
        seeds = sorted(args.seeds)
    if args.quick:
        seeds = seeds[:1]
    if not seeds:
        raise RuntimeError("No validation-accuracy-matched checkpoints were found")
    representative_seed = args.representative_seed or seeds[0]
    if representative_seed not in seeds:
        raise ValueError("representative-seed must be one of the analyzed seeds")

    print(f"Device: {device}", flush=True)
    bin_rows = bin_accuracy_rows(args.results_root, seeds)
    unit_rows: list[dict] = []
    summary_rows: list[dict] = []
    support_rows: list[dict] = []
    representative_tuning = None
    for seed in seeds:
        print(f"\n=== Accuracy-matched SIC: seed {seed} ===", flush=True)
        probe = frozen.generate_probe(bridge, seed, args.probe_trajectories)
        units, summaries, support, tuning = sic_rows_for_seed(
            seed, checkpoint_map[seed], probe, device, args
        )
        unit_rows.extend(units)
        summary_rows.extend(summaries)
        support_rows.extend(support)
        if seed == representative_seed:
            representative_tuning = tuning
        for row in summaries:
            print(
                f"{row['model']:11s}: place significant="
                f"{row['place-only_units'] + row['dual-selective_units']}/256 | "
                f"surface significant={row['surface-only_units'] + row['dual-selective_units']}/256",
                flush=True,
            )

    write_csv(args.output_dir / "per_bin_accuracy.csv", bin_rows)
    write_csv(args.output_dir / "sic_unit_scores.csv", unit_rows)
    write_csv(args.output_dir / "sic_network_summary.csv", summary_rows)
    write_csv(args.output_dir / "sic_probe_bin_support.csv", support_rows)
    plot_bin_accuracy(bin_rows, args.output_dir / "objective_per_bin_accuracy.png")
    plot_sic_summary(summary_rows, args.output_dir / "objective_sic_population.png")
    plot_sic_scatter(unit_rows, args.output_dir / "objective_sic_scatter.png")
    representative_rows = plot_representatives(
        representative_tuning,
        representative_seed,
        args.output_dir / "objective_individual_unit_tuning.png",
    )
    write_csv(args.output_dir / "representative_units.csv", representative_rows)
    metadata = {
        "comparison": "validation-accuracy-matched factorized versus conjunction",
        "seeds": seeds,
        "behavioral_data": "saved held-out test predictions from the original comparison",
        "sic_probe_seed": "3000000 + training seed",
        "probe_trajectories_per_seed": args.probe_trajectories,
        "same_probe_inputs_within_seed": True,
        "hidden_activity": "post-tanh recurrent state; padded timesteps excluded",
        "sic_transform": "each unit shifted by its own minimum over the probe bank",
        "sic_min_visits": args.min_visits,
        "sic_null": "pooled units from untrained RNNs with the same recurrent architecture on identical probe inputs",
        "null_networks_per_seed": args.null_networks,
        "null_percentile": args.null_percentile,
        "important_interpretation": "dual-selective SIC means information about both variables; it does not establish nonlinear interaction dependence",
        "representative_unit_rule": "highest relevant SIC among units in each place-only, surface-only or dual-selective category",
        "representative_seed": representative_seed,
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nResults written to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

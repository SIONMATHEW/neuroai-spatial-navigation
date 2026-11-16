"""Per-bin performance and SIC for frozen open/occluded factorized RNNs.

Open and occluded models are evaluated independently in their own label spaces:
25/8 place/surface bins for the open maze and 21/24 for the occluded maze.
The analysis therefore describes where each model succeeds; it does not pair
unrelated bin identifiers across environments.

Test accuracy uses independent noisy test trajectories.  SIC uses a separate
noisy probe bank and a 99th-percentile untrained-network null computed within
each environment, which accounts for its different geometry and label space.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the Julia-backed trial library before torch.
import train_environments as experiment
import evaluate_environments as existing

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


ENVIRONMENTS = experiment.ENVIRONMENTS
ENVIRONMENT_LABELS = {"open": "Open", "occluded": "Occluded"}
ENVIRONMENT_COLORS = {"open": "#3976AF", "occluded": "#D95F55"}
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


def place_grid(values: np.ndarray, environment: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if environment == "open":
        if len(values) != 25:
            raise ValueError("The open environment requires 25 place bins")
        return values.reshape(5, 5).T
    if environment == "occluded":
        if len(values) != 21:
            raise ValueError("The occluded environment requires 21 place bins")
        grid = np.full((5, 5), np.nan)
        pillars = {(1, 1), (3, 1), (1, 3), (3, 3)}
        label = 0
        for x_bin in range(5):
            for y_bin in range(5):
                if (x_bin, y_bin) not in pillars:
                    grid[y_bin, x_bin] = values[label]
                    label += 1
        return grid
    raise ValueError(environment)


@torch.inference_mode()
def predictions(model, batch: experiment.Batch, device: torch.device) -> dict[str, np.ndarray]:
    place_logits, surface_logits = model(batch.inputs.to(device))
    valid = batch.valid.to(device)
    return {
        "true_place": batch.place.to(device)[valid].cpu().numpy(),
        "true_surface": batch.surface.to(device)[valid].cpu().numpy(),
        "pred_place": place_logits[valid].argmax(-1).cpu().numpy(),
        "pred_surface": surface_logits[valid].argmax(-1).cpu().numpy(),
    }


def per_bin_rows(seed: int, environment: str, arrays: dict[str, np.ndarray]) -> list[dict]:
    rows: list[dict] = []
    counts = experiment.LABEL_COUNTS[environment]
    for variable in ("place", "surface"):
        correct, total, accuracy = per_bin_accuracy(
            arrays[f"true_{variable}"], arrays[f"pred_{variable}"], counts[variable]
        )
        for label in range(counts[variable]):
            rows.append(
                {
                    "seed": seed,
                    "environment": environment,
                    "variable": variable,
                    "bin": label + 1,
                    "correct": int(correct[label]),
                    "total": int(total[label]),
                    "accuracy": float(accuracy[label]),
                }
            )
    return rows


def null_model(environment: str, seed: int, device: torch.device):
    counts = experiment.LABEL_COUNTS[environment]
    torch.manual_seed(seed)
    core = torch.nn.RNN(
        experiment.INPUT_DIM,
        experiment.HIDDEN_UNITS,
        nonlinearity="tanh",
        batch_first=True,
    )
    return experiment.FactorizedRNN(core, counts["place"], counts["surface"]).to(device).eval()


def pooled_null_scores(
    environment: str,
    probe: experiment.Batch,
    device: torch.device,
    number_of_networks: int,
    seed: int,
    min_visits: int,
) -> tuple[np.ndarray, np.ndarray]:
    place_scores, surface_scores = [], []
    for index in range(number_of_networks):
        model = null_model(
            environment, 91_000_000 + 10_000 * seed + index, device
        )
        data = existing.extract_hidden(model, probe, device)
        shifted = min_shift(data["hidden"])
        place_scores.append(sic_score(shifted, data["place"], min_visits))
        surface_scores.append(sic_score(shifted, data["surface"], min_visits))
        del model, data, shifted
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(place_scores), np.concatenate(surface_scores)


def analyze_sic(
    seed: int,
    environment: str,
    model,
    payload: dict,
    probe: experiment.Batch,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict], dict, list[dict], dict]:
    null_place, null_surface = pooled_null_scores(
        environment,
        probe,
        device,
        args.null_networks,
        seed,
        args.min_visits,
    )
    place_threshold = null_threshold(null_place, args.null_percentile)
    surface_threshold = null_threshold(null_surface, args.null_percentile)
    data = existing.extract_hidden(model, probe, device)
    shifted = min_shift(data["hidden"])
    place_sic = sic_score(shifted, data["place"], args.min_visits)
    surface_sic = sic_score(shifted, data["surface"], args.min_visits)
    categories = classify_units(
        place_sic, surface_sic, place_threshold, surface_threshold
    )
    counts = Counter(categories)

    unit_rows = []
    for unit, category in enumerate(categories):
        unit_rows.append(
            {
                "seed": seed,
                "environment": environment,
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

    label_counts = experiment.LABEL_COUNTS[environment]
    summary = {
        "seed": seed,
        "environment": environment,
        "checkpoint": args.checkpoint,
        "checkpoint_update": int(payload["update"]),
        "probe_trajectories": int(probe.inputs.shape[0]),
        "probe_valid_timesteps": int(len(shifted)),
        "place_bins_meeting_min_visits": int(
            np.sum(np.bincount(data["place"], minlength=label_counts["place"]) >= args.min_visits)
        ),
        "surface_bins_meeting_min_visits": int(
            np.sum(np.bincount(data["surface"], minlength=label_counts["surface"]) >= args.min_visits)
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
    place_tuning, place_counts = mean_activity_by_bin(
        shifted, data["place"], label_counts["place"]
    )
    surface_tuning, surface_counts = mean_activity_by_bin(
        shifted, data["surface"], label_counts["surface"]
    )
    tuning = {
        "place_sic": place_sic,
        "surface_sic": surface_sic,
        "categories": categories,
        "place_tuning": place_tuning,
        "surface_tuning": surface_tuning,
        "place_counts": place_counts,
        "surface_counts": surface_counts,
    }
    support_rows = []
    for variable, counts_for_variable in (
        ("place", place_counts),
        ("surface", surface_counts),
    ):
        support_rows.extend(
            {
                "seed": seed,
                "environment": environment,
                "variable": variable,
                "bin": label + 1,
                "valid_timesteps": int(counts_for_variable[label]),
                "included_in_sic": int(counts_for_variable[label] >= args.min_visits),
            }
            for label in range(len(counts_for_variable))
        )
    return unit_rows, summary, support_rows, tuning


def plot_bin_accuracy(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    place_means = {}
    for environment in ENVIRONMENTS:
        subset = [r for r in rows if r["environment"] == environment and r["variable"] == "place"]
        seeds = sorted({r["seed"] for r in subset})
        count = experiment.LABEL_COUNTS[environment]["place"]
        matrix = np.array(
            [[next(r["accuracy"] for r in subset if r["seed"] == seed and r["bin"] == label + 1)
              for label in range(count)] for seed in seeds]
        )
        place_means[environment] = mean_and_sem(matrix)[0]
    lower = max(0.0, np.floor(10 * min(np.nanmin(v) for v in place_means.values())) / 10)

    image = None
    for column, environment in enumerate(ENVIRONMENTS):
        grid = place_grid(place_means[environment], environment)
        axis = axes[0, column]
        image = axis.imshow(grid, origin="lower", cmap="viridis", vmin=lower, vmax=1.0)
        for y, x in np.argwhere(np.isfinite(grid)):
            axis.text(x, y, f"{100 * grid[y, x]:.0f}", ha="center", va="center",
                      fontsize=8, color="white" if grid[y, x] < (lower + 1) / 2 else "black")
        axis.set_title(
            f"{ENVIRONMENT_LABELS[environment]}: "
            f"{experiment.LABEL_COUNTS[environment]['place']} place bins"
        )
        axis.set_xlabel("Floor x-bin")
        axis.set_ylabel("Floor y-bin")

        subset = [r for r in rows if r["environment"] == environment and r["variable"] == "surface"]
        seeds = sorted({r["seed"] for r in subset})
        count = experiment.LABEL_COUNTS[environment]["surface"]
        matrix = np.array(
            [[next(r["accuracy"] for r in subset if r["seed"] == seed and r["bin"] == label + 1)
              for label in range(count)] for seed in seeds]
        )
        mean, sem = mean_and_sem(matrix)
        x = np.arange(1, count + 1)
        axis = axes[1, column]
        axis.errorbar(x, mean, yerr=sem, color=ENVIRONMENT_COLORS[environment],
                      marker="o", markersize=4, linewidth=1.4, capsize=2)
        axis.set(
            xlabel="True gaze-surface bin",
            ylabel="Held-out accuracy",
            ylim=(0, 1.02),
            title=f"{ENVIRONMENT_LABELS[environment]}: {count} gaze-surface bins",
        )
        axis.grid(axis="y", alpha=0.25)
    if image is not None:
        fig.colorbar(image, ax=axes[0].tolist(), label="Held-out place accuracy", shrink=0.8)
    fig.suptitle("Factorized RNN performance within each environment's true bins", fontsize=15)
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
        for seed in seeds:
            pair = {r["environment"]: r for r in rows if r["seed"] == seed}
            if metric == "place":
                values = [pair[e]["place-only_percent"] + pair[e]["dual-selective_percent"] for e in ENVIRONMENTS]
            elif metric == "surface":
                values = [pair[e]["surface-only_percent"] + pair[e]["dual-selective_percent"] for e in ENVIRONMENTS]
            else:
                values = [pair[e]["dual-selective_percent"] for e in ENVIRONMENTS]
            axis.plot((0, 1), values, color="#B7BEC8", linewidth=1.2)
            axis.scatter((0, 1), values, color=[ENVIRONMENT_COLORS[e] for e in ENVIRONMENTS], s=28)
        for x, environment in enumerate(ENVIRONMENTS):
            selected = [r for r in rows if r["environment"] == environment]
            if metric == "place":
                values = np.array([r["place-only_percent"] + r["dual-selective_percent"] for r in selected])
            elif metric == "surface":
                values = np.array([r["surface-only_percent"] + r["dual-selective_percent"] for r in selected])
            else:
                values = np.array([r["dual-selective_percent"] for r in selected])
            mean = values.mean()
            sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            axis.errorbar(x, mean, yerr=sem, fmt="o", color=ENVIRONMENT_COLORS[environment],
                          markersize=9, capsize=4)
        axis.set_xticks((0, 1), [ENVIRONMENT_LABELS[e] for e in ENVIRONMENTS])
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Environment-specific SIC significance relative to untrained networks")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_sic_scatter(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    for axis, environment in zip(axes, ENVIRONMENTS):
        selected = [row for row in rows if row["environment"] == environment]
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
        axis.set(
            xlabel="Place SIC / environment-specific null threshold",
            ylabel="Surface SIC / environment-specific null threshold",
            title=ENVIRONMENT_LABELS[environment],
        )
        axis.grid(alpha=0.15)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Individual hidden-unit selectivity across paired seeds")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_representatives(tuning: dict[str, dict], seed: int, path: Path) -> list[dict]:
    categories = ("place-only", "surface-only", "dual-selective")
    fig, axes = plt.subplots(4, 3, figsize=(13, 13), constrained_layout=True)
    records: list[dict] = []
    for environment_index, environment in enumerate(ENVIRONMENTS):
        values = tuning[environment]
        selected = choose_representative_units(
            values["place_sic"], values["surface_sic"], values["categories"]
        )
        for column, category in enumerate(categories):
            place_axis = axes[2 * environment_index, column]
            surface_axis = axes[2 * environment_index + 1, column]
            if category not in selected:
                place_axis.axis("off")
                surface_axis.axis("off")
                place_axis.text(0.5, 0.5, f"No {category} unit", ha="center", va="center")
                continue
            unit = selected[category]
            records.append(
                {
                    "seed": seed,
                    "environment": environment,
                    "selection_rule": category,
                    "unit": unit,
                    "place_sic": float(values["place_sic"][unit]),
                    "surface_sic": float(values["surface_sic"][unit]),
                }
            )
            grid = place_grid(values["place_tuning"][:, unit], environment)
            image = place_axis.imshow(grid, origin="lower", cmap="magma")
            place_axis.set_title(
                f"{ENVIRONMENT_LABELS[environment]} unit {unit}: {category}\n"
                f"place SIC={values['place_sic'][unit]:.3f}"
            )
            place_axis.set_xlabel("Floor x-bin")
            place_axis.set_ylabel("Floor y-bin")
            fig.colorbar(image, ax=place_axis, fraction=0.045, pad=0.03)
            count = experiment.LABEL_COUNTS[environment]["surface"]
            surface_axis.bar(
                np.arange(1, count + 1),
                values["surface_tuning"][:, unit],
                color=ENVIRONMENT_COLORS[environment],
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
    parser.add_argument("--results-root", type=Path, default=Path("environment_results_full"))
    parser.add_argument("--output-dir", type=Path, default=Path("environment_sic_bin_analysis"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--checkpoint", choices=("final", "best"), default="final")
    parser.add_argument("--test-trials", type=int, default=4_096)
    parser.add_argument("--probe-trials", type=int, default=1_024)
    parser.add_argument("--noise-sd", type=float, default=0.25)
    parser.add_argument("--min-visits", type=int, default=50)
    parser.add_argument("--null-networks", type=int, default=20)
    parser.add_argument("--null-percentile", type=float, default=99.0)
    parser.add_argument("--representative-seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds = args.seeds[:1]
        args.test_trials = 32
        args.probe_trials = 40
        args.min_visits = 5
        args.null_networks = 2
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = experiment.choose_device(args.device)
    representative_seed = args.representative_seed or args.seeds[0]
    if representative_seed not in args.seeds:
        raise ValueError("representative-seed must be one of the analyzed seeds")

    print(f"Device: {device}", flush=True)
    bin_rows: list[dict] = []
    unit_rows: list[dict] = []
    summary_rows: list[dict] = []
    support_rows: list[dict] = []
    representative_tuning: dict[str, dict] = {}
    for seed in args.seeds:
        print(f"\n=== Environment SIC: seed {seed} ===", flush=True)
        seed_root = args.results_root / f"seed_{seed}"
        for environment in ENVIRONMENTS:
            environment_root = seed_root / environment
            statistics = existing.load_statistics(environment_root / "noise_statistics.npz")
            model, payload = existing.load_model(
                environment_root / f"{args.checkpoint}.pt", device
            )
            trialstruct = experiment.make_trialstruct(environment)

            clean_test = experiment.convert_batch(
                experiment.make_generator(
                    trialstruct, args.test_trials, 4_000_000 + seed
                )(),
                environment,
            )
            noisy_test = experiment.add_sensory_noise(
                clean_test,
                statistics,
                args.noise_sd,
                torch.Generator().manual_seed(5_000_000 + seed),
            )
            bin_rows.extend(per_bin_rows(seed, environment, predictions(model, noisy_test, device)))

            clean_probe = experiment.convert_batch(
                experiment.make_generator(
                    trialstruct, args.probe_trials, 6_000_000 + seed
                )(),
                environment,
            )
            noisy_probe = experiment.add_sensory_noise(
                clean_probe,
                statistics,
                args.noise_sd,
                torch.Generator().manual_seed(7_000_000 + seed),
            )
            units, summary, support, tuning = analyze_sic(
                seed, environment, model, payload, noisy_probe, device, args
            )
            unit_rows.extend(units)
            summary_rows.append(summary)
            support_rows.extend(support)
            if seed == representative_seed:
                representative_tuning[environment] = tuning
            print(
                f"{environment:8s}: place significant="
                f"{summary['place-only_units'] + summary['dual-selective_units']}/256 | "
                f"surface significant={summary['surface-only_units'] + summary['dual-selective_units']}/256",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(args.output_dir / "per_bin_accuracy.csv", bin_rows)
    write_csv(args.output_dir / "sic_unit_scores.csv", unit_rows)
    write_csv(args.output_dir / "sic_network_summary.csv", summary_rows)
    write_csv(args.output_dir / "sic_probe_bin_support.csv", support_rows)
    plot_bin_accuracy(bin_rows, args.output_dir / "environment_per_bin_accuracy.png")
    plot_sic_summary(summary_rows, args.output_dir / "environment_sic_population.png")
    plot_sic_scatter(unit_rows, args.output_dir / "environment_sic_scatter.png")
    representative_rows = plot_representatives(
        representative_tuning,
        representative_seed,
        args.output_dir / "environment_individual_unit_tuning.png",
    )
    write_csv(args.output_dir / "representative_units.csv", representative_rows)
    metadata = {
        "comparison": "paired factorized RNNs in open and occluded environments",
        "checkpoint": args.checkpoint,
        "seeds": args.seeds,
        "training_noise_sd": 0.2,
        "test_and_probe_noise_sd": args.noise_sd,
        "test_trajectory_seed": "4000000 + training seed",
        "test_noise_seed": "5000000 + training seed",
        "probe_trajectory_seed": "6000000 + training seed",
        "probe_noise_seed": "7000000 + training seed",
        "test_trials_per_seed_and_environment": args.test_trials,
        "probe_trajectories_per_seed_and_environment": args.probe_trials,
        "hidden_activity": "post-tanh recurrent state; padded timesteps excluded",
        "sic_transform": "each unit shifted by its own minimum over the probe bank",
        "sic_min_visits": args.min_visits,
        "sic_null": "separate environment-specific pooled null from untrained RNNs on identical probe inputs",
        "null_networks_per_seed_and_environment": args.null_networks,
        "null_percentile": args.null_percentile,
        "bin_comparison_policy": "open and occluded bins are reported independently because the label spaces differ",
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

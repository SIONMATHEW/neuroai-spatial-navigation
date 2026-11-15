"""Evaluate open and occluded factorized RNNs on independent trajectories.

The script reports held-out place, gaze-surface and exact-joint accuracy.  It
then analyzes post-tanh hidden activity with four categorical encoding models:
place, gaze, additive place+gaze, and full place-by-gaze interaction.

All model fitting uses probe-fit trajectories, ridge selection uses separate
probe-validation trajectories, and reported R-squared values use untouched
probe-test trajectories.
"""

from __future__ import annotations

import train_environments as experiment

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
CATEGORIES = (
    "place-dominant",
    "gaze-dominant",
    "additive-mixed",
    "interaction-dependent",
    "poorly-explained",
    "unclassified",
)
ENVIRONMENT_COLORS = {"open": "#3B82B8", "occluded": "#DD5A50"}
CATEGORY_COLORS = {
    "place-dominant": "#4C78A8",
    "gaze-dominant": "#F2A541",
    "additive-mixed": "#59A14F",
    "interaction-dependent": "#B34E8C",
    "poorly-explained": "#9C9C9C",
    "unclassified": "#D6D6D6",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_statistics(path: Path) -> dict[str, torch.Tensor]:
    stored = np.load(path)
    return {key: torch.from_numpy(stored[key]) for key in ("mean", "std", "min", "max")}


def load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    core = torch.nn.RNN(
        experiment.INPUT_DIM,
        experiment.HIDDEN_UNITS,
        nonlinearity="tanh",
        batch_first=True,
    )
    model = experiment.FactorizedRNN(
        core, payload["place_bins"], payload["surface_bins"]
    )
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval(), payload


@torch.inference_mode()
def behavioral_metrics(model, batch, surface_bins: int, device) -> dict:
    place_logits, surface_logits = model(batch.inputs.to(device))
    valid = batch.valid.to(device)
    true_place = batch.place.to(device)[valid]
    true_surface = batch.surface.to(device)[valid]
    true_joint = batch.joint.to(device)[valid]
    pred_place = place_logits[valid].argmax(-1)
    pred_surface = surface_logits[valid].argmax(-1)
    pred_joint = pred_place * surface_bins + pred_surface
    return {
        "valid_timesteps": int(valid.sum()),
        "place_accuracy": float((pred_place == true_place).float().mean()),
        "surface_accuracy": float((pred_surface == true_surface).float().mean()),
        "joint_accuracy": float((pred_joint == true_joint).float().mean()),
    }


@torch.inference_mode()
def extract_hidden(model, batch, device) -> dict[str, np.ndarray]:
    hidden, _ = model.rnn(batch.inputs.to(device))
    valid = batch.valid.to(device)
    n_trials, n_steps = batch.valid.shape
    trajectory = np.broadcast_to(np.arange(n_trials)[:, None], (n_trials, n_steps))
    return {
        "hidden": hidden[valid].cpu().numpy().astype(np.float64),
        "place": batch.place[batch.valid].numpy(),
        "surface": batch.surface[batch.valid].numpy(),
        "trajectory": trajectory[batch.valid.numpy()],
    }


def one_hot(labels: np.ndarray, levels: int) -> np.ndarray:
    return np.eye(levels, dtype=np.float64)[labels]


def feature_matrices(place, surface, place_bins: int, surface_bins: int):
    place_design = one_hot(place, place_bins)
    surface_design = one_hot(surface, surface_bins)
    joint = place * surface_bins + surface
    return {
        "place": place_design,
        "gaze": surface_design,
        "additive": np.concatenate((place_design, surface_design), axis=1),
        "interaction": one_hot(joint, place_bins * surface_bins),
    }


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    residual = np.square(actual - predicted).sum(axis=0)
    centered = actual - actual.mean(axis=0, keepdims=True)
    total = np.square(centered).sum(axis=0)
    return 1.0 - residual / np.maximum(total, 1e-12)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    """Fit ridge coefficients while leaving the intercept unpenalized."""

    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xc = x - x_mean
    yc = y - y_mean
    coefficients = np.linalg.solve(
        xc.T @ xc + alpha * np.eye(x.shape[1]),
        xc.T @ yc,
    )
    return {"x_mean": x_mean, "y_mean": y_mean, "coef": coefficients}


def predict_ridge(fit: dict, x: np.ndarray) -> np.ndarray:
    return (x - fit["x_mean"]) @ fit["coef"] + fit["y_mean"]


def select_alpha(x_fit, y_fit, x_validation, y_validation) -> float:
    candidates = []
    for alpha in RIDGE_GRID:
        fit = fit_ridge(x_fit, y_fit, alpha)
        score = np.median(r2_score(y_validation, predict_ridge(fit, x_validation)))
        candidates.append((score, -alpha, alpha))
    return max(candidates)[2]


def split_trajectories(n_trials: int, seed: int) -> dict[str, np.ndarray]:
    shuffled = np.random.default_rng(seed).permutation(n_trials)
    n_fit = round(0.70 * n_trials)
    n_validation = round(0.15 * n_trials)
    return {
        "fit": shuffled[:n_fit],
        "validation": shuffled[n_fit : n_fit + n_validation],
        "test": shuffled[n_fit + n_validation :],
    }


def split_masks(trajectory: np.ndarray, split: dict) -> dict[str, np.ndarray]:
    return {name: np.isin(trajectory, values) for name, values in split.items()}


def bootstrap_bounds(actual, predictions, trajectory, draws: int, seed: int):
    """Bootstrap complete held-out trajectories, never individual timesteps."""

    trial_ids = np.unique(trajectory)
    n_units = actual.shape[1]
    counts = np.zeros(len(trial_ids), dtype=np.int64)
    sums = np.zeros((len(trial_ids), n_units))
    sums_squared = np.zeros_like(sums)
    errors = {name: np.zeros_like(sums) for name in predictions}
    for index, trial_id in enumerate(trial_ids):
        keep = trajectory == trial_id
        values = actual[keep]
        counts[index] = len(values)
        sums[index] = values.sum(axis=0)
        sums_squared[index] = np.square(values).sum(axis=0)
        for name, prediction in predictions.items():
            errors[name][index] = np.square(values - prediction[keep]).sum(axis=0)

    rng = np.random.default_rng(seed)
    samples = {"unique_place": [], "unique_gaze": [], "interaction": []}
    for _ in range(draws):
        chosen = rng.integers(0, len(trial_ids), size=len(trial_ids))
        total_n = counts[chosen].sum()
        total_sum = sums[chosen].sum(axis=0)
        total_sum_squared = sums_squared[chosen].sum(axis=0)
        total_variance = np.maximum(
            total_sum_squared - total_sum**2 / total_n, 1e-12
        )
        score = {
            name: 1.0 - error[chosen].sum(axis=0) / total_variance
            for name, error in errors.items()
        }
        samples["unique_place"].append(score["additive"] - score["gaze"])
        samples["unique_gaze"].append(score["additive"] - score["place"])
        samples["interaction"].append(score["interaction"] - score["additive"])
    return {
        name: (
            np.quantile(np.asarray(values), 0.025, axis=0),
            np.quantile(np.asarray(values), 0.975, axis=0),
        )
        for name, values in samples.items()
    }


def classify(scores: dict, bounds: dict, threshold: float = 0.01) -> list[str]:
    lower_place, upper_place = bounds["unique_place"]
    lower_gaze, upper_gaze = bounds["unique_gaze"]
    lower_interaction, upper_interaction = bounds["interaction"]
    categories = []
    for unit in range(experiment.HIDDEN_UNITS):
        if scores["interaction"][unit] >= threshold and lower_interaction[unit] > threshold:
            category = "interaction-dependent"
        elif (
            scores["additive"][unit] >= threshold
            and lower_place[unit] > threshold
            and upper_gaze[unit] <= threshold
            and upper_interaction[unit] <= threshold
        ):
            category = "place-dominant"
        elif (
            scores["additive"][unit] >= threshold
            and lower_gaze[unit] > threshold
            and upper_place[unit] <= threshold
            and upper_interaction[unit] <= threshold
        ):
            category = "gaze-dominant"
        elif (
            scores["additive"][unit] >= threshold
            and lower_place[unit] > threshold
            and lower_gaze[unit] > threshold
            and upper_interaction[unit] <= threshold
        ):
            category = "additive-mixed"
        elif max(scores["additive"][unit], scores["interaction"][unit]) < threshold:
            category = "poorly-explained"
        else:
            category = "unclassified"
        categories.append(category)
    return categories


def support_audit(place, surface, masks, surface_bins: int) -> dict:
    joint = place * surface_bins + surface
    fit_support = set(joint[masks["fit"] | masks["validation"]].tolist())
    test_values = joint[masks["test"]]

    adjacency = {}
    for label in fit_support:
        p, g = divmod(label, surface_bins)
        p_node, g_node = ("p", p), ("g", g)
        adjacency.setdefault(p_node, set()).add(g_node)
        adjacency.setdefault(g_node, set()).add(p_node)
    remaining = set(adjacency)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            neighbours = adjacency[stack.pop()] & remaining
            remaining -= neighbours
            stack.extend(neighbours)
    return {
        "fit_supported_joint_classes": len(fit_support),
        "test_observed_joint_classes": len(set(test_values.tolist())),
        "test_timesteps_supported_in_fit_fraction": float(
            np.mean(np.isin(test_values, list(fit_support)))
        ),
        "fit_support_graph_components": components,
    }


def effect_basis(effects: np.ndarray, energy=0.90, rank_cap=10):
    centered = effects - effects.mean(axis=0, keepdims=True)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    power = singular_values**2
    if power.sum() <= 1e-12:
        return np.zeros((effects.shape[1], 0)), 0
    rank = int(np.searchsorted(np.cumsum(power) / power.sum(), energy) + 1)
    rank = min(rank, rank_cap, right.shape[0])
    return right[:rank].T, rank


def subspace_metrics(place_effects, gaze_effects) -> dict:
    place_basis, place_rank = effect_basis(place_effects)
    gaze_basis, gaze_rank = effect_basis(gaze_effects)
    if min(place_rank, gaze_rank) == 0:
        return {
            "minimum_angle_degrees": np.nan,
            "alignment_index": np.nan,
            "place_rank": place_rank,
            "gaze_rank": gaze_rank,
        }
    singular = np.clip(
        np.linalg.svd(place_basis.T @ gaze_basis, compute_uv=False), 0.0, 1.0
    )
    return {
        "minimum_angle_degrees": float(np.degrees(np.arccos(singular.max()))),
        "alignment_index": float(np.square(singular).mean()),
        "place_rank": place_rank,
        "gaze_rank": gaze_rank,
    }


def random_baseline(place_rank: int, gaze_rank: int, seed: int, draws=200):
    if min(place_rank, gaze_rank) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    angles, alignments = [], []
    for _ in range(draws):
        place, _ = np.linalg.qr(rng.normal(size=(experiment.HIDDEN_UNITS, place_rank)))
        gaze, _ = np.linalg.qr(rng.normal(size=(experiment.HIDDEN_UNITS, gaze_rank)))
        singular = np.linalg.svd(place.T @ gaze, compute_uv=False)
        angles.append(np.degrees(np.arccos(np.clip(singular.max(), 0.0, 1.0))))
        alignments.append(np.square(singular).mean())
    return float(np.median(angles)), float(np.median(alignments))


def analyze_hidden(
    seed: int,
    environment: str,
    model,
    payload: dict,
    probe,
    split: dict,
    device,
    bootstrap_draws: int,
):
    data = extract_hidden(model, probe, device)
    place_bins = payload["place_bins"]
    surface_bins = payload["surface_bins"]
    designs = feature_matrices(data["place"], data["surface"], place_bins, surface_bins)
    masks = split_masks(data["trajectory"], split)

    fits, predictions, alphas = {}, {}, {}
    for name, design in designs.items():
        alpha = select_alpha(
            design[masks["fit"]], data["hidden"][masks["fit"]],
            design[masks["validation"]], data["hidden"][masks["validation"]],
        )
        train_mask = masks["fit"] | masks["validation"]
        fits[name] = fit_ridge(design[train_mask], data["hidden"][train_mask], alpha)
        predictions[name] = predict_ridge(fits[name], design[masks["test"]])
        alphas[name] = alpha

    actual = data["hidden"][masks["test"]]
    scores = {
        name: r2_score(actual, prediction) for name, prediction in predictions.items()
    }
    scores["unique_place"] = scores["additive"] - scores["gaze"]
    scores["unique_gaze"] = scores["additive"] - scores["place"]
    scores["interaction_gain"] = scores["interaction"] - scores["additive"]
    bounds = bootstrap_bounds(
        actual,
        predictions,
        data["trajectory"][masks["test"]],
        bootstrap_draws,
        9_000_000 + 100 * seed + (0 if environment == "open" else 1),
    )
    categories = classify(scores, bounds)

    unit_rows = []
    for unit, category in enumerate(categories):
        unit_rows.append({
            "seed": seed,
            "environment": environment,
            "checkpoint_update": int(payload["update"]),
            "unit": unit,
            "category": category,
            "r2_place": scores["place"][unit],
            "r2_gaze": scores["gaze"][unit],
            "r2_additive": scores["additive"][unit],
            "r2_interaction": scores["interaction"][unit],
            "unique_place_gain": scores["unique_place"][unit],
            "unique_place_ci_low": bounds["unique_place"][0][unit],
            "unique_place_ci_high": bounds["unique_place"][1][unit],
            "unique_gaze_gain": scores["unique_gaze"][unit],
            "unique_gaze_ci_low": bounds["unique_gaze"][0][unit],
            "unique_gaze_ci_high": bounds["unique_gaze"][1][unit],
            "interaction_gain": scores["interaction_gain"][unit],
            "interaction_ci_low": bounds["interaction"][0][unit],
            "interaction_ci_high": bounds["interaction"][1][unit],
        })

    counts = Counter(categories)
    summary = {
        "seed": seed,
        "environment": environment,
        "checkpoint_update": int(payload["update"]),
        "probe_fit_trajectories": len(split["fit"]),
        "probe_validation_trajectories": len(split["validation"]),
        "probe_test_trajectories": len(split["test"]),
        "median_additive_r2": float(np.median(scores["additive"])),
        "median_interaction_r2": float(np.median(scores["interaction"])),
        "median_interaction_gain": float(np.median(scores["interaction_gain"])),
        **{f"{name}_units": counts[name] for name in CATEGORIES},
        **{f"ridge_alpha_{name}": value for name, value in alphas.items()},
        **support_audit(data["place"], data["surface"], masks, surface_bins),
    }

    additive_coefficients = fits["additive"]["coef"]
    subspace = subspace_metrics(
        additive_coefficients[:place_bins], additive_coefficients[place_bins:]
    )
    random_angle, random_alignment = random_baseline(
        subspace["place_rank"],
        subspace["gaze_rank"],
        10_000_000 + 100 * seed + (0 if environment == "open" else 1),
    )
    subspace.update({
        "seed": seed,
        "environment": environment,
        "checkpoint_update": int(payload["update"]),
        "random_median_minimum_angle_degrees": random_angle,
        "random_median_alignment_index": random_alignment,
    })
    return unit_rows, summary, subspace


def synthetic_recovery_check(output_directory: Path) -> dict:
    """Verify that the pipeline distinguishes known additive/interaction units."""

    rng = np.random.default_rng(12345)
    n = 8_000
    place = rng.integers(0, 25, size=n)
    gaze = rng.integers(0, 8, size=n)
    designs = feature_matrices(place, gaze, 25, 8)
    additive_activity = designs["additive"] @ rng.normal(size=(33, 8))
    interaction_activity = designs["interaction"] @ rng.normal(size=(200, 8))
    activity = np.concatenate((additive_activity, interaction_activity), axis=1)
    activity += rng.normal(scale=0.2, size=activity.shape)
    order = rng.permutation(n)
    fit_indices, test_indices = order[:6_000], order[6_000:]
    additive_fit = fit_ridge(designs["additive"][fit_indices], activity[fit_indices], 1.0)
    interaction_fit = fit_ridge(
        designs["interaction"][fit_indices], activity[fit_indices], 1.0
    )
    additive_r2 = r2_score(
        activity[test_indices], predict_ridge(additive_fit, designs["additive"][test_indices])
    )
    interaction_r2 = r2_score(
        activity[test_indices],
        predict_ridge(interaction_fit, designs["interaction"][test_indices]),
    )
    gain = interaction_r2 - additive_r2
    result = {
        "median_gain_known_additive_units": float(np.median(gain[:8])),
        "median_gain_known_interaction_units": float(np.median(gain[8:])),
        "passed": bool(np.median(gain[8:]) > np.median(gain[:8]) + 0.05),
    }
    write_json(output_directory / "synthetic_recovery_check.json", result)
    if not result["passed"]:
        raise RuntimeError("Synthetic interaction-recovery check failed")
    return result


def mean_sem(values: np.ndarray):
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def paired_plot(rows, value_key: str, ylabel: str, title: str, path: Path):
    fig, axis = plt.subplots(figsize=(5.2, 4.5))
    for seed in sorted({row["seed"] for row in rows}):
        pair = {row["environment"]: row for row in rows if row["seed"] == seed}
        axis.plot((0, 1), (pair["open"][value_key], pair["occluded"][value_key]),
                  color="#AEB6C2", linewidth=1.5)
    for x, environment in enumerate(experiment.ENVIRONMENTS):
        values = np.array([row[value_key] for row in rows if row["environment"] == environment])
        mean, sem = mean_sem(values)
        axis.scatter(np.full(len(values), x), values, color=ENVIRONMENT_COLORS[environment],
                     alpha=0.55, s=35)
        axis.errorbar(x, mean, yerr=sem, fmt="o", markersize=11,
                      color=ENVIRONMENT_COLORS[environment], capsize=4)
    axis.set_xticks((0, 1), ("Open", "Occluded"))
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_behavior(rows: list[dict], output_directory: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    metrics = (
        ("place_accuracy", "Place bin"),
        ("surface_accuracy", "Gaze surface"),
        ("joint_accuracy", "Exact conjunction"),
    )
    for axis, (metric, title) in zip(axes, metrics):
        for seed in sorted({row["seed"] for row in rows}):
            pair = {row["environment"]: row for row in rows if row["seed"] == seed}
            axis.plot((0, 1), (pair["open"][metric], pair["occluded"][metric]),
                      color="#AEB6C2", linewidth=1.5)
        for x, environment in enumerate(experiment.ENVIRONMENTS):
            values = np.array([row[metric] for row in rows if row["environment"] == environment])
            mean, sem = mean_sem(values)
            axis.scatter(np.full(len(values), x), values,
                         color=ENVIRONMENT_COLORS[environment], alpha=0.55)
            axis.errorbar(x, mean, yerr=sem, fmt="o", markersize=10,
                          color=ENVIRONMENT_COLORS[environment], capsize=4)
        axis.set_xticks((0, 1), ("Open", "Occluded"))
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Held-out accuracy")
    fig.suptitle("Factorized RNN: environmental comparison")
    fig.tight_layout()
    fig.savefig(output_directory / "behavior_open_vs_occluded.png", dpi=220)
    plt.close(fig)


def plot_categories(rows: list[dict], output_directory: Path):
    fig, axis = plt.subplots(figsize=(6.5, 4.8))
    bottom = np.zeros(2)
    for category in CATEGORIES:
        values = []
        for environment in experiment.ENVIRONMENTS:
            selected = [row for row in rows if row["environment"] == environment]
            values.append(np.mean([
                100 * row[f"{category}_units"] / experiment.HIDDEN_UNITS
                for row in selected
            ]))
        axis.bar((0, 1), values, bottom=bottom, label=category,
                 color=CATEGORY_COLORS[category], edgecolor="white")
        bottom += values
    axis.set_xticks((0, 1), ("Open", "Occluded"))
    axis.set_ylabel("Hidden units (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Held-out hidden-state encoding categories")
    axis.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_directory / "hidden_categories_open_vs_occluded.png", dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("environment_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("environment_evaluation"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--checkpoint", choices=("final", "best"), default="final")
    parser.add_argument("--test-trials", type=int, default=4_096)
    parser.add_argument("--probe-trials", type=int, default=512)
    parser.add_argument("--noise-sd", type=float, default=0.25)
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.seeds = args.seeds[:1]
        args.test_trials = 32
        args.probe_trials = 40
        args.bootstrap_draws = 20
    device = experiment.choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    synthetic = synthetic_recovery_check(args.output_dir)
    behavior_rows, unit_rows, summary_rows, subspace_rows = [], [], [], []

    for seed in args.seeds:
        seed_root = args.results_root / f"seed_{seed}"
        manifest = read_json(seed_root / "manifest.json")
        print(f"\n=== Evaluating seed {seed} ===", flush=True)
        for environment in experiment.ENVIRONMENTS:
            environment_root = seed_root / environment
            statistics = load_statistics(environment_root / "noise_statistics.npz")
            model, payload = load_model(environment_root / f"{args.checkpoint}.pt", device)
            trialstruct = experiment.make_trialstruct(environment)

            test_generator = experiment.make_generator(
                trialstruct, args.test_trials, 4_000_000 + seed
            )
            clean_test = experiment.convert_batch(test_generator(), environment)
            test_noise = torch.Generator().manual_seed(5_000_000 + seed)
            noisy_test = experiment.add_sensory_noise(
                clean_test, statistics, args.noise_sd, test_noise
            )
            metrics = behavioral_metrics(
                model, noisy_test, payload["surface_bins"], device
            )
            behavior_rows.append({
                "seed": seed,
                "environment": environment,
                "checkpoint": args.checkpoint,
                "checkpoint_update": int(payload["update"]),
                "test_trials": args.test_trials,
                "test_noise_sd": args.noise_sd,
                **metrics,
            })

            probe_generator = experiment.make_generator(
                trialstruct, args.probe_trials, 6_000_000 + seed
            )
            clean_probe = experiment.convert_batch(probe_generator(), environment)
            probe_noise = torch.Generator().manual_seed(7_000_000 + seed)
            noisy_probe = experiment.add_sensory_noise(
                clean_probe, statistics, args.noise_sd, probe_noise
            )
            split = split_trajectories(args.probe_trials, 8_000_000 + seed)
            units, summary, subspace = analyze_hidden(
                seed, environment, model, payload, noisy_probe, split,
                device, args.bootstrap_draws,
            )
            unit_rows.extend(units)
            summary_rows.append(summary)
            subspace_rows.append(subspace)
            print(
                f"{environment:8s}: joint={100 * metrics['joint_accuracy']:.2f}% | "
                f"interaction units={summary['interaction-dependent_units']}",
                flush=True,
            )

    write_csv(args.output_dir / "behavior_metrics.csv", behavior_rows)
    write_csv(args.output_dir / "unit_scores.csv", unit_rows)
    write_csv(args.output_dir / "network_summary.csv", summary_rows)
    write_csv(args.output_dir / "subspace_metrics.csv", subspace_rows)
    plot_behavior(behavior_rows, args.output_dir)
    plot_categories(summary_rows, args.output_dir)
    paired_plot(
        summary_rows, "median_interaction_gain", "Median held-out interaction gain",
        "Interaction dependence", args.output_dir / "interaction_gain_open_vs_occluded.png",
    )
    paired_plot(
        subspace_rows, "minimum_angle_degrees", "Minimum principal angle (degrees)",
        "Position-gaze subspace angle", args.output_dir / "principal_angle_open_vs_occluded.png",
    )
    paired_plot(
        subspace_rows, "alignment_index", "Position-gaze alignment index",
        "Position-gaze subspace overlap", args.output_dir / "alignment_open_vs_occluded.png",
    )
    write_json(args.output_dir / "analysis_manifest.json", {
        "comparison": "open versus occluded, factorized objective only",
        "checkpoint": args.checkpoint,
        "training_noise_sd": 0.2,
        "test_and_probe_noise_sd": args.noise_sd,
        "test_trials": args.test_trials,
        "probe_trials": args.probe_trials,
        "probe_split": "70% fit, 15% validation, 15% test by whole trajectory",
        "test_trajectory_seed": "4000000 + training seed",
        "probe_trajectory_seed": "6000000 + training seed",
        "ridge_grid": list(RIDGE_GRID),
        "bootstrap_draws": args.bootstrap_draws,
        "synthetic_recovery": synthetic,
        "environment_manifests_verified": len(args.seeds),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    })
    print(f"\nResults written to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

"""Hidden-state analysis of the frozen tuned objective-comparison RNNs.

Primary analysis
----------------
The validation-accuracy-matched factorized and conjunction checkpoints are
compared for seeds 2--5.  Checkpoints were selected using validation data; this
script never retrains a network or selects a checkpoint using probe results.

Sensitivity analysis
--------------------
The independently validation-best checkpoints from all five seeds can also be
analysed.  These are the best checkpoints observed during an 8,000-update
training budget, not necessarily the weights at update 8,000.

For every seed, both objectives receive exactly the same newly generated clean
probe trajectories.  Post-tanh recurrent activity is explained with four
categorical ridge-encoding models: place only, gaze only, additive place+gaze,
and joint place-gaze.  Model evaluation is grouped by whole trajectory.
Regularisation is selected inside each outer training fold, so held-out probe
trajectories do not influence model fitting or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import itertools
from dataclasses import dataclass
from pathlib import Path

# Keep this import order: the local experiment module imports juliacall before
# torch, which avoids Julia/PyTorch runtime conflicts on the cluster.
import factorized_vs_conjunction as experiment
import controlled_comparison as base

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


MODEL_NAMES = ("factorized", "conjunction")
ENCODING_NAMES = ("place", "gaze", "additive", "joint")
CATEGORIES = (
    "place-dominant",
    "gaze-dominant",
    "additive-mixed",
    "interaction-dependent",
    "poorly-explained",
    "unclassified",
)
COLORS = {
    "factorized": "#3976AF",
    "conjunction": "#D64B40",
    "place-dominant": "#4C78A8",
    "gaze-dominant": "#F2A541",
    "additive-mixed": "#59A14F",
    "interaction-dependent": "#B34E8C",
    "poorly-explained": "#9C9C9C",
    "unclassified": "#D8D8D8",
}


@dataclass(frozen=True)
class CategoricalDesign:
    """One or more complete categorical one-hot blocks, stored as labels.

    Keeping category IDs instead of an explicit 100,000-by-504 sparse matrix
    removes the SciPy dependency and makes the ridge solve both faster and
    easier to audit.
    """

    labels: tuple[np.ndarray, ...]
    levels: tuple[int, ...]

    def subset(self, rows: np.ndarray) -> "CategoricalDesign":
        return CategoricalDesign(tuple(values[rows] for values in self.labels), self.levels)

    def __len__(self) -> int:
        return len(self.labels[0])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_model(name: str) -> torch.nn.Module:
    if name == "factorized":
        return experiment.FactorizedDiscreteRNN()
    if name == "conjunction":
        return base.ConjunctionRNN()
    raise ValueError(f"Unknown objective: {name}")


def load_model(path: Path, name: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "seed", "best_update", "selection_split"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise RuntimeError(f"{path} is missing checkpoint fields: {missing}")
    if checkpoint["selection_split"] != "validation":
        raise RuntimeError(f"Checkpoint was not selected on validation data: {path}")
    model = make_model(name)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def best_checkpoint(root: Path, seed: int, name: str) -> Path:
    return root / "seed_runs" / f"seed_{seed}" / "checkpoints" / f"{name}_best.pt"


def matched_checkpoint_map(root: Path) -> dict[int, dict[str, Path]]:
    status_path = root / "accuracy_matched" / "matching_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    mapping: dict[int, dict[str, Path]] = {}
    for item in status["matching_by_seed"]:
        if item["status"] != "matched":
            continue
        seed = int(item["seed"])
        seed_root = root / "seed_runs" / f"seed_{seed}"
        mapping[seed] = {
            name: seed_root / item[name]["checkpoint"] for name in MODEL_NAMES
        }
    return mapping


def generate_probe(bridge: base.JuliaTrialBridge, seed: int, trajectories: int) -> dict:
    """Generate a clean probe bank that was not used for model selection."""
    probe_seed = 3_000_000 + seed
    raw = bridge.make_generator(trajectories, rseed=probe_seed)()
    inputs, _, joint, valid = base.unpack_batch(raw, torch.device("cpu"))
    batch, steps, _ = inputs.shape
    trajectory = np.broadcast_to(np.arange(batch)[:, None], (batch, steps))
    valid_np = valid.numpy()
    return {
        "inputs": inputs,
        "valid": valid_np,
        "joint": joint.numpy()[valid_np],
        "place": (joint.numpy()[valid_np] // experiment.SURFACE_BINS),
        "gaze": (joint.numpy()[valid_np] % experiment.SURFACE_BINS),
        "trajectory": trajectory[valid_np],
        "probe_seed": probe_seed,
    }


@torch.inference_mode()
def extract_hidden(
    model: torch.nn.Module,
    probe: dict,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return valid post-tanh recurrent activity in probe-row order."""
    inputs = probe["inputs"]
    valid = probe["valid"]
    pieces = []
    for start in range(0, len(inputs), batch_size):
        stop = min(start + batch_size, len(inputs))
        hidden, _ = model.rnn(inputs[start:stop].to(device))
        mask = torch.from_numpy(valid[start:stop]).to(device)
        pieces.append(hidden[mask].cpu().numpy().astype(np.float32))
    return np.concatenate(pieces, axis=0)


def make_designs(place: np.ndarray, gaze: np.ndarray) -> dict[str, CategoricalDesign]:
    joint = place * experiment.SURFACE_BINS + gaze
    return {
        "place": CategoricalDesign((place,), (experiment.PLACE_BINS,)),
        "gaze": CategoricalDesign((gaze,), (experiment.SURFACE_BINS,)),
        "additive": CategoricalDesign(
            (place, gaze), (experiment.PLACE_BINS, experiment.SURFACE_BINS)
        ),
        # A joint one-hot model spans arbitrary place-by-gaze condition means.
        # It is equivalent in predictive capacity to main effects + interaction,
        # without adding redundant one-hot columns that alter ridge penalties.
        "joint": CategoricalDesign((joint,), (experiment.JOINT_BINS,)),
    }


def joint_support_diagnostics(
    place: np.ndarray,
    gaze: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> dict[str, float]:
    """Report, rather than hide, conjunctions absent from outer training folds."""
    joint = place * experiment.SURFACE_BINS + gaze
    coverage = []
    for train, test in grouped_folds(groups, folds, seed):
        supported = np.zeros(experiment.JOINT_BINS, dtype=bool)
        supported[np.unique(joint[train])] = True
        coverage.append(float(np.mean(supported[joint[test]])))
    return {
        "observed_joint_classes": int(len(np.unique(joint))),
        "total_joint_classes": int(experiment.JOINT_BINS),
        "mean_outer_test_supported_fraction": float(np.mean(coverage)),
        "minimum_outer_test_supported_fraction": float(np.min(coverage)),
    }


def grouped_folds(groups: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups)
    if len(unique) < folds:
        raise ValueError(f"Need at least {folds} trajectories, found {len(unique)}")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    parts = np.array_split(shuffled, folds)
    output = []
    for part in parts:
        test = np.isin(groups, part)
        output.append((np.flatnonzero(~test), np.flatnonzero(test)))
    return output


def raw_r2(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    residual = np.sum((actual - predicted) ** 2, axis=0, dtype=np.float64)
    centred = actual - actual.mean(axis=0, keepdims=True)
    total = np.sum(centred**2, axis=0, dtype=np.float64)
    return np.where(total > 1e-12, 1.0 - residual / total, np.nan)


def fit_ridge(x: CategoricalDesign, y: np.ndarray, alpha: float) -> np.ndarray:
    """Fit all 256 units together by solving the ridge normal equations.

    Every design contains a complete categorical one-hot block, so an explicit
    intercept is unnecessary.  This multi-output solve is much faster than
    fitting 256 separate sklearn estimators and gives the same predictions for
    the specified objective.
    """
    offsets = np.cumsum((0,) + x.levels)
    feature_count = int(offsets[-1])
    gram = np.zeros((feature_count, feature_count), dtype=np.float64)
    right = np.zeros((feature_count, y.shape[1]), dtype=np.float64)

    # Diagonal blocks contain per-level observation counts. The right-hand
    # side contains the activity sum for every categorical level.
    for block, (labels, levels) in enumerate(zip(x.labels, x.levels)):
        start, stop = int(offsets[block]), int(offsets[block + 1])
        counts = np.bincount(labels, minlength=levels).astype(np.float64)
        diagonal = start + np.arange(levels)
        gram[diagonal, diagonal] = counts
        np.add.at(right[start:stop], labels, y)

    # Off-diagonal blocks contain co-occurrence counts. This is needed only by
    # the additive place+gaze model; the other models have one block.
    for left in range(len(x.labels)):
        for right_block in range(left + 1, len(x.labels)):
            left_labels, right_labels = x.labels[left], x.labels[right_block]
            left_levels, right_levels = x.levels[left], x.levels[right_block]
            pair = left_labels * right_levels + right_labels
            counts = np.bincount(
                pair, minlength=left_levels * right_levels
            ).reshape(left_levels, right_levels)
            ls, le = int(offsets[left]), int(offsets[left + 1])
            rs, re = int(offsets[right_block]), int(offsets[right_block + 1])
            gram[ls:le, rs:re] = counts
            gram[rs:re, ls:le] = counts.T

    gram.flat[:: len(gram) + 1] += alpha
    return np.linalg.solve(gram, right)


def predict_ridge(coefficients: np.ndarray, x: CategoricalDesign) -> np.ndarray:
    output = np.zeros((len(x), coefficients.shape[1]), dtype=np.float32)
    offset = 0
    for labels, levels in zip(x.labels, x.levels):
        output += coefficients[offset + labels].astype(np.float32, copy=False)
        offset += levels
    return output


def choose_alpha(
    x,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
    folds: int,
    seed: int,
) -> float:
    """Choose one regularisation value using only outer-training rows."""
    splits = grouped_folds(groups, folds, seed)
    candidates = []
    for alpha in alphas:
        prediction = np.empty_like(y)
        for train, test in splits:
            coefficients = fit_ridge(x.subset(train), y[train], alpha)
            prediction[test] = predict_ridge(coefficients, x.subset(test))
        median_r2 = float(np.nanmedian(raw_r2(y, prediction)))
        candidates.append((median_r2, -alpha, alpha))
    return float(max(candidates)[2])


def nested_group_predictions(
    designs: dict,
    hidden: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    """Generate out-of-fold predictions with nested grouped alpha selection."""
    predictions = {name: np.empty_like(hidden) for name in ENCODING_NAMES}
    selected = {name: [] for name in ENCODING_NAMES}
    outer = grouped_folds(groups, outer_folds, seed)
    for outer_index, (train, test) in enumerate(outer):
        for model_index, name in enumerate(ENCODING_NAMES):
            alpha = choose_alpha(
                designs[name].subset(train),
                hidden[train],
                groups[train],
                alphas,
                inner_folds,
                seed + 1000 * outer_index + 37 * model_index,
            )
            coefficients = fit_ridge(designs[name].subset(train), hidden[train], alpha)
            predictions[name][test] = predict_ridge(
                coefficients, designs[name].subset(test)
            )
            selected[name].append(alpha)
    return predictions, selected


def scores_from_predictions(actual: np.ndarray, predictions: dict) -> dict[str, np.ndarray]:
    r2 = {name: raw_r2(actual, predictions[name]) for name in ENCODING_NAMES}
    return {
        **{f"r2_{name}": value for name, value in r2.items()},
        "unique_place": r2["additive"] - r2["gaze"],
        "unique_gaze": r2["additive"] - r2["place"],
        "interaction_gain": r2["joint"] - r2["additive"],
    }


def trajectory_summaries(
    actual: np.ndarray,
    predictions: dict,
    groups: np.ndarray,
) -> dict:
    unique = np.unique(groups)
    counts, sums, sums2 = [], [], []
    residual = {name: [] for name in ENCODING_NAMES}
    for group in unique:
        rows = groups == group
        y = actual[rows].astype(np.float64)
        counts.append(int(rows.sum()))
        sums.append(y.sum(axis=0))
        sums2.append((y * y).sum(axis=0))
        for name in ENCODING_NAMES:
            error = y - predictions[name][rows]
            residual[name].append((error * error).sum(axis=0))
    return {
        "groups": unique,
        "count": np.asarray(counts, dtype=np.int64),
        "sum": np.asarray(sums),
        "sum2": np.asarray(sums2),
        "residual": {name: np.asarray(value) for name, value in residual.items()},
    }


def bootstrap_intervals(
    actual: np.ndarray,
    predictions: dict,
    groups: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Bootstrap whole trajectories while reusing fixed OOF predictions."""
    summary = trajectory_summaries(actual, predictions, groups)
    number = len(summary["groups"])
    rng = np.random.default_rng(seed)
    bootstrap = {name: [] for name in ENCODING_NAMES}
    for _ in range(draws):
        picked = rng.integers(0, number, size=number)
        n = summary["count"][picked].sum()
        total_sum = summary["sum"][picked].sum(axis=0)
        total_sum2 = summary["sum2"][picked].sum(axis=0)
        denominator = total_sum2 - total_sum * total_sum / n
        for name in ENCODING_NAMES:
            residual = summary["residual"][name][picked].sum(axis=0)
            bootstrap[name].append(np.where(denominator > 1e-12, 1 - residual / denominator, np.nan))
    r2 = {name: np.asarray(values) for name, values in bootstrap.items()}
    gains = {
        "unique_place": r2["additive"] - r2["gaze"],
        "unique_gaze": r2["additive"] - r2["place"],
        "interaction_gain": r2["joint"] - r2["additive"],
    }
    return {
        name: (
            np.nanpercentile(values, 2.5, axis=0),
            np.nanpercentile(values, 97.5, axis=0),
        )
        for name, values in gains.items()
    }


def classify_units(
    scores: dict[str, np.ndarray],
    intervals: dict[str, tuple[np.ndarray, np.ndarray]],
    threshold: float,
    explained_threshold: float,
) -> list[str]:
    labels = []
    p_low, p_high = intervals["unique_place"]
    g_low, g_high = intervals["unique_gaze"]
    i_low, i_high = intervals["interaction_gain"]
    for unit in range(len(scores["r2_joint"])):
        if scores["r2_joint"][unit] < explained_threshold:
            label = "poorly-explained"
        elif i_low[unit] > threshold:
            label = "interaction-dependent"
        elif p_low[unit] > threshold and g_low[unit] > threshold and i_high[unit] <= threshold:
            label = "additive-mixed"
        elif p_low[unit] > threshold and g_high[unit] <= threshold and i_high[unit] <= threshold:
            label = "place-dominant"
        elif g_low[unit] > threshold and p_high[unit] <= threshold and i_high[unit] <= threshold:
            label = "gaze-dominant"
        else:
            label = "unclassified"
        labels.append(label)
    return labels


def variance_rank(values: np.ndarray, fraction: float, cap: int) -> int:
    energy = values**2
    if energy.sum() <= 0:
        return 1
    return min(cap, int(np.searchsorted(np.cumsum(energy) / energy.sum(), fraction) + 1))


def subspace_geometry(designs: dict, hidden: np.ndarray, alpha: float) -> dict:
    """Fit an additive model and compare its place and gaze tuning subspaces."""
    coefficients = fit_ridge(designs["additive"], hidden, alpha)
    place = coefficients[: experiment.PLACE_BINS]
    gaze = coefficients[experiment.PLACE_BINS :]
    place -= place.mean(axis=0, keepdims=True)
    gaze -= gaze.mean(axis=0, keepdims=True)
    _, place_s, place_v = np.linalg.svd(place, full_matrices=False)
    _, gaze_s, gaze_v = np.linalg.svd(gaze, full_matrices=False)
    place_rank = variance_rank(place_s, 0.90, 10)
    gaze_rank = variance_rank(gaze_s, 0.90, 10)
    singular = np.linalg.svd(
        place_v[:place_rank] @ gaze_v[:gaze_rank].T,
        compute_uv=False,
    )
    singular = np.clip(singular, 0.0, 1.0)
    angles = np.degrees(np.arccos(singular))
    return {
        "place_rank": place_rank,
        "gaze_rank": gaze_rank,
        "minimum_principal_angle_degrees": float(angles.min()),
        "mean_principal_angle_degrees": float(angles.mean()),
        "subspace_alignment_mean_cos2": float(np.mean(singular**2)),
        "principal_angles_degrees": ";".join(f"{value:.6f}" for value in angles),
    }


def analyze_network(
    model: torch.nn.Module,
    probe: dict,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> tuple[list[dict], dict, dict]:
    hidden = extract_hidden(model, probe, device, args.inference_batch_size)
    designs = make_designs(probe["place"], probe["gaze"])
    predictions, alphas = nested_group_predictions(
        designs,
        hidden,
        probe["trajectory"],
        args.ridge_alphas,
        args.outer_folds,
        args.inner_folds,
        40_000 + seed,
    )
    scores = scores_from_predictions(hidden, predictions)
    intervals = bootstrap_intervals(
        hidden,
        predictions,
        probe["trajectory"],
        args.bootstrap_draws,
        50_000 + seed,
    )
    labels = classify_units(
        scores, intervals, args.effect_threshold, args.explained_threshold
    )
    unit_rows = []
    for unit, label in enumerate(labels):
        row = {"unit": unit, "category": label}
        row.update({name: float(values[unit]) for name, values in scores.items()})
        for name, (low, high) in intervals.items():
            row[f"{name}_ci_low"] = float(low[unit])
            row[f"{name}_ci_high"] = float(high[unit])
        unit_rows.append(row)
    summary = {
        "valid_timesteps": len(hidden),
        "probe_trajectories": int(len(np.unique(probe["trajectory"]))),
        "median_additive_r2": float(np.nanmedian(scores["r2_additive"])),
        "median_joint_r2": float(np.nanmedian(scores["r2_joint"])),
        "median_interaction_gain": float(np.nanmedian(scores["interaction_gain"])),
        **{f"{category}_units": labels.count(category) for category in CATEGORIES},
        **{
            f"median_selected_alpha_{name}": float(np.median(values))
            for name, values in alphas.items()
        },
        **joint_support_diagnostics(
            probe["place"],
            probe["gaze"],
            probe["trajectory"],
            args.outer_folds,
            40_000 + seed,
        ),
    }
    geometry = subspace_geometry(
        designs, hidden, summary["median_selected_alpha_additive"]
    )
    return unit_rows, summary, geometry


def plot_categories(summary_rows: list[dict], kind: str, path: Path) -> None:
    rows = [row for row in summary_rows if row["analysis"] == kind]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    x = np.arange(len(MODEL_NAMES), dtype=float)
    bottom = np.zeros(len(MODEL_NAMES))
    for category in CATEGORIES:
        means = []
        for name in MODEL_NAMES:
            values = np.asarray(
                [100 * row[f"{category}_units"] / base.HIDDEN_DIM for row in rows if row["objective"] == name]
            )
            means.append(values.mean())
        ax.bar(x, means, bottom=bottom, color=COLORS[category], label=category)
        bottom += means
    ax.set_xticks(x, ("Factorized", "Conjunction"))
    ax.set_ylim(0, 100)
    ax.set_ylabel("Hidden units (%)")
    ax.set_title(f"Hidden-state encoding categories — {kind.replace('_', ' ')}")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_paired(summary_rows: list[dict], kind: str, field: str, ylabel: str, path: Path) -> None:
    rows = [row for row in summary_rows if row["analysis"] == kind]
    by_seed = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[row["objective"]] = float(row[field])
    fig, ax = plt.subplots(figsize=(6.5, 5.3), dpi=180)
    for seed, values in sorted(by_seed.items()):
        if set(values) != set(MODEL_NAMES):
            continue
        ax.plot([0, 1], [values["factorized"], values["conjunction"]], color="#AAB2BD", lw=1.8)
        ax.scatter(0, values["factorized"], color=COLORS["factorized"], s=55, zorder=3)
        ax.scatter(1, values["conjunction"], color=COLORS["conjunction"], s=55, zorder=3)
    ax.set_xticks([0, 1], ["Factorized", "Conjunction"])
    ax.set_ylabel(ylabel)
    ax.set_title(kind.replace("_", " ").title())
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def paired_statistics(summary_rows: list[dict], geometry_rows: list[dict]) -> list[dict]:
    """Use training seeds—not units or timesteps—as statistical replicates."""
    rows = []
    fields = [
        ("median_interaction_gain", summary_rows, 1.0),
        ("median_additive_r2", summary_rows, 1.0),
        ("median_joint_r2", summary_rows, 1.0),
        ("minimum_principal_angle_degrees", geometry_rows, 1.0),
        ("subspace_alignment_mean_cos2", geometry_rows, 1.0),
    ] + [
        (f"{category}_percent", summary_rows, 100.0 / base.HIDDEN_DIM)
        for category in CATEGORIES
    ]
    analyses = sorted({row["analysis"] for row in summary_rows})
    for analysis in analyses:
        for output_name, source, scale in fields:
            source_field = output_name.replace("_percent", "_units")
            selected = [row for row in source if row["analysis"] == analysis]
            paired = {}
            for row in selected:
                paired.setdefault(int(row["seed"]), {})[row["objective"]] = (
                    float(row[source_field]) * scale
                )
            differences = np.asarray(
                [
                    values["conjunction"] - values["factorized"]
                    for _, values in sorted(paired.items())
                    if set(values) == set(MODEL_NAMES)
                ],
                dtype=float,
            )
            if len(differences) == 0:
                continue
            observed = abs(float(differences.mean()))
            null = []
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
                null.append(abs(float(np.mean(differences * np.asarray(signs)))))
            p_value = float(np.mean(np.asarray(null) >= observed - 1e-15))
            rows.append(
                {
                    "analysis": analysis,
                    "outcome": output_name,
                    "paired_seeds": len(differences),
                    "mean_conjunction_minus_factorized": float(differences.mean()),
                    "standard_deviation_of_paired_difference": (
                        float(differences.std(ddof=1)) if len(differences) > 1 else math.nan
                    ),
                    "exact_two_sided_sign_flip_p": p_value,
                    "interpretation_note": (
                        f"Exploratory seed-level inference; with {len(differences)} "
                        "paired seeds, the minimum attainable two-sided p-value is "
                        f"{2 / (2 ** len(differences)):.4f}."
                    ),
                }
            )
    return rows


def checkpoint_sets(root: Path, requested: str) -> list[tuple[str, dict[int, dict[str, Path]]]]:
    sets = []
    if requested in ("matched", "both"):
        sets.append(("accuracy_matched", matched_checkpoint_map(root)))
    if requested in ("best", "both"):
        sets.append(
            (
                "validation_best",
                {
                    seed: {name: best_checkpoint(root, seed, name) for name in MODEL_NAMES}
                    for seed in range(1, 6)
                },
            )
        )
    return sets


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = base.choose_device(args.device)
    print(f"Device: {device}", flush=True)
    bridge = base.JuliaTrialBridge()
    unit_rows, summaries, geometries, checkpoint_rows = [], [], [], []
    probe_cache = {}
    for analysis, seeds in checkpoint_sets(args.results_root, args.analysis):
        for seed, paths in sorted(seeds.items()):
            print(f"\n=== {analysis}: seed {seed} ===", flush=True)
            probe = probe_cache.setdefault(
                seed, generate_probe(bridge, seed, args.probe_trajectories)
            )
            for name, checkpoint_path in paths.items():
                if not checkpoint_path.exists():
                    raise FileNotFoundError(checkpoint_path)
                model, checkpoint = load_model(checkpoint_path, name, device)
                network_units, summary, geometry = analyze_network(
                    model, probe, device, args, seed
                )
                prefix = {"analysis": analysis, "seed": seed, "objective": name}
                unit_rows.extend([{**prefix, **row} for row in network_units])
                summaries.append({**prefix, **summary})
                geometries.append({**prefix, **geometry})
                checkpoint_rows.append(
                    {
                        **prefix,
                        "checkpoint": str(checkpoint_path),
                        "selected_update": int(checkpoint["best_update"]),
                        "validation_joint_accuracy": float(
                            checkpoint["validation_joint_accuracy"]
                        ),
                        "training_trials_seen": int(checkpoint["training_trials_seen"]),
                        "probe_seed": int(probe["probe_seed"]),
                    }
                )
                print(
                    f"{name:11s}: update={checkpoint['best_update']} "
                    f"median gain={summary['median_interaction_gain']:.4f} "
                    f"interaction units={summary['interaction-dependent_units']}",
                    flush=True,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    write_csv(args.output_dir / "unit_scores.csv", unit_rows)
    write_csv(args.output_dir / "network_summary.csv", summaries)
    write_csv(args.output_dir / "subspace_geometry.csv", geometries)
    write_csv(args.output_dir / "checkpoint_manifest.csv", checkpoint_rows)
    write_csv(
        args.output_dir / "paired_seed_statistics.csv",
        paired_statistics(summaries, geometries),
    )
    for analysis, _ in checkpoint_sets(args.results_root, args.analysis):
        plot_categories(
            summaries, analysis, args.output_dir / f"{analysis}_unit_categories.png"
        )
        plot_paired(
            summaries,
            analysis,
            "median_interaction_gain",
            "Median held-out interaction gain (ΔR²)",
            args.output_dir / f"{analysis}_interaction_gain.png",
        )
        plot_paired(
            geometries,
            analysis,
            "minimum_principal_angle_degrees",
            "Minimum principal angle (degrees)",
            args.output_dir / f"{analysis}_principal_angle.png",
        )
    metadata = {
        "primary_analysis": "validation-accuracy-matched checkpoints, paired seeds 2-5",
        "sensitivity_analysis": "validation-best checkpoints, paired seeds 1-5",
        "checkpoint_selection": "validation only; probe data never select checkpoints",
        "probe_noise_standard_deviation": 0.0,
        "probe_trajectories_per_seed": args.probe_trajectories,
        "same_probe_trajectories_within_seed": True,
        "hidden_activity": "post-tanh recurrent state; padded timesteps excluded",
        "cross_validation": f"nested trajectory-grouped {args.outer_folds}-fold outer / {args.inner_folds}-fold inner",
        "ridge_alpha_grid": args.ridge_alphas,
        "bootstrap": f"{args.bootstrap_draws} whole-trajectory resamples of out-of-fold predictions",
        "effect_threshold_delta_r2": args.effect_threshold,
        "explained_threshold_joint_r2": args.explained_threshold,
        "unit_categories_are_secondary": True,
        "continuous_delta_r2_is_primary": True,
        "rare_joint_policy": (
            "No full-dataset filtering. All valid rows are evaluated; joint-class "
            "support in each outer training fold is reported in network_summary.csv."
        ),
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nResults written to {args.output_dir.resolve()}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("objective_results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("frozen_hidden_analysis"),
    )
    parser.add_argument("--analysis", choices=("matched", "best", "both"), default="both")
    parser.add_argument("--probe-trajectories", type=int, default=1024)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--ridge-alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--effect-threshold", type=float, default=0.01)
    parser.add_argument("--explained-threshold", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.analysis = "matched"
        args.probe_trajectories = 40
        args.outer_folds = 2
        args.inner_folds = 2
        args.ridge_alphas = [1.0]
        args.bootstrap_draws = 20
    return args


if __name__ == "__main__":
    run(parse_args())

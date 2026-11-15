"""Controlled comparison of factorized-discrete and conjunction RNN objectives.

Both models receive identical 36-channel inputs, batches, recurrent
initialization, optimizer settings, and held-out trials.  They differ only in
the categorical output parameterization:

* FactorizedDiscreteRNN: a 21-way place head and a 24-way surface head.
* ConjunctionRNN: one 504-way joint place-surface head.

The factorized loss is CE(place) + CE(surface).  This is the negative log
likelihood of the factorized joint distribution and has the same chance-level
magnitude as CE over 21*24 conjunction classes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

# This module imports juliacall before torch, which avoids bridge/runtime issues
# seen on some platforms.  It also supplies the corrected Julia trial bridge.
import controlled_comparison as base

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PLACE_BINS = base.PLACE_BINS
SURFACE_BINS = base.SURFACE_BINS
JOINT_BINS = base.OUTPUT_DIM


class FactorizedDiscreteRNN(nn.Module):
    """One recurrent core with independent place and surface readouts."""

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            base.INPUT_DIM,
            base.HIDDEN_DIM,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.place_head = nn.Linear(base.HIDDEN_DIM, PLACE_BINS)
        self.surface_head = nn.Linear(base.HIDDEN_DIM, SURFACE_BINS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, _ = self.rnn(x)
        return self.place_head(hidden), self.surface_head(hidden)


def factorized_predictions(
    place_logits: torch.Tensor,
    surface_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    place = place_logits.argmax(dim=-1)
    surface = surface_logits.argmax(dim=-1)
    joint = place * SURFACE_BINS + surface
    return place, surface, joint


def joint_predictions(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    joint = logits.argmax(dim=-1)
    return joint // SURFACE_BINS, joint % SURFACE_BINS, joint


def factorized_loss(
    place_logits: torch.Tensor,
    surface_logits: torch.Tensor,
    true_joint: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    true_place = true_joint // SURFACE_BINS
    true_surface = true_joint % SURFACE_BINS
    place_ce = F.cross_entropy(place_logits[valid], true_place[valid])
    surface_ce = F.cross_entropy(surface_logits[valid], true_surface[valid])
    # Sum, rather than average: log(21) + log(24) == log(504), so this is
    # directly comparable to joint negative log likelihood at initialization.
    return place_ce + surface_ce, place_ce, surface_ce


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_pair(
    bridge: base.JuliaTrialBridge,
    seed: int,
    updates: int,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    log_every: int,
) -> tuple[FactorizedDiscreteRNN, base.ConjunctionRNN, list[dict]]:
    base.set_seed(seed)
    generator = bridge.make_generator(batch_size, rseed=seed)

    factorized = FactorizedDiscreteRNN().to(device)
    conjunction = base.ConjunctionRNN().to(device)
    # Exact same recurrent initialization within a seed.  Only the readout
    # parameterization differs at update zero.
    conjunction.rnn.load_state_dict(factorized.rnn.state_dict())

    factorized_optimizer = torch.optim.Adam(
        factorized.parameters(), lr=base.LEARNING_RATE
    )
    conjunction_optimizer = torch.optim.Adam(
        conjunction.parameters(), lr=base.LEARNING_RATE
    )

    history: list[dict] = []
    start = time.time()
    for update in range(1, updates + 1):
        x, _, true_joint, valid = base.unpack_batch(generator(), device)
        true_place = true_joint // SURFACE_BINS
        true_surface = true_joint % SURFACE_BINS

        factorized.train()
        factorized_optimizer.zero_grad(set_to_none=True)
        factorized_place_logits, factorized_surface_logits = factorized(x)
        loss_factorized, loss_place, loss_surface = factorized_loss(
            factorized_place_logits,
            factorized_surface_logits,
            true_joint,
            valid,
        )
        loss_factorized.backward()
        factorized_norm = torch.nn.utils.clip_grad_norm_(
            factorized.parameters(), base.GRAD_CLIP
        )
        factorized_optimizer.step()

        conjunction.train()
        conjunction_optimizer.zero_grad(set_to_none=True)
        conjunction_logits = conjunction(x)
        loss_conjunction = F.cross_entropy(
            conjunction_logits[valid], true_joint[valid]
        )
        loss_conjunction.backward()
        conjunction_norm = torch.nn.utils.clip_grad_norm_(
            conjunction.parameters(), base.GRAD_CLIP
        )
        conjunction_optimizer.step()

        should_log = update == 1 or update == updates or update % log_every == 0
        if should_log:
            with torch.no_grad():
                fp, fs, fj = factorized_predictions(
                    factorized_place_logits, factorized_surface_logits
                )
                cp, cs, cj = joint_predictions(conjunction_logits)
                valid_true = true_joint[valid]
                row = {
                    "seed": seed,
                    "update": update,
                    "factorized_loss": loss_factorized.item(),
                    "factorized_place_loss": loss_place.item(),
                    "factorized_surface_loss": loss_surface.item(),
                    "conjunction_loss": loss_conjunction.item(),
                    "factorized_place_accuracy": (fp[valid] == true_place[valid])
                    .float()
                    .mean()
                    .item(),
                    "factorized_surface_accuracy": (
                        fs[valid] == true_surface[valid]
                    )
                    .float()
                    .mean()
                    .item(),
                    "factorized_joint_accuracy": (fj[valid] == valid_true)
                    .float()
                    .mean()
                    .item(),
                    "conjunction_place_accuracy": (cp[valid] == true_place[valid])
                    .float()
                    .mean()
                    .item(),
                    "conjunction_surface_accuracy": (
                        cs[valid] == true_surface[valid]
                    )
                    .float()
                    .mean()
                    .item(),
                    "conjunction_joint_accuracy": (cj[valid] == valid_true)
                    .float()
                    .mean()
                    .item(),
                    "factorized_gradient_norm_before_clip": float(factorized_norm),
                    "conjunction_gradient_norm_before_clip": float(conjunction_norm),
                    "factorized_was_clipped": int(
                        float(factorized_norm) > base.GRAD_CLIP
                    ),
                    "conjunction_was_clipped": int(
                        float(conjunction_norm) > base.GRAD_CLIP
                    ),
                    "elapsed_seconds": time.time() - start,
                }
            history.append(row)
            print(
                f"seed={seed} update={update}/{updates} "
                f"factorized_joint={100 * row['factorized_joint_accuracy']:.2f}% "
                f"conjunction_joint={100 * row['conjunction_joint_accuracy']:.2f}% "
                f"factorized_loss={loss_factorized.item():.4f} "
                f"conjunction_loss={loss_conjunction.item():.4f}",
                flush=True,
            )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": seed,
        "input_dim": base.INPUT_DIM,
        "hidden_dim": base.HIDDEN_DIM,
        "train_updates": updates,
        "batch_size": batch_size,
        "learning_rate": base.LEARNING_RATE,
        "gradient_clip": base.GRAD_CLIP,
        "min_steps": base.MIN_STEPS,
        "max_steps": base.MAX_STEPS,
    }
    torch.save(
        {
            **metadata,
            "objective": "factorized_place_surface",
            "model_state_dict": factorized.state_dict(),
        },
        checkpoint_dir / f"seed_{seed}_factorized.pt",
    )
    torch.save(
        {
            **metadata,
            "objective": "joint_place_surface_conjunction",
            "model_state_dict": conjunction.state_dict(),
        },
        checkpoint_dir / f"seed_{seed}_conjunction.pt",
    )
    return factorized, conjunction, history


def macro_accuracy(
    true: np.ndarray,
    prediction: np.ndarray,
    class_count: int,
) -> float:
    values = []
    for label in range(class_count):
        selected = true == label
        if np.any(selected):
            values.append(float(np.mean(prediction[selected] == label)))
    return float(np.mean(values))


def model_metrics(
    prefix: str,
    true_place: np.ndarray,
    true_surface: np.ndarray,
    true_joint: np.ndarray,
    pred_place: np.ndarray,
    pred_surface: np.ndarray,
    pred_joint: np.ndarray,
    top5_correct: np.ndarray,
    supported_mask: np.ndarray,
) -> dict:
    place_correct = pred_place == true_place
    surface_correct = pred_surface == true_surface
    joint_correct = pred_joint == true_joint
    return {
        f"{prefix}_place_accuracy": float(np.mean(place_correct)),
        f"{prefix}_surface_accuracy": float(np.mean(surface_correct)),
        f"{prefix}_joint_accuracy": float(np.mean(joint_correct)),
        f"{prefix}_joint_top5_accuracy": float(np.mean(top5_correct)),
        f"{prefix}_macro_place_accuracy": macro_accuracy(
            true_place, pred_place, PLACE_BINS
        ),
        f"{prefix}_macro_surface_accuracy": macro_accuracy(
            true_surface, pred_surface, SURFACE_BINS
        ),
        f"{prefix}_macro_joint_accuracy": macro_accuracy(
            true_joint, pred_joint, JOINT_BINS
        ),
        f"{prefix}_unsupported_joint_rate": float(
            np.mean(~supported_mask[pred_joint])
        ),
        f"{prefix}_both_correct": float(np.mean(place_correct & surface_correct)),
        f"{prefix}_place_only_correct": float(
            np.mean(place_correct & ~surface_correct)
        ),
        f"{prefix}_surface_only_correct": float(
            np.mean(~place_correct & surface_correct)
        ),
        f"{prefix}_neither_correct": float(
            np.mean(~place_correct & ~surface_correct)
        ),
        f"{prefix}_surface_accuracy_given_place_correct": float(
            np.mean(surface_correct[place_correct])
        ) if np.any(place_correct) else float("nan"),
        f"{prefix}_place_accuracy_given_surface_correct": float(
            np.mean(place_correct[surface_correct])
        ) if np.any(surface_correct) else float("nan"),
    }


@torch.no_grad()
def evaluate_pair(
    bridge: base.JuliaTrialBridge,
    factorized: FactorizedDiscreteRNN,
    conjunction: base.ConjunctionRNN,
    seed: int,
    eval_trials: int,
    batch_size: int,
    device: torch.device,
    test_seed_base: int = 100_000,
) -> tuple[dict, dict[str, np.ndarray]]:
    if eval_trials % batch_size:
        raise ValueError("eval_trials must be divisible by batch_size")
    generator = bridge.make_generator(batch_size, rseed=test_seed_base + seed)
    factorized.eval()
    conjunction.eval()

    true_parts, true_xy_parts = [], []
    fp_parts, fs_parts, fj_parts, f5_parts = [], [], [], []
    cp_parts, cs_parts, cj_parts, c5_parts = [], [], [], []

    for _ in range(eval_trials // batch_size):
        x, true_xy, true_joint, valid = base.unpack_batch(generator(), device)
        factorized_place_logits, factorized_surface_logits = factorized(x)
        conjunction_logits = conjunction(x)

        fp, fs, fj = factorized_predictions(
            factorized_place_logits, factorized_surface_logits
        )
        cp, cs, cj = joint_predictions(conjunction_logits)

        valid_true = true_joint[valid]
        factorized_joint_scores = (
            factorized_place_logits[valid].unsqueeze(-1)
            + factorized_surface_logits[valid].unsqueeze(-2)
        ).flatten(start_dim=-2)
        factorized_top5 = (
            factorized_joint_scores.topk(5, dim=-1).indices
            == valid_true.unsqueeze(-1)
        ).any(dim=-1)
        conjunction_top5 = (
            conjunction_logits[valid].topk(5, dim=-1).indices
            == valid_true.unsqueeze(-1)
        ).any(dim=-1)

        true_parts.append(valid_true.cpu().numpy())
        true_xy_parts.append(true_xy[valid].cpu().numpy())
        fp_parts.append(fp[valid].cpu().numpy())
        fs_parts.append(fs[valid].cpu().numpy())
        fj_parts.append(fj[valid].cpu().numpy())
        f5_parts.append(factorized_top5.cpu().numpy())
        cp_parts.append(cp[valid].cpu().numpy())
        cs_parts.append(cs[valid].cpu().numpy())
        cj_parts.append(cj[valid].cpu().numpy())
        c5_parts.append(conjunction_top5.cpu().numpy())

    true_joint = np.concatenate(true_parts)
    true_xy = np.concatenate(true_xy_parts)
    true_place = true_joint // SURFACE_BINS
    true_surface = true_joint % SURFACE_BINS
    fp, fs, fj, f5 = map(
        np.concatenate, [fp_parts, fs_parts, fj_parts, f5_parts]
    )
    cp, cs, cj, c5 = map(
        np.concatenate, [cp_parts, cs_parts, cj_parts, c5_parts]
    )

    supported_mask = np.zeros(JOINT_BINS, dtype=bool)
    supported_mask[np.unique(true_joint)] = True
    position_centres = base.normalized_position_centres()
    surface_centres = base.normalized_surface_centres()
    factorized_position_error = np.linalg.norm(
        position_centres[fp] - true_xy[:, :2], axis=1
    ) / base.GRID_SQUARE_NORMALIZED
    conjunction_position_error = np.linalg.norm(
        position_centres[cp] - true_xy[:, :2], axis=1
    ) / base.GRID_SQUARE_NORMALIZED
    factorized_gaze_error = np.linalg.norm(
        surface_centres[fs] - true_xy[:, 2:], axis=1
    ) / base.GRID_SQUARE_NORMALIZED
    conjunction_gaze_error = np.linalg.norm(
        surface_centres[cs] - true_xy[:, 2:], axis=1
    ) / base.GRID_SQUARE_NORMALIZED
    row = {
        "seed": seed,
        "valid_timesteps": int(len(true_joint)),
        "supported_joint_classes": int(np.sum(supported_mask)),
        **model_metrics(
            "factorized",
            true_place,
            true_surface,
            true_joint,
            fp,
            fs,
            fj,
            f5,
            supported_mask,
        ),
        **model_metrics(
            "conjunction",
            true_place,
            true_surface,
            true_joint,
            cp,
            cs,
            cj,
            c5,
            supported_mask,
        ),
        "factorized_position_error_median": float(
            np.median(factorized_position_error)
        ),
        "conjunction_position_error_median": float(
            np.median(conjunction_position_error)
        ),
        "factorized_gaze_error_median": float(np.median(factorized_gaze_error)),
        "conjunction_gaze_error_median": float(
            np.median(conjunction_gaze_error)
        ),
        "factorized_position_error_mean": float(
            np.mean(factorized_position_error)
        ),
        "conjunction_position_error_mean": float(
            np.mean(conjunction_position_error)
        ),
        "factorized_gaze_error_mean": float(np.mean(factorized_gaze_error)),
        "conjunction_gaze_error_mean": float(np.mean(conjunction_gaze_error)),
    }
    arrays = {
        "true_xy": true_xy,
        "true_joint": true_joint,
        "true_place": true_place,
        "true_surface": true_surface,
        "factorized_place": fp,
        "factorized_surface": fs,
        "factorized_joint": fj,
        "conjunction_place": cp,
        "conjunction_surface": cs,
        "conjunction_joint": cj,
        "supported_joint_mask": supported_mask,
        "factorized_position_error": factorized_position_error,
        "conjunction_position_error": conjunction_position_error,
        "factorized_gaze_error": factorized_gaze_error,
        "conjunction_gaze_error": conjunction_gaze_error,
    }
    return row, arrays


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    # Student-t critical values for a two-sided 95% interval.  Runs normally
    # use five seeds (df=4, critical value 2.776).
    critical = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
    }.get(len(values) - 1, 1.96)
    return mean, float(critical * np.std(values, ddof=1) / math.sqrt(len(values)))


def grouped_accuracy_plot(rows: list[dict], path: Path) -> None:
    metrics = [
        ("Position bin", "place_accuracy"),
        ("Surface bin", "surface_accuracy"),
        ("Exact conjunction", "joint_accuracy"),
    ]
    models = [
        ("Factorized discrete", "factorized", "#3976AF"),
        ("Conjunction trained", "conjunction", "#D64B40"),
    ]
    x = np.arange(len(metrics), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=170)
    for model_index, (label, prefix, color) in enumerate(models):
        positions = x + (model_index - 0.5) * width
        for metric_index, (_, suffix) in enumerate(metrics):
            values = np.asarray(
                [row[f"{prefix}_{suffix}"] for row in rows], dtype=float
            )
            mean, error = mean_ci(values)
            ax.bar(
                positions[metric_index],
                mean,
                width=width * 0.92,
                color=color,
                alpha=0.86,
                yerr=error,
                capsize=4,
                label=label if metric_index == 0 else None,
            )
            jitter = np.linspace(-0.04, 0.04, len(values))
            ax.scatter(
                positions[metric_index] + jitter,
                values,
                color="black",
                s=20,
                alpha=0.72,
                zorder=3,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out accuracy")
    ax.set_title("Factorized discrete versus conjunction training")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def dumbbell_plot(rows: list[dict], path: Path) -> None:
    metrics = [
        ("Position bin", "place_accuracy"),
        ("Surface bin", "surface_accuracy"),
        ("Exact conjunction", "joint_accuracy"),
    ]
    y = np.arange(len(metrics))[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=170)
    for row_index, ((_, suffix), ypos) in enumerate(zip(metrics, y)):
        factorized = np.asarray(
            [row[f"factorized_{suffix}"] for row in rows], dtype=float
        )
        conjunction = np.asarray(
            [row[f"conjunction_{suffix}"] for row in rows], dtype=float
        )
        ax.plot(
            [factorized.mean(), conjunction.mean()],
            [ypos, ypos],
            color="#444444",
            linewidth=2.4,
            zorder=1,
        )
        jitter = np.linspace(-0.055, 0.055, len(rows))
        ax.scatter(
            factorized,
            ypos + jitter,
            color="#3976AF",
            alpha=0.35,
            s=28,
            zorder=2,
        )
        ax.scatter(
            conjunction,
            ypos + jitter,
            color="#D64B40",
            alpha=0.35,
            s=28,
            zorder=2,
        )
        ax.scatter(
            factorized.mean(),
            ypos,
            color="#3976AF",
            s=130,
            label="Factorized discrete" if row_index == 0 else None,
            zorder=3,
        )
        ax.scatter(
            conjunction.mean(),
            ypos,
            color="#D64B40",
            s=130,
            label="Conjunction trained" if row_index == 0 else None,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _ in metrics])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Held-out accuracy")
    ax.set_title("Paired comparison by training seed")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.22, top=0.9)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def floor_grid(values: np.ndarray) -> np.ndarray:
    pillar_cells = {(1, 1), (3, 1), (1, 3), (3, 3)}
    grid = np.full((5, 5), np.nan)
    place = 0
    for xbin in range(5):
        for ybin in range(5):
            if (xbin, ybin) not in pillar_cells:
                grid[ybin, xbin] = values[place]
                place += 1
    return grid


def nanmean_where_available(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Mean of finite values, leaving all-missing bins as NaN without warnings."""
    count = np.isfinite(values).sum(axis=axis)
    total = np.nansum(values, axis=axis)
    result = np.full(np.shape(total), np.nan, dtype=float)
    return np.divide(total, count, out=result, where=count > 0)


def per_bin_errors(
    arrays_by_seed: list[dict[str, np.ndarray]],
    component: str,
    model: str,
    class_count: int,
) -> np.ndarray:
    per_seed = []
    truth_key = f"true_{component}"
    prediction_key = f"{model}_{component}"
    for arrays in arrays_by_seed:
        truth = arrays[truth_key]
        prediction = arrays[prediction_key]
        values = np.full(class_count, np.nan)
        for label in range(class_count):
            selected = truth == label
            if np.any(selected):
                values[label] = np.mean(prediction[selected] != label)
        per_seed.append(values)
    return nanmean_where_available(np.stack(per_seed), axis=0)


def place_error_map(arrays_by_seed: list[dict[str, np.ndarray]], path: Path) -> None:
    models = [
        ("Factorized discrete", "factorized"),
        ("Conjunction trained", "conjunction"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=180)
    image = None
    for ax, (title, model) in zip(axes, models):
        grid = floor_grid(
            per_bin_errors(arrays_by_seed, "place", model, PLACE_BINS)
        )
        image = ax.imshow(
            np.ma.masked_invalid(grid),
            origin="lower",
            vmin=0,
            vmax=1,
            cmap="magma",
        )
        for ybin in range(5):
            for xbin in range(5):
                if np.isfinite(grid[ybin, xbin]):
                    color = "white" if grid[ybin, xbin] < 0.55 else "black"
                    ax.text(
                        xbin,
                        ybin,
                        f"{100 * grid[ybin, xbin]:.1f}%",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=9,
                    )
                else:
                    ax.text(xbin, ybin, "pillar", ha="center", va="center", color="white")
        ax.set_title(title)
        ax.set_xlabel("Floor x-bin")
        ax.set_ylabel("Floor y-bin")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
    assert image is not None
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.04)
    bar.set_label("Wrong place-bin rate")
    fig.suptitle("Place-bin error by true floor bin (mean across seeds)")
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.1, top=0.85, wspace=0.24)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def surface_error_map(
    arrays_by_seed: list[dict[str, np.ndarray]], path: Path
) -> None:
    factorized = per_bin_errors(
        arrays_by_seed, "surface", "factorized", SURFACE_BINS
    )
    conjunction = per_bin_errors(
        arrays_by_seed, "surface", "conjunction", SURFACE_BINS
    )
    matrix = np.stack([factorized, conjunction])
    fig, ax = plt.subplots(figsize=(15, 3.6), dpi=180)
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="magma")
    for row in range(2):
        for column in range(SURFACE_BINS):
            color = "white" if matrix[row, column] < 0.55 else "black"
            ax.text(
                column,
                row,
                f"{100 * matrix[row, column]:.0f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Factorized discrete", "Conjunction trained"])
    ax.set_xticks(np.arange(SURFACE_BINS))
    ax.set_xticklabels(np.arange(1, SURFACE_BINS + 1))
    ax.set_xlabel("True surface bin")
    ax.set_title("Surface-bin error by true gaze surface (mean across seeds)")
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    bar.set_label("Wrong surface-bin rate")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def per_bin_median_value(
    arrays_by_seed: list[dict[str, np.ndarray]],
    truth_key: str,
    value_key: str,
    class_count: int,
) -> np.ndarray:
    """Mean across seeds of each seed's median value in every true bin."""
    per_seed = []
    for arrays in arrays_by_seed:
        truth = arrays[truth_key]
        observations = arrays[value_key]
        values = np.full(class_count, np.nan, dtype=float)
        for label in range(class_count):
            selected = truth == label
            if np.any(selected):
                values[label] = float(np.median(observations[selected]))
        per_seed.append(values)
    return nanmean_where_available(np.stack(per_seed), axis=0)


def position_coordinate_error_map(
    arrays_by_seed: list[dict[str, np.ndarray]], path: Path
) -> None:
    """Plot decoded position-coordinate distance by the true floor bin."""
    models = [
        ("Factorized discrete", "factorized"),
        ("Conjunction trained", "conjunction"),
    ]
    grids = []
    for _, model in models:
        values = per_bin_median_value(
            arrays_by_seed,
            "true_place",
            f"{model}_position_error",
            PLACE_BINS,
        )
        grids.append(floor_grid(values))

    shared_max = max(float(np.nanmax(grid)) for grid in grids)
    shared_max = max(shared_max, np.finfo(float).eps)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=180)
    image = None
    for ax, (title, _), grid in zip(axes, models, grids):
        image = ax.imshow(
            np.ma.masked_invalid(grid),
            origin="lower",
            vmin=0,
            vmax=shared_max,
            cmap="magma",
        )
        for ybin in range(5):
            for xbin in range(5):
                value = grid[ybin, xbin]
                if np.isfinite(value):
                    color = "white" if value < 0.55 * shared_max else "black"
                    ax.text(
                        xbin,
                        ybin,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=9,
                    )
                else:
                    ax.text(
                        xbin,
                        ybin,
                        "pillar",
                        ha="center",
                        va="center",
                        color="white",
                    )
        ax.set_title(title)
        ax.set_xlabel("True floor x-bin")
        ax.set_ylabel("True floor y-bin")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
    assert image is not None
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.04)
    bar.set_label("Median decoded position error (grid squares)")
    fig.suptitle(
        "Position-coordinate error by true floor bin "
        f"(mean of {len(arrays_by_seed)} seed medians)"
    )
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.1, top=0.85, wspace=0.24)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def restore_physical(normalized: np.ndarray) -> np.ndarray:
    """Convert normalized coordinates back to the 0--10 arena."""
    return 10.0 * (normalized - 0.05) / 0.8


def arena_surface_segments() -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the 24 gaze surfaces in RNNTrialStructures bin order."""
    segments: list[tuple[np.ndarray, np.ndarray]] = []

    # The outer wall contributes two five-square surface bins per side.
    corners = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    for start, end in zip(corners, corners[1:] + corners[:1]):
        first = np.asarray(start, dtype=float)
        second = np.asarray(end, dtype=float)
        midpoint = (first + second) / 2.0
        segments.append((first, midpoint))
        segments.append((midpoint, second))

    # The four faces of each pillar contribute the remaining sixteen bins.
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


def make_surface_subsegments(
    resolution: float = 0.5,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Split each gaze surface into short spatial plotting segments."""
    lines: list[np.ndarray] = []
    counts: list[int] = []
    for start, end in arena_surface_segments():
        count = max(1, int(round(float(np.linalg.norm(end - start)) / resolution)))
        counts.append(count)
        for local_index in range(count):
            left = start + (end - start) * (local_index / count)
            right = start + (end - start) * ((local_index + 1) / count)
            lines.append(np.stack([left, right]))
    return lines, np.asarray(counts, dtype=int)


def surface_spatial_groups(
    gaze_xy: np.ndarray,
    true_surface: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Assign each true gaze coordinate to a subsegment of its true surface."""
    surfaces = arena_surface_segments()
    starts = np.stack([item[0] for item in surfaces])
    deltas = np.stack([item[1] - item[0] for item in surfaces])
    squared_lengths = np.sum(deltas * deltas, axis=1)
    offsets = np.concatenate([[0], np.cumsum(counts[:-1])])
    start = starts[true_surface]
    delta = deltas[true_surface]
    fraction = np.sum((gaze_xy - start) * delta, axis=1) / squared_lengths[
        true_surface
    ]
    fraction = np.clip(fraction, 0.0, np.nextafter(1.0, 0.0))
    local = np.floor(fraction * counts[true_surface]).astype(int)
    return offsets[true_surface] + local


def grouped_medians(
    observations: np.ndarray, groups: np.ndarray, group_count: int
) -> np.ndarray:
    values = np.full(group_count, np.nan, dtype=float)
    for group in np.unique(groups):
        values[group] = float(np.median(observations[groups == group]))
    return values


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


def gaze_coordinate_error_spatial_map(
    arrays_by_seed: list[dict[str, np.ndarray]], path: Path
) -> None:
    """Plot decoded gaze-coordinate error along walls and pillar surfaces."""
    lines, counts = make_surface_subsegments(resolution=0.5)
    factorized_by_seed = []
    conjunction_by_seed = []
    for arrays in arrays_by_seed:
        true_gaze_physical = restore_physical(arrays["true_xy"][:, 2:])
        groups = surface_spatial_groups(
            true_gaze_physical, arrays["true_surface"], counts
        )
        factorized_by_seed.append(
            grouped_medians(
                arrays["factorized_gaze_error"], groups, len(lines)
            )
        )
        conjunction_by_seed.append(
            grouped_medians(
                arrays["conjunction_gaze_error"], groups, len(lines)
            )
        )

    factorized = nanmean_where_available(np.stack(factorized_by_seed), axis=0)
    conjunction = nanmean_where_available(np.stack(conjunction_by_seed), axis=0)
    difference = factorized - conjunction
    shared_max = max(
        float(np.nanmax(np.concatenate([factorized, conjunction]))),
        np.finfo(float).eps,
    )
    difference_max = max(float(np.nanmax(np.abs(difference))), 1e-12)
    shared_norm = Normalize(vmin=0.0, vmax=shared_max)
    difference_norm = TwoSlopeNorm(
        vmin=-difference_max, vcenter=0.0, vmax=difference_max
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7), dpi=180)
    titles = [
        "Factorized discrete",
        "Conjunction trained",
        "Error difference\n(factorized minus conjunction)",
    ]
    for ax, title in zip(axes, titles):
        draw_arena_background(ax)
        ax.set_title(title)
    first = add_colored_segments(
        axes[0], lines, factorized, "magma", shared_norm
    )
    add_colored_segments(axes[1], lines, conjunction, "magma", shared_norm)
    difference_plot = add_colored_segments(
        axes[2], lines, difference, "RdBu_r", difference_norm
    )
    shared_bar = fig.colorbar(first, ax=axes[:2], fraction=0.025, pad=0.025)
    shared_bar.set_label("Median decoded gaze error (grid squares)")
    difference_bar = fig.colorbar(
        difference_plot, ax=axes[2], fraction=0.046, pad=0.04
    )
    difference_bar.set_label("Positive = conjunction has lower error")
    fig.suptitle(
        "Gaze-coordinate error across arena surfaces "
        f"(mean of {len(arrays_by_seed)} seed medians)",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.12, top=0.84, wspace=0.28)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def diagnostic_bars(
    rows: list[dict], path: Path,
    unobserved_label: str = "Unsupported joint",
) -> None:
    models = [
        ("Factorized discrete", "factorized", "#3976AF"),
        ("Conjunction trained", "conjunction", "#D64B40"),
    ]
    categories = [
        ("Both correct", "both_correct"),
        ("Place only", "place_only_correct"),
        ("Surface only", "surface_only_correct"),
        ("Neither", "neither_correct"),
        (unobserved_label, "unsupported_joint_rate"),
    ]
    x = np.arange(len(categories), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=170)
    for model_index, (label, prefix, color) in enumerate(models):
        positions = x + (model_index - 0.5) * width
        means, errors = [], []
        all_values = []
        for _, suffix in categories:
            values = np.asarray([row[f"{prefix}_{suffix}"] for row in rows])
            mean, error = mean_ci(values)
            means.append(mean)
            errors.append(error)
            all_values.append(values)
        ax.bar(
            positions,
            means,
            width=width * 0.92,
            yerr=errors,
            capsize=4,
            color=color,
            alpha=0.86,
            label=label,
        )
        for position, values in zip(positions, all_values):
            jitter = np.linspace(-0.035, 0.035, len(values))
            ax.scatter(
                position + jitter,
                values,
                color="black",
                alpha=0.7,
                s=18,
                zorder=3,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in categories])
    ax.set_ylabel("Proportion of held-out timesteps")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Joint-error decomposition and unsupported combinations"
        if unobserved_label == "Unsupported joint"
        else "Joint-error decomposition and test-unobserved combinations"
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def training_plot(history: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), dpi=170)
    model_specs = [
        ("Factorized discrete", "factorized", "#3976AF"),
        ("Conjunction trained", "conjunction", "#D64B40"),
    ]
    for label, prefix, color in model_specs:
        seed_series = []
        for seed in sorted({row["seed"] for row in history}):
            selected = [row for row in history if row["seed"] == seed]
            updates = np.asarray([row["update"] for row in selected])
            accuracy = np.asarray(
                [row[f"{prefix}_joint_accuracy"] for row in selected]
            )
            loss = np.asarray([row[f"{prefix}_loss"] for row in selected])
            axes[0].plot(updates, accuracy, color=color, alpha=0.18, linewidth=1)
            axes[1].plot(updates, loss, color=color, alpha=0.18, linewidth=1)
            seed_series.append((accuracy, loss))
        accuracy_mean = np.mean([item[0] for item in seed_series], axis=0)
        loss_mean = np.mean([item[1] for item in seed_series], axis=0)
        axes[0].plot(updates, accuracy_mean, color=color, linewidth=2.4, label=label)
        axes[1].plot(updates, loss_mean, color=color, linewidth=2.4, label=label)
    axes[0].set_title("Exact conjunction training accuracy")
    axes[0].set_xlabel("Training update")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[1].set_title("Training objective")
    axes[1].set_xlabel("Training update")
    axes[1].set_ylabel("Cross-entropy loss")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--updates", type=int, default=base.TRAIN_UPDATES)
    parser.add_argument("--eval-trials", type=int, default=base.EVAL_TRIALS)
    parser.add_argument("--batch-size", type=int, default=base.BATCH_SIZE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(base.DEFAULT_SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "factorized_results",
    )
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.updates = 2
        args.eval_trials = args.batch_size
        args.seeds = [1]
        args.log_every = 1
        if args.output_dir.name == "factorized_results":
            args.output_dir = args.output_dir.parent / "factorized_quick_results"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = base.choose_device(args.device)
    print(f"Device: {device}", flush=True)
    print("Initializing Julia RNNTrialStructures bridge...", flush=True)
    bridge = base.JuliaTrialBridge()
    print("Julia bridge ready.", flush=True)

    # Parameter counts are architecture-level constants, so instantiate on CPU.
    factorized_parameter_count = count_parameters(FactorizedDiscreteRNN())
    conjunction_parameter_count = count_parameters(base.ConjunctionRNN())
    config = {
        "seeds": args.seeds,
        "comparison": "factorized_discrete_vs_joint_conjunction",
        "input_channels": ["distance", "movement", "texture"],
        "input_dim": base.INPUT_DIM,
        "hidden_dim": base.HIDDEN_DIM,
        "factorized_outputs": {"place": PLACE_BINS, "surface": SURFACE_BINS},
        "conjunction_outputs": JOINT_BINS,
        "factorized_loss": "cross_entropy(place) + cross_entropy(surface)",
        "conjunction_loss": "cross_entropy(joint)",
        "min_steps": base.MIN_STEPS,
        "max_steps": base.MAX_STEPS,
        "batch_size": args.batch_size,
        "train_updates": args.updates,
        "eval_trials": args.eval_trials,
        "learning_rate": base.LEARNING_RATE,
        "gradient_clip": base.GRAD_CLIP,
        "device": str(device),
        "evaluation_noise_std": 0.0,
        "factorized_parameter_count": factorized_parameter_count,
        "conjunction_parameter_count": conjunction_parameter_count,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    rows: list[dict] = []
    history: list[dict] = []
    arrays_by_seed: list[dict[str, np.ndarray]] = []
    raw_dir = args.output_dir / "raw_evaluation"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===", flush=True)
        factorized, conjunction, seed_history = train_pair(
            bridge,
            seed,
            args.updates,
            args.batch_size,
            device,
            args.output_dir,
            args.log_every,
        )
        history.extend(seed_history)
        metrics, arrays = evaluate_pair(
            bridge,
            factorized,
            conjunction,
            seed,
            args.eval_trials,
            args.batch_size,
            device,
        )
        rows.append(metrics)
        arrays_by_seed.append(arrays)
        np.savez_compressed(raw_dir / f"seed_{seed}.npz", **arrays)
        print(
            f"seed={seed} held-out: "
            f"factorized joint={100 * metrics['factorized_joint_accuracy']:.2f}% | "
            f"conjunction joint={100 * metrics['conjunction_joint_accuracy']:.2f}% | "
            f"unsupported={100 * metrics['factorized_unsupported_joint_rate']:.2f}%/"
            f"{100 * metrics['conjunction_unsupported_joint_rate']:.2f}%",
            flush=True,
        )

    write_csv(args.output_dir / "metrics_by_seed.csv", rows)
    write_csv(args.output_dir / "training_history.csv", history)
    summary = {
        "n_seeds": len(rows),
        "metrics_mean": {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
            if key not in {"seed"}
        },
        "metrics_seed_sd": {
            key: float(np.std([row[key] for row in rows], ddof=1))
            if len(rows) > 1
            else 0.0
            for key in rows[0]
            if key not in {"seed"}
        },
        "parameter_counts": {
            "factorized": factorized_parameter_count,
            "conjunction": conjunction_parameter_count,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    grouped_accuracy_plot(rows, args.output_dir / "overall_accuracy.png")
    dumbbell_plot(rows, args.output_dir / "paired_accuracy_dumbbell.png")
    place_error_map(arrays_by_seed, args.output_dir / "place_bin_error_map.png")
    surface_error_map(arrays_by_seed, args.output_dir / "surface_bin_error_map.png")
    position_coordinate_error_map(
        arrays_by_seed, args.output_dir / "position_coordinate_error_map.png"
    )
    gaze_coordinate_error_spatial_map(
        arrays_by_seed,
        args.output_dir / "gaze_coordinate_error_spatial_map.png",
    )
    diagnostic_bars(rows, args.output_dir / "joint_error_diagnostics.png")
    training_plot(history, args.output_dir / "training_comparison.png")

    print(f"\nResults written to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

"""Paired comparison of continuous and conjunction-trained navigation RNNs.

Both models receive the same 36-channel inputs and exactly the same generated
training/evaluation trials.  They differ only in their output head and loss:

* ContinuousRNN: normalized (position_x, position_y, gaze_x, gaze_y), MSE.
* ConjunctionRNN: 21 x 24 place-surface classes, cross entropy.

The script produces three primary figures:
  1. overall_accuracy.png
  2. position_error.png
  3. gaze_error.png

Import juliacall before torch.  Importing torch first can crash some platforms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import juliacall

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_DIM = 36
HIDDEN_DIM = 256
PLACE_BINS = 21
SURFACE_BINS = 24
OUTPUT_DIM = PLACE_BINS * SURFACE_BINS
CONTINUOUS_DIM = 4

MIN_STEPS = 80
MAX_STEPS = 120
BATCH_SIZE = 64
TRAIN_UPDATES = 3000
EVAL_TRIALS = 4096
LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0
BIN_SIZE = 2.0
WALL_BIN_SIZE = 5.0
GRID_SQUARE_NORMALIZED = 0.08
DT = 20.0
FOV = math.pi / 3
HEAD_DIRECTION_STEP = math.pi / 3
P_STAY = 1 / 3
P_HD = 0.8
DEFAULT_SEEDS = (1, 2, 3, 4, 5)


class JuliaTrialBridge:
    """Thin bridge around the installed RNNTrialStructures Julia package."""

    def __init__(self) -> None:
        self.jl = juliacall.newmodule("ControlledObjectiveComparison")
        self.jl.seval("using RNNTrialStructures")
        loaded_path = str(self.jl.seval("pathof(RNNTrialStructures)"))
        print(f"RNNTrialStructures source: {loaded_path}", flush=True)
        # Perform binning in one Julia call rather than once per timestep.
        self.jl.seval(
            """
            function codex_make_navigation_generator(min_steps, max_steps,
                                                      ntrials, rseed, dt, fov,
                                                      hd_step, p_stay, p_hd,
                                                      binsize, binsize_wall)
                arena = RNNTrialStructures.MazeArena()
                angles = collect(range(Float32(0), stop=Float32(2*pi), length=16))
                apref = RNNTrialStructures.AngularPreference(
                    angles, Float32(5), Float32(0.8))
                inputs = Symbol[:distance, :movement, :texture]
                outputs = Symbol[:position, :gaze, :conjunction]
                trialstruct = RNNTrialStructures.NavigationTrial(
                    Int64(min_steps), Int64(max_steps), inputs, outputs,
                    arena, apref)
                raw_generator = RNNTrialStructures.generate_trials(
                    trialstruct, Int64(ntrials), Float32(dt);
                    fov=Float32(fov), hd_step=Float32(hd_step),
                    p_stay=Float32(p_stay), p_hd=Float32(p_hd),
                    binsize=Float32(binsize),
                    binsize_wall=Float32(binsize_wall),
                    rseed=Int64(rseed))

                # Some installed package versions contain the historical bug
                # where conjunction position used the y-bin (1:5) rather than
                # the valid floor-bin index (1:21). Preserve the generated
                # surface label, reconstruct the correct place label from the
                # continuous position, and return a zero-based class matrix.
                n_surface = RNNTrialStructures.num_surface_bins(
                    arena; binsize=Float32(binsize),
                    binsize_wall=Float32(binsize_wall))
                function corrected_data_provider()
                    x, y, w = raw_generator()
                    nt = size(y, 2)
                    nb = size(y, 3)
                    target = fill(Int64(-1), nt, nb)
                    for b in 1:nb, t in 1:nt
                        if w[1, t, b] > 0.5f0
                            raw_class0 = argmax(@view y[5:end, t, b]) - 1
                            surface0 = mod(raw_class0, n_surface)
                            px = Float32(10) * (y[1, t, b] - 0.05f0) / 0.8f0
                            py = Float32(10) * (y[2, t, b] - 0.05f0) / 0.8f0
                            _, _, place1 = RNNTrialStructures.assign_bin(
                                px, py, arena; binsize=Float32(binsize))
                            target[t, b] = (place1 - 1) * n_surface + surface0
                        end
                    end
                    return x, y[1:4, :, :], w[1:4, :, :], target
                end
                return corrected_data_provider
            end

            function codex_assign_navigation_bins(xs, ys, gxs, gys,
                                                   binsize, binsize_wall)
                arena = RNNTrialStructures.MazeArena()
                n = length(xs)
                place = zeros(Int64, n)
                surface = zeros(Int64, n)
                for k in 1:n
                    _, _, place[k] = RNNTrialStructures.assign_bin(
                        Float32(xs[k]), Float32(ys[k]), arena;
                        binsize=Float32(binsize))
                    _, _, surface[k] = RNNTrialStructures.assign_surface_bin(
                        Float32(gxs[k]), Float32(gys[k]), arena;
                        binsize=Float32(binsize),
                        binsize_wall=Float32(binsize_wall))
                end
                return place, surface
            end
            """
        )

    def make_generator(self, ntrials: int, rseed: int):
        return self.jl.codex_make_navigation_generator(
            MIN_STEPS,
            MAX_STEPS,
            ntrials,
            rseed,
            DT,
            FOV,
            HEAD_DIRECTION_STEP,
            P_STAY,
            P_HD,
            BIN_SIZE,
            WALL_BIN_SIZE,
        )

    def assign_bins(
        self,
        position_physical: np.ndarray,
        gaze_physical: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(position_physical, dtype=np.float32).reshape(-1, 2)
        gaze = np.asarray(gaze_physical, dtype=np.float32).reshape(-1, 2)
        place, surface = self.jl.codex_assign_navigation_bins(
            position[:, 0],
            position[:, 1],
            gaze[:, 0],
            gaze[:, 1],
            BIN_SIZE,
            WALL_BIN_SIZE,
        )
        # Julia labels are one-based; all evaluation arrays are zero-based.
        return np.asarray(place, dtype=np.int64) - 1, np.asarray(
            surface, dtype=np.int64
        ) - 1


class ContinuousRNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            INPUT_DIM,
            HIDDEN_DIM,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.fc = nn.Linear(HIDDEN_DIM, CONTINUOUS_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(x)
        return torch.sigmoid(self.fc(hidden))


class ConjunctionRNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            INPUT_DIM,
            HIDDEN_DIM,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.fc = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(x)
        return self.fc(hidden)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unpack_batch(raw_batch, device: torch.device):
    x_np, y_np, w_np, target_np = raw_batch
    x_np = np.asarray(x_np, dtype=np.float32)
    y_np = np.asarray(y_np, dtype=np.float32)
    w_np = np.asarray(w_np, dtype=np.float32)

    if x_np.ndim != 3 or x_np.shape[0] != INPUT_DIM:
        raise RuntimeError(f"Expected input shape (36,T,B), received {x_np.shape}")
    if y_np.ndim != 3 or y_np.shape[0] != CONTINUOUS_DIM:
        raise RuntimeError(
            f"Expected target shape ({CONTINUOUS_DIM},T,B), received {y_np.shape}"
        )

    x = torch.from_numpy(x_np).permute(2, 1, 0).contiguous().to(device)
    y = torch.from_numpy(y_np).permute(2, 1, 0).contiguous().to(device)
    w = torch.from_numpy(w_np).permute(2, 1, 0).contiguous().to(device)
    target = torch.from_numpy(np.asarray(target_np, dtype=np.int64)).T.contiguous().to(device)

    continuous_target = y
    conjunction_target = target
    valid = w[..., 0] > 0.5

    valid_targets = conjunction_target[valid]
    if valid_targets.numel() == 0:
        raise RuntimeError("Batch contains no valid timesteps")
    if valid_targets.min() < 0 or valid_targets.max() >= OUTPUT_DIM:
        raise RuntimeError("Corrected conjunction targets fall outside 0:503")
    if torch.unique(valid_targets // SURFACE_BINS).numel() <= 5:
        raise RuntimeError("Corrected targets still contain the five-place labeling bug")

    return x, continuous_target, conjunction_target, valid


def continuous_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    # Position and gaze each contribute one mean term, giving them equal weight.
    position = ((prediction[..., :2] - target[..., :2]) ** 2)[valid].mean()
    gaze = ((prediction[..., 2:] - target[..., 2:]) ** 2)[valid].mean()
    return position + gaze


def train_pair(
    bridge: JuliaTrialBridge,
    seed: int,
    updates: int,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    log_every: int,
) -> tuple[ContinuousRNN, ConjunctionRNN, list[dict]]:
    set_seed(seed)
    generator = bridge.make_generator(batch_size, rseed=seed)

    continuous_model = ContinuousRNN().to(device)
    conjunction_model = ConjunctionRNN().to(device)
    # Match the starting recurrent representation exactly within each seed.
    conjunction_model.rnn.load_state_dict(continuous_model.rnn.state_dict())

    continuous_optimizer = torch.optim.Adam(
        continuous_model.parameters(), lr=LEARNING_RATE
    )
    conjunction_optimizer = torch.optim.Adam(
        conjunction_model.parameters(), lr=LEARNING_RATE
    )

    history: list[dict] = []
    start = time.time()
    for update in range(1, updates + 1):
        x, continuous_target, conjunction_target, valid = unpack_batch(
            generator(), device
        )

        continuous_model.train()
        continuous_optimizer.zero_grad(set_to_none=True)
        continuous_prediction = continuous_model(x)
        loss_continuous = continuous_loss(
            continuous_prediction, continuous_target, valid
        )
        loss_continuous.backward()
        continuous_norm = torch.nn.utils.clip_grad_norm_(
            continuous_model.parameters(), GRAD_CLIP
        )
        continuous_optimizer.step()

        conjunction_model.train()
        conjunction_optimizer.zero_grad(set_to_none=True)
        conjunction_logits = conjunction_model(x)
        loss_conjunction = F.cross_entropy(
            conjunction_logits[valid], conjunction_target[valid]
        )
        loss_conjunction.backward()
        conjunction_norm = torch.nn.utils.clip_grad_norm_(
            conjunction_model.parameters(), GRAD_CLIP
        )
        conjunction_optimizer.step()

        should_log = update == 1 or update == updates or update % log_every == 0
        if should_log:
            with torch.no_grad():
                cont_class_accuracy = float("nan")
                conj_accuracy = (
                    conjunction_logits[valid].argmax(dim=-1)
                    == conjunction_target[valid]
                ).float().mean().item()
            row = {
                "seed": seed,
                "update": update,
                "continuous_loss": loss_continuous.item(),
                "conjunction_loss": loss_conjunction.item(),
                "conjunction_training_accuracy": conj_accuracy,
                "continuous_gradient_norm_before_clip": float(continuous_norm),
                "conjunction_gradient_norm_before_clip": float(conjunction_norm),
                "continuous_was_clipped": int(float(continuous_norm) > GRAD_CLIP),
                "conjunction_was_clipped": int(float(conjunction_norm) > GRAD_CLIP),
                "elapsed_seconds": time.time() - start,
            }
            history.append(row)
            print(
                f"seed={seed} update={update}/{updates} "
                f"continuous_mse={loss_continuous.item():.5f} "
                f"conjunction_ce={loss_conjunction.item():.5f} "
                f"conjunction_acc={100 * conj_accuracy:.2f}%",
                flush=True,
            )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    common_metadata = {
        "seed": seed,
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "train_updates": updates,
        "batch_size": batch_size,
        "learning_rate": LEARNING_RATE,
        "gradient_clip": GRAD_CLIP,
        "min_steps": MIN_STEPS,
        "max_steps": MAX_STEPS,
    }
    torch.save(
        {
            **common_metadata,
            "objective": "continuous_position_gaze",
            "model_state_dict": continuous_model.state_dict(),
        },
        checkpoint_dir / f"seed_{seed}_continuous.pt",
    )
    torch.save(
        {
            **common_metadata,
            "objective": "place_surface_conjunction",
            "model_state_dict": conjunction_model.state_dict(),
        },
        checkpoint_dir / f"seed_{seed}_conjunction.pt",
    )
    return continuous_model, conjunction_model, history


def restore_physical(normalized: np.ndarray) -> np.ndarray:
    return 10.0 * (normalized - 0.05) / 0.8


def normalized_position_centres() -> np.ndarray:
    # Same x-major, then y-major ordering used by Julia assign_bin and floor_grid.
    pillar_cells = {(1, 1), (3, 1), (1, 3), (3, 3)}
    physical = []
    for xbin in range(5):
        for ybin in range(5):
            if (xbin, ybin) not in pillar_cells:
                physical.append(((xbin + 0.5) * BIN_SIZE, (ybin + 0.5) * BIN_SIZE))
    values = np.asarray(physical, dtype=np.float32)
    if values.shape != (PLACE_BINS, 2):
        raise AssertionError(values.shape)
    return 0.05 + 0.8 * values / 10.0


def normalized_surface_centres() -> np.ndarray:
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    # Outer walls in the exact order used by assign_surface_bin.
    corners = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    for start, end in zip(corners, corners[1:] + corners[:1]):
        p1 = np.asarray(start, dtype=float)
        p2 = np.asarray(end, dtype=float)
        direction = (p2 - p1) / 2
        segments.append((tuple(p1), tuple(p1 + direction)))
        segments.append((tuple(p1 + direction), tuple(p2)))

    # Pillars are ordered exactly as in MazeArena().
    pillar_bounds = [(2, 2, 4, 4), (6, 2, 8, 4), (6, 6, 8, 8), (2, 6, 4, 8)]
    for xmin, ymin, xmax, ymax in pillar_bounds:
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        segments.extend(zip(points, points[1:] + points[:1]))

    centres = np.asarray(
        [
            ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            for a, b in segments
        ],
        dtype=np.float32,
    )
    if centres.shape != (SURFACE_BINS, 2):
        raise AssertionError(centres.shape)
    return 0.05 + 0.8 * centres / 10.0


@torch.no_grad()
def evaluate_pair(
    bridge: JuliaTrialBridge,
    continuous_model: ContinuousRNN,
    conjunction_model: ConjunctionRNN,
    seed: int,
    eval_trials: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict, dict[str, np.ndarray]]:
    if eval_trials % batch_size:
        raise ValueError("eval_trials must be divisible by batch_size")

    generator = bridge.make_generator(batch_size, rseed=100_000 + seed)
    continuous_model.eval()
    conjunction_model.eval()

    true_xy_parts = []
    true_class_parts = []
    continuous_xy_parts = []
    conjunction_class_parts = []

    for _ in range(eval_trials // batch_size):
        x, continuous_target, conjunction_target, valid = unpack_batch(
            generator(), device
        )
        continuous_prediction = continuous_model(x)
        conjunction_prediction = conjunction_model(x).argmax(dim=-1)

        true_xy_parts.append(continuous_target[valid].cpu().numpy())
        true_class_parts.append(conjunction_target[valid].cpu().numpy())
        continuous_xy_parts.append(continuous_prediction[valid].cpu().numpy())
        conjunction_class_parts.append(conjunction_prediction[valid].cpu().numpy())

    true_xy = np.concatenate(true_xy_parts)
    true_class = np.concatenate(true_class_parts)
    continuous_xy = np.concatenate(continuous_xy_parts)
    conjunction_class = np.concatenate(conjunction_class_parts)

    true_place = true_class // SURFACE_BINS
    true_surface = true_class % SURFACE_BINS
    conjunction_place = conjunction_class // SURFACE_BINS
    conjunction_surface = conjunction_class % SURFACE_BINS

    continuous_place, continuous_surface = bridge.assign_bins(
        restore_physical(continuous_xy[:, :2]),
        restore_physical(continuous_xy[:, 2:]),
    )
    truth_place_from_xy, truth_surface_from_xy = bridge.assign_bins(
        restore_physical(true_xy[:, :2]),
        restore_physical(true_xy[:, 2:]),
    )

    place_match = truth_place_from_xy == true_place
    surface_match = truth_surface_from_xy == true_surface
    place_mapping_accuracy = float(np.mean(place_match))
    surface_mapping_accuracy = float(np.mean(surface_match))
    truth_mapping_accuracy = float(np.mean(place_match & surface_match))
    # Position must be exact. A small number of gaze points lie exactly on
    # surface-bin corners; Float32 normalize/restore can move those ties to the
    # adjacent surface, so surface reconstruction is allowed up to 1% ambiguity.
    if place_mapping_accuracy < 0.99999 or surface_mapping_accuracy < 0.99:
        mismatches = int(np.sum(~(place_match & surface_match)))
        place_accuracy = float(np.mean(place_match))
        surface_accuracy = float(np.mean(surface_match))
        bad = np.flatnonzero(~(place_match & surface_match))[:5]
        examples = [
            {
                "true_class": int(true_class[k]),
                "decoded_place": int(true_place[k]),
                "xy_place": int(truth_place_from_xy[k]),
                "decoded_surface": int(true_surface[k]),
                "xy_surface": int(truth_surface_from_xy[k]),
                "position_xy": true_xy[k, :2].tolist(),
                "gaze_xy": true_xy[k, 2:].tolist(),
            }
            for k in bad
        ]
        raise RuntimeError(
            "Continuous truth does not reconstruct conjunction truth: "
            f"{mismatches}/{len(true_class)} mismatches "
            f"({100 * truth_mapping_accuracy:.4f}% joint mapping accuracy; "
            f"place={100 * place_accuracy:.4f}%, "
            f"surface={100 * surface_accuracy:.4f}%). "
            f"Examples: {examples}"
        )

    position_centres = normalized_position_centres()
    surface_centres = normalized_surface_centres()

    continuous_position_error = np.linalg.norm(
        continuous_xy[:, :2] - true_xy[:, :2], axis=1
    ) / GRID_SQUARE_NORMALIZED
    continuous_gaze_error = np.linalg.norm(
        continuous_xy[:, 2:] - true_xy[:, 2:], axis=1
    ) / GRID_SQUARE_NORMALIZED

    conjunction_position_error = np.linalg.norm(
        position_centres[conjunction_place] - true_xy[:, :2], axis=1
    ) / GRID_SQUARE_NORMALIZED
    conjunction_gaze_error = np.linalg.norm(
        surface_centres[conjunction_surface] - true_xy[:, 2:], axis=1
    ) / GRID_SQUARE_NORMALIZED

    ideal_position_error = np.linalg.norm(
        position_centres[true_place] - true_xy[:, :2], axis=1
    ) / GRID_SQUARE_NORMALIZED
    ideal_gaze_error = np.linalg.norm(
        surface_centres[true_surface] - true_xy[:, 2:], axis=1
    ) / GRID_SQUARE_NORMALIZED

    continuous_valid_class = (
        (continuous_place >= 0)
        & (continuous_place < PLACE_BINS)
        & (continuous_surface >= 0)
        & (continuous_surface < SURFACE_BINS)
    )
    continuous_class = np.full_like(true_class, -1)
    continuous_class[continuous_valid_class] = (
        continuous_place[continuous_valid_class] * SURFACE_BINS
        + continuous_surface[continuous_valid_class]
    )

    row = {
        "seed": seed,
        "valid_timesteps": int(len(true_class)),
        "supported_classes": int(len(np.unique(true_class))),
        "truth_mapping_accuracy": truth_mapping_accuracy,
        "truth_place_mapping_accuracy": place_mapping_accuracy,
        "truth_surface_mapping_accuracy": surface_mapping_accuracy,
        "continuous_invalid_prediction_rate": float(
            np.mean(~continuous_valid_class)
        ),
        "continuous_position_accuracy": float(
            np.mean(continuous_place == true_place)
        ),
        "conjunction_position_accuracy": float(
            np.mean(conjunction_place == true_place)
        ),
        "continuous_surface_accuracy": float(
            np.mean(continuous_surface == true_surface)
        ),
        "conjunction_surface_accuracy": float(
            np.mean(conjunction_surface == true_surface)
        ),
        "continuous_conjunction_accuracy": float(
            np.mean(continuous_class == true_class)
        ),
        "conjunction_conjunction_accuracy": float(
            np.mean(conjunction_class == true_class)
        ),
        "continuous_position_error_median": float(
            np.median(continuous_position_error)
        ),
        "conjunction_position_error_median": float(
            np.median(conjunction_position_error)
        ),
        "ideal_position_error_median": float(np.median(ideal_position_error)),
        "continuous_gaze_error_median": float(np.median(continuous_gaze_error)),
        "conjunction_gaze_error_median": float(np.median(conjunction_gaze_error)),
        "ideal_gaze_error_median": float(np.median(ideal_gaze_error)),
        "continuous_position_error_mean": float(np.mean(continuous_position_error)),
        "conjunction_position_error_mean": float(np.mean(conjunction_position_error)),
        "ideal_position_error_mean": float(np.mean(ideal_position_error)),
        "continuous_gaze_error_mean": float(np.mean(continuous_gaze_error)),
        "conjunction_gaze_error_mean": float(np.mean(conjunction_gaze_error)),
        "ideal_gaze_error_mean": float(np.mean(ideal_gaze_error)),
    }
    arrays = {
        "true_xy": true_xy,
        "true_class": true_class,
        "continuous_xy": continuous_xy,
        "continuous_class": continuous_class,
        "conjunction_class": conjunction_class,
        "continuous_position_error": continuous_position_error,
        "conjunction_position_error": conjunction_position_error,
        "ideal_position_error": ideal_position_error,
        "continuous_gaze_error": continuous_gaze_error,
        "conjunction_gaze_error": conjunction_gaze_error,
        "ideal_gaze_error": ideal_gaze_error,
    }
    return row, arrays


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    return mean, float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


def grouped_seed_plot(
    rows: list[dict],
    groups: list[tuple[str, list[tuple[str, str]]]],
    ylabel: str,
    title: str,
    path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    colors = ["#3976AF", "#D64B40", "#6F7D3C"]
    group_x = np.arange(len(groups), dtype=float)
    n_series = len(groups[0][1])
    width = 0.72 / n_series

    for series_index in range(n_series):
        values_by_group = []
        labels = []
        for _, series in groups:
            label, key = series[series_index]
            labels.append(label)
            values_by_group.append(np.asarray([row[key] for row in rows], dtype=float))

        means, errors = zip(*(mean_ci(values) for values in values_by_group))
        positions = group_x + (series_index - (n_series - 1) / 2) * width
        ax.bar(
            positions,
            means,
            width=width * 0.9,
            yerr=errors,
            capsize=4,
            color=colors[series_index],
            alpha=0.86,
            label=labels[0],
        )
        for position, values in zip(positions, values_by_group):
            jitter = np.linspace(-width * 0.18, width * 0.18, len(values))
            ax.scatter(
                position + jitter,
                values,
                color="black",
                s=18,
                zorder=3,
                alpha=0.75,
            )

    ax.set_xticks(group_x)
    ax.set_xticklabels([label for label, _ in groups])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def make_plots(rows: list[dict], output_dir: Path) -> None:
    accuracy_groups = [
        (
            "Position bin",
            [
                ("Continuous-output model", "continuous_position_accuracy"),
                ("Conjunction-trained model", "conjunction_position_accuracy"),
            ],
        ),
        (
            "Surface bin",
            [
                ("Continuous-output model", "continuous_surface_accuracy"),
                ("Conjunction-trained model", "conjunction_surface_accuracy"),
            ],
        ),
        (
            "Exact conjunction",
            [
                ("Continuous-output model", "continuous_conjunction_accuracy"),
                ("Conjunction-trained model", "conjunction_conjunction_accuracy"),
            ],
        ),
    ]
    grouped_seed_plot(
        rows,
        accuracy_groups,
        "Held-out accuracy",
        "Controlled objective comparison",
        output_dir / "overall_accuracy.png",
        ylim=(0, 1),
    )

    position_groups = [
        (
            "Position error",
            [
                ("Continuous-output model", "continuous_position_error_median"),
                ("Conjunction-trained model", "conjunction_position_error_median"),
                ("Correct-bin baseline", "ideal_position_error_median"),
            ],
        )
    ]
    grouped_seed_plot(
        rows,
        position_groups,
        "Median position error (grid squares)",
        "Position error on matched held-out trials",
        output_dir / "position_error.png",
    )

    gaze_groups = [
        (
            "Gaze-intersection error",
            [
                ("Continuous-output model", "continuous_gaze_error_median"),
                ("Conjunction-trained model", "conjunction_gaze_error_median"),
                ("Correct-bin baseline", "ideal_gaze_error_median"),
            ],
        )
    ]
    grouped_seed_plot(
        rows,
        gaze_groups,
        "Median gaze error (grid squares)",
        "Gaze-intersection error on matched held-out trials",
        output_dir / "gaze_error.png",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="2 updates, 64 eval trials, seed 1")
    parser.add_argument("--updates", type=int, default=TRAIN_UPDATES)
    parser.add_argument("--eval-trials", type=int, default=EVAL_TRIALS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
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
        if args.output_dir.name == "results":
            args.output_dir = args.output_dir.parent / "quick_results"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"Device: {device}", flush=True)
    print("Initializing Julia RNNTrialStructures bridge...", flush=True)
    bridge = JuliaTrialBridge()
    print("Julia bridge ready.", flush=True)

    config = {
        "seeds": args.seeds,
        "input_channels": ["movement", "distance", "texture"],
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "min_steps": MIN_STEPS,
        "max_steps": MAX_STEPS,
        "batch_size": args.batch_size,
        "train_updates": args.updates,
        "eval_trials": args.eval_trials,
        "learning_rate": LEARNING_RATE,
        "gradient_clip": GRAD_CLIP,
        "bin_size": BIN_SIZE,
        "wall_bin_size": WALL_BIN_SIZE,
        "device": str(device),
        "evaluation_noise_std": 0.0,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    metric_rows: list[dict] = []
    history_rows: list[dict] = []
    raw_dir = args.output_dir / "raw_evaluation"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===", flush=True)
        continuous_model, conjunction_model, history = train_pair(
            bridge,
            seed,
            args.updates,
            args.batch_size,
            device,
            args.output_dir,
            args.log_every,
        )
        history_rows.extend(history)
        metrics, arrays = evaluate_pair(
            bridge,
            continuous_model,
            conjunction_model,
            seed,
            args.eval_trials,
            args.batch_size,
            device,
        )
        metric_rows.append(metrics)
        np.savez_compressed(raw_dir / f"seed_{seed}.npz", **arrays)
        # Persist after every seed so a long CPU run always leaves usable work.
        write_csv(args.output_dir / "metrics_by_seed.csv", metric_rows)
        write_csv(args.output_dir / "training_history.csv", history_rows)
        make_plots(metric_rows, args.output_dir)
        print(
            f"seed={seed} held-out: "
            f"continuous joint={100 * metrics['continuous_conjunction_accuracy']:.2f}% | "
            f"conjunction joint={100 * metrics['conjunction_conjunction_accuracy']:.2f}%",
            flush=True,
        )

    summary = {
        "n_seeds": len(metric_rows),
        "metrics_mean": {
            key: float(np.mean([row[key] for row in metric_rows]))
            for key in metric_rows[0]
            if key not in {"seed", "valid_timesteps", "supported_classes"}
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nResults written to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

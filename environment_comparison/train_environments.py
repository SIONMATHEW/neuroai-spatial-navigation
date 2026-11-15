"""Train paired factorized RNNs in open and occluded navigation arenas.

Only the arena changes.  Both networks use the same architecture, training
budget, optimizer, sensory-noise rule and initial recurrent weights.  The
Julia library supplies the versioned arena presets and navigation trials.

This script uses training and validation trajectories only.  The final test
and hidden-state probe trajectories are created later by
``evaluate_environments.py``.
"""

from __future__ import annotations

# Load the Julia bridge before PyTorch (more reliable on the cluster).
import RNNTrialStructures as rts

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ENVIRONMENTS = ("open", "occluded")
INPUTS = ("distance", "movement", "texture")
INPUT_DIM = 36
HIDDEN_UNITS = 256

# With the declared arena geometry and bin sizes, these are fixed label spaces.
LABEL_COUNTS = {
    "open": {"place": 25, "surface": 8},
    "occluded": {"place": 21, "surface": 24},
}

MIN_STEPS = 80
MAX_STEPS = 120
BIN_SIZE = 2.0
WALL_BIN_SIZE = 5.0
DT = 20.0
FOV = np.pi / 3
HEAD_DIRECTION_STEP = np.pi / 3
P_STAY = 1 / 3
P_HEAD_DIRECTION = 0.8

# Julia's canonical layout is movement[0:4], distance[4:20], texture[20:36].
# Noise is added only to the sensory distance and texture channels.
SENSORY_CHANNELS = slice(4, 36)


@dataclass(frozen=True)
class Protocol:
    updates: int = 4_000
    batch_size: int = 256
    validation_trials: int = 1_024
    validation_every: int = 50
    learning_rate: float = 1e-3
    gradient_clip: float = 3.0
    training_noise_sd: float = 0.2
    evaluation_noise_sd: float = 0.25


@dataclass
class Batch:
    inputs: torch.Tensor
    place: torch.Tensor
    surface: torch.Tensor
    joint: torch.Tensor
    valid: torch.Tensor


class FactorizedRNN(nn.Module):
    """A tanh RNN with separate categorical place and gaze-surface heads."""

    def __init__(self, core: nn.RNN, place_bins: int, surface_bins: int):
        super().__init__()
        self.rnn = copy.deepcopy(core)
        self.place_head = nn.Linear(HIDDEN_UNITS, place_bins)
        self.surface_head = nn.Linear(HIDDEN_UNITS, surface_bins)

    def forward(self, inputs: torch.Tensor):
        hidden, _ = self.rnn(inputs)
        return self.place_head(hidden), self.surface_head(hidden)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a GPU")
    return torch.device(requested)


def make_trialstruct(environment: str):
    if environment not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {environment}")
    config = rts.NavigationConfig(
        min_n_steps=MIN_STEPS,
        max_n_steps=MAX_STEPS,
        inputs=INPUTS,
        outputs=("conjunction",),
        arena=rts.ArenaConfig(preset=environment),
    )
    return rts.make_navigation_trialstruct(config)


def make_generator(trialstruct, ntrials: int, seed: int):
    return rts.get_batch_generator(
        trialstruct,
        ntrials,
        trajectory_seed=seed,
        fov=FOV,
        head_direction_step=HEAD_DIRECTION_STEP,
        p_stay=P_STAY,
        p_hd=P_HEAD_DIRECTION,
        dt=DT,
        binsize=BIN_SIZE,
        binsize_wall=WALL_BIN_SIZE,
        named=True,
    )


def convert_batch(raw: rts.NavigationBatch, environment: str) -> Batch:
    """Convert Julia arrays to PyTorch order and zero-based labels."""

    inputs = torch.from_numpy(np.array(raw.inputs, dtype=np.float32, copy=True))
    inputs = inputs.permute(2, 1, 0).contiguous()  # trial, time, feature
    if inputs.shape[-1] != INPUT_DIM:
        raise RuntimeError(f"Expected {INPUT_DIM} inputs; got {inputs.shape[-1]}")

    lengths = torch.from_numpy(np.array(raw.lengths, dtype=np.int64, copy=True))
    valid = torch.arange(inputs.shape[1])[None, :] < lengths[:, None]
    place = torch.from_numpy(np.array(raw.place_bin, dtype=np.int64, copy=True)).T - 1
    surface = torch.from_numpy(np.array(raw.surface_bin, dtype=np.int64, copy=True)).T - 1
    joint = torch.from_numpy(np.array(raw.joint_bin, dtype=np.int64, copy=True)).T - 1

    place_bins = LABEL_COUNTS[environment]["place"]
    surface_bins = LABEL_COUNTS[environment]["surface"]
    expected_joint = place_bins * surface_bins
    if int(raw.targets.shape[0]) != expected_joint:
        raise RuntimeError(
            f"{environment}: expected {expected_joint} output classes; "
            f"library returned {raw.targets.shape[0]}"
        )
    if not torch.equal(joint[valid], place[valid] * surface_bins + surface[valid]):
        raise RuntimeError(f"{environment}: inconsistent categorical labels")
    return Batch(inputs, place, surface, joint, valid)


def channel_statistics(batch: Batch) -> dict[str, torch.Tensor]:
    """Estimate noise scale from an independent training-calibration bank."""

    values = batch.inputs[batch.valid]
    standard_deviation = values.std(dim=0, unbiased=False)
    standard_deviation = torch.where(
        standard_deviation > 1e-6,
        standard_deviation,
        torch.zeros_like(standard_deviation),
    )
    return {
        "mean": values.mean(dim=0),
        "std": standard_deviation,
        "min": values.amin(dim=0),
        "max": values.amax(dim=0),
    }


def add_sensory_noise(
    batch: Batch,
    statistics: dict[str, torch.Tensor],
    relative_sd: float,
    generator: torch.Generator,
) -> Batch:
    """Add Gaussian sensory noise, expressed in channel-SD units."""

    if relative_sd == 0:
        return Batch(batch.inputs.clone(), batch.place, batch.surface, batch.joint, batch.valid)
    result = batch.inputs.clone()
    sensory = result[..., SENSORY_CHANNELS]
    random_values = torch.randn(sensory.shape, generator=generator, dtype=sensory.dtype)
    changed = sensory + random_values * statistics["std"][SENSORY_CHANNELS] * relative_sd
    changed = torch.maximum(changed, statistics["min"][SENSORY_CHANNELS])
    changed = torch.minimum(changed, statistics["max"][SENSORY_CHANNELS])
    result[..., SENSORY_CHANNELS] = torch.where(batch.valid[..., None], changed, sensory)
    return Batch(result, batch.place, batch.surface, batch.joint, batch.valid)


def make_paired_models(seed: int, device: torch.device) -> dict[str, FactorizedRNN]:
    """Create open/occluded models with identical recurrent parameters."""

    torch.manual_seed(10_000 + seed)
    core = nn.RNN(INPUT_DIM, HIDDEN_UNITS, nonlinearity="tanh", batch_first=True)
    models = {}
    for index, environment in enumerate(ENVIRONMENTS):
        counts = LABEL_COUNTS[environment]
        torch.manual_seed(20_000 + 100 * seed + index)
        models[environment] = FactorizedRNN(
            core, counts["place"], counts["surface"]
        ).to(device)

    open_state = models["open"].rnn.state_dict()
    occluded_state = models["occluded"].rnn.state_dict()
    if any(not torch.equal(open_state[key], occluded_state[key]) for key in open_state):
        raise RuntimeError("Paired recurrent initializations are not identical")
    return models


def model_loss(model: FactorizedRNN, batch: Batch, device: torch.device):
    place_logits, surface_logits = model(batch.inputs.to(device))
    valid = batch.valid.to(device)
    place_loss = F.cross_entropy(place_logits[valid], batch.place.to(device)[valid])
    surface_loss = F.cross_entropy(surface_logits[valid], batch.surface.to(device)[valid])
    return place_loss + surface_loss


@torch.inference_mode()
def accuracy(model: FactorizedRNN, batch: Batch, surface_bins: int, device) -> dict:
    model.eval()
    place_logits, surface_logits = model(batch.inputs.to(device))
    valid = batch.valid.to(device)
    true_place = batch.place.to(device)[valid]
    true_surface = batch.surface.to(device)[valid]
    true_joint = batch.joint.to(device)[valid]
    pred_place = place_logits[valid].argmax(-1)
    pred_surface = surface_logits[valid].argmax(-1)
    pred_joint = pred_place * surface_bins + pred_surface
    return {
        "place_accuracy": float((pred_place == true_place).float().mean()),
        "surface_accuracy": float((pred_surface == true_surface).float().mean()),
        "joint_accuracy": float((pred_joint == true_joint).float().mean()),
        "valid_timesteps": int(valid.sum()),
    }


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def recurrent_digest(model: FactorizedRNN) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.rnn.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path: Path, model, seed: int, update: int, environment: str, protocol):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "environment": environment,
            "objective": "factorized",
            "seed": seed,
            "update": update,
            "place_bins": LABEL_COUNTS[environment]["place"],
            "surface_bins": LABEL_COUNTS[environment]["surface"],
            "protocol": asdict(protocol),
            "model_state_dict": cpu_state(model),
        },
        path,
    )


def train_seed(seed: int, protocol: Protocol, output_root: Path, device, overwrite=False):
    seed_root = output_root / f"seed_{seed}"
    completed = seed_root / "completed.json"
    if completed.exists() and not overwrite:
        print(f"Seed {seed} is already complete; skipping", flush=True)
        return
    seed_root.mkdir(parents=True, exist_ok=True)
    set_determinism(seed)

    trialstructs = {environment: make_trialstruct(environment) for environment in ENVIRONMENTS}
    statistics, validation, training_generators, noise_generators = {}, {}, {}, {}
    for index, environment in enumerate(ENVIRONMENTS):
        calibration = convert_batch(
            make_generator(trialstructs[environment], 1_024, 100_000 + seed)(),
            environment,
        )
        statistics[environment] = channel_statistics(calibration)
        environment_root = seed_root / environment
        environment_root.mkdir(parents=True, exist_ok=True)
        np.savez(
            environment_root / "noise_statistics.npz",
            **{name: value.numpy() for name, value in statistics[environment].items()},
        )

        clean_validation = convert_batch(
            make_generator(
                trialstructs[environment], protocol.validation_trials, 1_000_000 + seed
            )(),
            environment,
        )
        validation_noise = torch.Generator().manual_seed(2_000_000 + seed)
        validation[environment] = add_sensory_noise(
            clean_validation,
            statistics[environment],
            protocol.training_noise_sd,
            validation_noise,
        )
        training_generators[environment] = make_generator(
            trialstructs[environment], protocol.batch_size, 10_000 + seed
        )
        # Matching seed identifiers provide the same noise protocol.  Shapes may
        # differ because trajectories interact with different geometry.
        noise_generators[environment] = torch.Generator().manual_seed(3_000_000 + seed)

    models = make_paired_models(seed, device)
    initial_digest = recurrent_digest(models["open"])
    optimizers = {
        environment: torch.optim.Adam(
            model.parameters(),
            lr=protocol.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        for environment, model in models.items()
    }
    torch.save(
        {"seed": seed, "rnn_state_dict": cpu_state(models["open"].rnn)},
        seed_root / "initial_core.pt",
    )

    best = {environment: {"joint_accuracy": -np.inf, "update": None} for environment in ENVIRONMENTS}
    history = []
    valid_timesteps_seen = {environment: 0 for environment in ENVIRONMENTS}
    start = time.perf_counter()

    for update in range(1, protocol.updates + 1):
        losses, gradient_norms = {}, {}
        for environment in ENVIRONMENTS:
            clean = convert_batch(training_generators[environment](), environment)
            batch = add_sensory_noise(
                clean,
                statistics[environment],
                protocol.training_noise_sd,
                noise_generators[environment],
            )
            valid_timesteps_seen[environment] += int(batch.valid.sum())
            model = models[environment]
            model.train()
            optimizers[environment].zero_grad(set_to_none=True)
            loss = model_loss(model, batch, device)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol.gradient_clip)
            optimizers[environment].step()
            losses[environment] = float(loss.detach())
            gradient_norms[environment] = float(norm)

        if update != 1 and update != protocol.updates and update % protocol.validation_every:
            continue

        row = {
            "seed": seed,
            "update": update,
            "training_trials_per_environment": update * protocol.batch_size,
            "elapsed_seconds": time.perf_counter() - start,
        }
        for environment in ENVIRONMENTS:
            metrics = accuracy(
                models[environment],
                validation[environment],
                LABEL_COUNTS[environment]["surface"],
                device,
            )
            row[f"{environment}_training_loss"] = losses[environment]
            row[f"{environment}_gradient_norm_before_clip"] = gradient_norms[environment]
            row[f"{environment}_training_valid_timesteps"] = valid_timesteps_seen[environment]
            for name, value in metrics.items():
                row[f"{environment}_validation_{name}"] = value
            if metrics["joint_accuracy"] > best[environment]["joint_accuracy"]:
                best[environment] = {
                    "joint_accuracy": metrics["joint_accuracy"],
                    "update": update,
                }
                save_checkpoint(
                    seed_root / environment / "best.pt",
                    models[environment], seed, update, environment, protocol,
                )
        history.append(row)
        write_csv(seed_root / "training_history.csv", history)
        print(
            f"seed={seed} update={update}/{protocol.updates} | "
            f"open={100 * row['open_validation_joint_accuracy']:.2f}% | "
            f"occluded={100 * row['occluded_validation_joint_accuracy']:.2f}%",
            flush=True,
        )

    for environment in ENVIRONMENTS:
        save_checkpoint(
            seed_root / environment / "final.pt",
            models[environment], seed, protocol.updates, environment, protocol,
        )

    manifest = {
        "seed": seed,
        "comparison": "open versus occluded environment",
        "objective": "factorized",
        "protocol": asdict(protocol),
        "environments": {
            "open": {
                "arena_preset": "open",
                "description": "four textured outer walls; no internal pillars",
                **LABEL_COUNTS["open"],
            },
            "occluded": {
                "arena_preset": "occluded",
                "description": "same walls plus four pillars with distinct textures",
                **LABEL_COUNTS["occluded"],
            },
        },
        "input_channel_layout": {
            "movement": [0, 4], "distance": [4, 20], "texture": [20, 36]
        },
        "hidden_units": HIDDEN_UNITS,
        "same_training_budget": True,
        "same_optimizer": True,
        "same_recurrent_initialization": True,
        "paired_stochastic_seed_identifiers": True,
        "initial_recurrent_sha256": initial_digest,
        "training_trajectory_seed": 10_000 + seed,
        "validation_trajectory_seed": 1_000_000 + seed,
        "best_validation_checkpoints": best,
        "optimizer": {
            "name": "Adam", "learning_rate": protocol.learning_rate,
            "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0.0,
            "gradient_clip_global_norm": protocol.gradient_clip,
            "scheduler": "none",
        },
        "hidden_state_initialization": "zeros at the start of every trajectory",
        "dropout": 0.0,
        "loss_reduction": "mean over all valid timesteps in the batch",
        "software": {
            "python": os.sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "RNNTrialStructures.py": package_version("RNNTrialStructures"),
        },
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    write_json(seed_root / "manifest.json", manifest)
    write_json(completed, {"seed": seed, "status": "complete"})
    print(f"Completed seed {seed}: {seed_root}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("environment_results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--updates", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-trials", type=int, default=1_024)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.seeds = args.seeds[:1]
        args.updates = 2
        args.batch_size = 8
        args.validation_trials = 32
        args.validation_every = 1
    protocol = Protocol(
        updates=args.updates,
        batch_size=args.batch_size,
        validation_trials=args.validation_trials,
        validation_every=args.validation_every,
    )
    device = choose_device(args.device)
    print(f"Device: {device}", flush=True)
    for seed in args.seeds:
        train_seed(seed, protocol, args.output_root, device, args.overwrite)


if __name__ == "__main__":
    main()

"""Hyperparameter sweep for the factorized-discrete navigation RNN.

This script tunes only on a fixed validation set.  It never touches the final
4,096-trial test set used by ``factorized_vs_conjunction.py``.

Example coarse sweep on one seed::

    python sweep_factorized.py --device cuda --seeds 1 \
      --learning-rates 0.001 0.0005 \
      --batch-sizes 64 128 256 \
      --gradient-clips 1 3 \
      --max-updates 8000 --val-trials 1024

After selecting a configuration, rerun that one configuration with five seeds.
"""

from __future__ import annotations

# Import this first: it initializes juliacall before torch, as required by the
# existing experiment code on the cluster.
import factorized_vs_conjunction as experiment

import argparse
import csv
from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


base = experiment.base


def make_model(objective: str) -> torch.nn.Module:
    if objective == "factorized":
        return experiment.FactorizedDiscreteRNN()
    if objective == "conjunction":
        return base.ConjunctionRNN()
    raise ValueError(f"Unknown objective: {objective}")


@dataclass(frozen=True)
class SweepConfig:
    learning_rate: float
    batch_size: int
    gradient_clip: float
    scheduler: str

    @property
    def identifier(self) -> str:
        def safe(value: float) -> str:
            return f"{value:g}".replace(".", "p").replace("-", "m")

        return (
            f"lr{safe(self.learning_rate)}_bs{self.batch_size}_"
            f"clip{safe(self.gradient_clip)}_{self.scheduler}"
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def validation_metrics(
    model: torch.nn.Module,
    x: torch.Tensor,
    true_joint: torch.Tensor,
    valid: torch.Tensor,
    chunk_size: int,
    objective: str = "factorized",
) -> dict[str, float]:
    model.eval()
    correct_place = 0
    correct_surface = 0
    correct_joint = 0
    valid_count = 0
    place_loss_sum = 0.0
    surface_loss_sum = 0.0
    joint_loss_sum = 0.0

    for start in range(0, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        chunk_x = x[start:stop]
        chunk_joint = true_joint[start:stop]
        chunk_valid = valid[start:stop]
        chunk_place = chunk_joint // experiment.SURFACE_BINS
        chunk_surface = chunk_joint % experiment.SURFACE_BINS

        selected_place = chunk_place[chunk_valid]
        selected_surface = chunk_surface[chunk_valid]
        n_valid = int(chunk_valid.sum().item())
        selected_joint = chunk_joint[chunk_valid]
        if objective == "factorized":
            place_logits, surface_logits = model(chunk_x)
            selected_place_logits = place_logits[chunk_valid]
            selected_surface_logits = surface_logits[chunk_valid]
            place_loss_sum += torch.nn.functional.cross_entropy(
                selected_place_logits, selected_place, reduction="sum"
            ).item()
            surface_loss_sum += torch.nn.functional.cross_entropy(
                selected_surface_logits, selected_surface, reduction="sum"
            ).item()
            predicted_place, predicted_surface, predicted_joint = (
                experiment.factorized_predictions(
                    selected_place_logits, selected_surface_logits
                )
            )
        elif objective == "conjunction":
            selected_logits = model(chunk_x)[chunk_valid]
            joint_loss_sum += torch.nn.functional.cross_entropy(
                selected_logits, selected_joint, reduction="sum"
            ).item()
            predicted_place, predicted_surface, predicted_joint = (
                experiment.joint_predictions(selected_logits)
            )
        else:
            raise ValueError(f"Unknown objective: {objective}")

        correct_place += int((predicted_place == selected_place).sum().item())
        correct_surface += int(
            (predicted_surface == selected_surface).sum().item()
        )
        correct_joint += int((predicted_joint == selected_joint).sum().item())
        valid_count += n_valid

    if valid_count == 0:
        raise RuntimeError("Validation set contains no valid timesteps")

    return {
        "validation_loss": (
            place_loss_sum + surface_loss_sum + joint_loss_sum
        ) / valid_count,
        "validation_place_accuracy": correct_place / valid_count,
        "validation_surface_accuracy": correct_surface / valid_count,
        "validation_joint_accuracy": correct_joint / valid_count,
        "validation_valid_timesteps": valid_count,
    }


def make_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
):
    if name == "none":
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            threshold=5e-4,
            min_lr=1e-5,
        )
    raise ValueError(f"Unknown scheduler: {name}")


def save_best_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: SweepConfig,
    seed: int,
    update: int,
    metrics: dict[str, float],
    total_valid_timesteps_seen: int,
    objective: str = "factorized",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "objective": (
                "factorized_place_surface" if objective == "factorized"
                else "joint_place_surface_conjunction"
            ),
            "selection_split": "validation",
            "seed": seed,
            "input_dim": base.INPUT_DIM,
            "hidden_dim": base.HIDDEN_DIM,
            "place_bins": experiment.PLACE_BINS,
            "surface_bins": experiment.SURFACE_BINS,
            "joint_bins": experiment.JOINT_BINS,
            "parameter_count": experiment.count_parameters(model),
            "best_update": update,
            "training_trials_seen": update * config.batch_size,
            "training_valid_timesteps_seen": total_valid_timesteps_seen,
            **asdict(config),
            **metrics,
            "model_state_dict": state,
        },
        path,
    )


def run_one_seed(
    bridge: base.JuliaTrialBridge,
    config: SweepConfig,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    run_name = f"{config.identifier}_seed{seed}"
    result_path = args.output_dir / "runs" / f"{run_name}.json"
    history_path = args.output_dir / "histories" / f"{run_name}.csv"
    checkpoint_path = args.output_dir / "checkpoints" / f"{run_name}_best.pt"

    if result_path.exists() and not args.overwrite:
        print(f"Skipping completed run: {run_name}", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))

    print(f"\n=== {run_name} ===", flush=True)
    train_generator = bridge.make_generator(config.batch_size, rseed=seed)
    validation_generator = bridge.make_generator(
        args.val_trials,
        rseed=args.validation_seed_base + seed,
    )
    validation_x, _, validation_joint, validation_valid = base.unpack_batch(
        validation_generator(), device
    )

    # Reset after creating the Julia generators so every configuration with the
    # same seed begins from identical PyTorch parameter initialization.
    base.set_seed(seed)
    model = make_model(args.objective).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = make_scheduler(config.scheduler, optimizer)

    best_accuracy = -math.inf
    best_metrics: dict[str, float] | None = None
    best_update = 0
    first_target_update: int | None = None
    first_target_trials: int | None = None
    first_target_valid_timesteps: int | None = None
    evaluations_without_improvement = 0
    total_valid_timesteps_seen = 0
    clipped_updates = 0
    gradient_norm_sum = 0.0
    history: list[dict] = []
    start_time = time.time()
    stopping_reason = "maximum_updates"

    for update in range(1, args.max_updates + 1):
        x, _, true_joint, valid = base.unpack_batch(train_generator(), device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if args.objective == "factorized":
            place_logits, surface_logits = model(x)
            loss, place_loss, surface_loss = experiment.factorized_loss(
                place_logits, surface_logits, true_joint, valid,
            )
        else:
            joint_logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                joint_logits[valid], true_joint[valid]
            )
            place_loss = surface_loss = None
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        optimizer.step()

        valid_this_update = int(valid.sum().item())
        total_valid_timesteps_seen += valid_this_update
        gradient_norm_value = float(gradient_norm)
        gradient_norm_sum += gradient_norm_value
        clipped_updates += int(gradient_norm_value > config.gradient_clip)

        should_evaluate = (
            update == 1
            or update == args.max_updates
            or update % args.eval_every == 0
        )
        if not should_evaluate:
            continue

        metrics = validation_metrics(
            model,
            validation_x,
            validation_joint,
            validation_valid,
            args.eval_batch_size,
            args.objective,
        )
        current_accuracy = metrics["validation_joint_accuracy"]
        if scheduler is not None:
            scheduler.step(current_accuracy)

        improved = current_accuracy > best_accuracy + args.min_delta
        if improved:
            best_accuracy = current_accuracy
            best_metrics = metrics
            best_update = update
            evaluations_without_improvement = 0
            save_best_checkpoint(
                checkpoint_path,
                model,
                config,
                seed,
                update,
                metrics,
                total_valid_timesteps_seen,
                args.objective,
            )
        else:
            evaluations_without_improvement += 1

        if first_target_update is None and current_accuracy >= args.target_accuracy:
            first_target_update = update
            first_target_trials = update * config.batch_size
            first_target_valid_timesteps = total_valid_timesteps_seen

        history.append(
            {
                "config_id": config.identifier,
                "seed": seed,
                "update": update,
                "training_trials_seen": update * config.batch_size,
                "training_valid_timesteps_seen": total_valid_timesteps_seen,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "training_loss": float(loss.item()),
                "training_place_loss": (
                    float(place_loss.item()) if place_loss is not None else None
                ),
                "training_surface_loss": (
                    float(surface_loss.item()) if surface_loss is not None else None
                ),
                "gradient_norm_before_clip": gradient_norm_value,
                "fraction_updates_clipped": clipped_updates / update,
                **metrics,
                "best_validation_joint_accuracy_so_far": best_accuracy,
                "elapsed_seconds": time.time() - start_time,
            }
        )
        write_csv(history_path, history)

        print(
            f"{run_name} update={update}/{args.max_updates} "
            f"val_joint={100 * current_accuracy:.2f}% "
            f"val_place={100 * metrics['validation_place_accuracy']:.2f}% "
            f"val_surface={100 * metrics['validation_surface_accuracy']:.2f}% "
            f"best={100 * best_accuracy:.2f}% "
            f"lr={optimizer.param_groups[0]['lr']:.2g}",
            flush=True,
        )

        if args.stop_at_target and current_accuracy >= args.target_accuracy:
            stopping_reason = "target_reached"
            break
        if (
            args.patience_evals > 0
            and evaluations_without_improvement >= args.patience_evals
        ):
            stopping_reason = "validation_patience"
            break

    if best_metrics is None:
        raise RuntimeError(f"No validation result was recorded for {run_name}")

    result = {
        "objective": args.objective,
        "config_id": config.identifier,
        **asdict(config),
        "seed": seed,
        "parameter_count": experiment.count_parameters(model),
        "target_accuracy": args.target_accuracy,
        "target_reached": first_target_update is not None,
        "first_target_update": first_target_update,
        "first_target_training_trials": first_target_trials,
        "first_target_training_valid_timesteps": first_target_valid_timesteps,
        "best_update": best_update,
        "best_training_trials": best_update * config.batch_size,
        "best_validation_joint_accuracy": best_accuracy,
        "best_validation_place_accuracy": best_metrics[
            "validation_place_accuracy"
        ],
        "best_validation_surface_accuracy": best_metrics[
            "validation_surface_accuracy"
        ],
        "best_validation_loss": best_metrics["validation_loss"],
        "mean_gradient_norm_before_clip": gradient_norm_sum / update,
        "fraction_updates_clipped": clipped_updates / update,
        "completed_updates": update,
        "stopping_reason": stopping_reason,
        "elapsed_seconds": time.time() - start_time,
        "checkpoint": str(checkpoint_path.resolve()),
        "history": str(history_path.resolve()),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    del model, validation_x, validation_joint, validation_valid
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def aggregate_results(
    results: list[dict],
    configs: list[SweepConfig],
    requested_seeds: list[int],
) -> list[dict]:
    rows: list[dict] = []
    for config in configs:
        members = [
            row
            for row in results
            if row["config_id"] == config.identifier
            and int(row["seed"]) in requested_seeds
        ]
        if not members:
            continue

        accuracies = np.asarray(
            [row["best_validation_joint_accuracy"] for row in members],
            dtype=float,
        )
        place = np.asarray(
            [row["best_validation_place_accuracy"] for row in members],
            dtype=float,
        )
        surface = np.asarray(
            [row["best_validation_surface_accuracy"] for row in members],
            dtype=float,
        )
        target_trials = [
            row["first_target_training_trials"]
            for row in members
            if row["first_target_training_trials"] is not None
        ]
        target_timesteps = [
            row["first_target_training_valid_timesteps"]
            for row in members
            if row["first_target_training_valid_timesteps"] is not None
        ]
        n = len(members)
        rows.append(
            {
                "config_id": config.identifier,
                **asdict(config),
                "completed_seeds": n,
                "requested_seeds": len(requested_seeds),
                "mean_best_validation_joint_accuracy": float(accuracies.mean()),
                "std_best_validation_joint_accuracy": float(
                    accuracies.std(ddof=1) if n > 1 else 0.0
                ),
                "mean_best_validation_place_accuracy": float(place.mean()),
                "mean_best_validation_surface_accuracy": float(surface.mean()),
                "target_reached_seeds": sum(row["target_reached"] for row in members),
                "mean_trials_to_target": (
                    float(np.mean(target_trials)) if target_trials else ""
                ),
                "mean_valid_timesteps_to_target": (
                    float(np.mean(target_timesteps)) if target_timesteps else ""
                ),
                "mean_best_update": float(
                    np.mean([row["best_update"] for row in members])
                ),
                "mean_fraction_updates_clipped": float(
                    np.mean([row["fraction_updates_clipped"] for row in members])
                ),
                "mean_elapsed_seconds": float(
                    np.mean([row["elapsed_seconds"] for row in members])
                ),
            }
        )

    rows.sort(
        key=lambda row: row["mean_best_validation_joint_accuracy"],
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def plot_ranking(
    rows: list[dict], path: Path, objective: str = "factorized",
    target_accuracy: float = 0.83,
) -> None:
    if not rows:
        return
    shown = rows[: min(20, len(rows))][::-1]
    labels = [row["config_id"] for row in shown]
    means = np.asarray(
        [row["mean_best_validation_joint_accuracy"] for row in shown]
    )
    errors = np.asarray(
        [row["std_best_validation_joint_accuracy"] for row in shown]
    )
    fig_height = max(5.5, 0.42 * len(shown) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y = np.arange(len(shown))
    ax.barh(y, means, xerr=errors, color="#3976AF", alpha=0.9, capsize=3)
    ax.axvline(
        target_accuracy, color="#D64B40", linestyle="--",
        label=f"{100 * target_accuracy:g}% reference",
    )
    ax.set_yticks(y, labels)
    ax.set_xlim(max(0.0, float(means.min()) - 0.08), 1.0)
    ax.set_xlabel("Best validation exact-conjunction accuracy")
    label = "Factorized-discrete" if objective == "factorized" else "Conjunction"
    ax.set_title(f"{label} hyperparameter sweep")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args(default_objective: str = "factorized") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Two-update smoke test")
    parser.add_argument(
        "--objective", choices=["factorized", "conjunction"],
        default=default_objective,
    )
    parser.add_argument(
        "--config-index", type=int, default=None,
        help="Run one zero-based grid index (for serialized Slurm arrays)",
    )
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-3, 5e-4])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--gradient-clips", type=float, nargs="+", default=[1.0, 3.0])
    parser.add_argument(
        "--schedulers",
        choices=["none", "plateau"],
        nargs="+",
        default=["none"],
    )
    parser.add_argument("--max-updates", type=int, default=8000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--val-trials", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--target-accuracy", type=float, default=0.83)
    parser.add_argument("--validation-seed-base", type=int, default=900000)
    parser.add_argument("--patience-evals", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=5e-4)
    parser.add_argument("--stop-at-target", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main(default_objective: str = "factorized") -> None:
    args = parse_args(default_objective)
    if args.output_dir is None:
        args.output_dir = Path(f"{args.objective}_sweep")
    if args.quick:
        args.learning_rates = [1e-3]
        args.batch_sizes = [8]
        args.gradient_clips = [1.0]
        args.schedulers = ["none"]
        args.max_updates = 2
        args.eval_every = 1
        args.val_trials = 32
        args.eval_batch_size = 32
        args.seeds = [1]
        args.patience_evals = 0
        args.output_dir = args.output_dir.parent / f"{args.output_dir.name}_quick"

    if args.max_updates < 1 or args.eval_every < 1 or args.val_trials < 1:
        raise ValueError("Updates, evaluation interval and validation trials must be positive")
    if not 0.0 < args.target_accuracy <= 1.0:
        raise ValueError("--target-accuracy must lie in (0, 1]")
    if any(value <= 0 for value in args.learning_rates):
        raise ValueError("Learning rates must be positive")
    if any(value <= 0 for value in args.batch_sizes):
        raise ValueError("Batch sizes must be positive")
    if any(value <= 0 for value in args.gradient_clips):
        raise ValueError("Gradient clipping thresholds must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = base.choose_device(args.device)
    print(f"Device: {device}", flush=True)
    print("Initializing Julia RNNTrialStructures bridge...", flush=True)
    bridge = base.JuliaTrialBridge()
    print("Julia bridge ready.", flush=True)

    configs = [
        SweepConfig(lr, batch_size, clip, scheduler)
        for lr, batch_size, clip, scheduler in itertools.product(
            args.learning_rates,
            args.batch_sizes,
            args.gradient_clips,
            args.schedulers,
        )
    ]
    if args.config_index is not None and not 0 <= args.config_index < len(configs):
        raise ValueError(f"--config-index must be in 0..{len(configs) - 1}")
    manifest = {
        "purpose": f"{args.objective} validation hyperparameter sweep",
        "objective": args.objective,
        "selection_metric": "mean best validation exact-conjunction accuracy",
        "final_test_used": False,
        "target_accuracy": args.target_accuracy,
        "max_updates": args.max_updates,
        "eval_every": args.eval_every,
        "validation_trials": args.val_trials,
        "validation_seed_base": args.validation_seed_base,
        "seeds": args.seeds,
        "patience_evals": args.patience_evals,
        "min_delta": args.min_delta,
        "stop_at_target": args.stop_at_target,
        "input_dim": base.INPUT_DIM,
        "hidden_dim": base.HIDDEN_DIM,
        "parameter_count": experiment.count_parameters(
            make_model(args.objective)
        ),
        "configs": [
            {"config_id": config.identifier, **asdict(config)}
            for config in configs
        ],
    }
    manifest_path = args.output_dir / "sweep_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Old factorized manifests lack the objective field. Their purpose still
        # identifies them; never reuse their records for conjunction training.
        previous_objective = previous_manifest.get("objective", "factorized")
        if previous_objective != args.objective:
            raise ValueError("Output directory belongs to a different objective")
        for key in (
            "max_updates", "eval_every", "validation_trials",
            "validation_seed_base", "seeds", "configs",
        ):
            if previous_manifest.get(key) != manifest[key]:
                raise ValueError(
                    f"Existing sweep differs in {key}; use a new output directory"
                )
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Reload finished records so sequential array elements produce one combined
    # ranking. Do not run elements concurrently in this shared output directory.
    results_by_name: dict[str, dict] = {}
    for config in configs:
        for seed in args.seeds:
            name = f"{config.identifier}_seed{seed}"
            path = args.output_dir / "runs" / f"{name}.json"
            if path.exists() and not args.overwrite:
                results_by_name[name] = json.loads(path.read_text(encoding="utf-8"))
    selected_configs = (
        configs if args.config_index is None else [configs[args.config_index]]
    )
    for config in selected_configs:
        for seed in args.seeds:
            name = f"{config.identifier}_seed{seed}"
            results_by_name[name] = run_one_seed(bridge, config, seed, args, device)
            ranking = aggregate_results(list(results_by_name.values()), configs, args.seeds)
            write_csv(args.output_dir / "sweep_results.csv", ranking)
            plot_ranking(
                ranking, args.output_dir / "sweep_ranking.png",
                args.objective, args.target_accuracy,
            )

    ranking = aggregate_results(list(results_by_name.values()), configs, args.seeds)
    complete = [
        row for row in ranking if row["completed_seeds"] == len(args.seeds)
    ]
    if not complete:
        raise RuntimeError("No configuration completed all requested seeds")
    best = complete[0]
    best_config = {
        "objective": args.objective,
        "completed_configs": len(complete),
        "grid_configs": len(configs),
        "selection_metric": "mean best validation exact-conjunction accuracy",
        "selected_config_id": best["config_id"],
        "learning_rate": best["learning_rate"],
        "batch_size": best["batch_size"],
        "gradient_clip": best["gradient_clip"],
        "scheduler": best["scheduler"],
        "mean_best_validation_joint_accuracy": best[
            "mean_best_validation_joint_accuracy"
        ],
        "std_best_validation_joint_accuracy": best[
            "std_best_validation_joint_accuracy"
        ],
        "mean_best_update": best["mean_best_update"],
        "target_accuracy": args.target_accuracy,
        "target_reached_seeds": best["target_reached_seeds"],
        "seeds": args.seeds,
        "note": (
            "Use these settings for the five-seed confirmation run. "
            "Do not select settings using the final 4,096-trial test set."
        ),
    }
    (args.output_dir / "best_config.json").write_text(
        json.dumps(best_config, indent=2), encoding="utf-8"
    )

    print("\nBest completed configuration:", flush=True)
    print(json.dumps(best_config, indent=2), flush=True)
    print(f"\nSweep results written to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

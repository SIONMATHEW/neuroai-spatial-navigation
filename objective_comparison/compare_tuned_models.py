"""Five-seed, clean, validation-selected comparison at the frozen tuning budget.

Both objectives train on each identical batch and begin with identical recurrent
weights. Each gets a separate validation-best checkpoint; test trials are only
generated after selection. This is best-within-budget, not proven optimality.

Run --quick on a GPU first. Cluster arrays write separate per-seed directories;
--aggregate-only collects all requested seeds and makes the figures on CPU.
"""

from __future__ import annotations

# Imports juliacall before torch, consistent with the existing experiments.
import sweep_factorized as sweep

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

experiment = sweep.experiment
base = sweep.base


def write_json(path: Path, value: dict) -> None:
    def finite_json(item):
        if isinstance(item, dict):
            return {key: finite_json(member) for key, member in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite_json(member) for member in item]
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def protocol(args: argparse.Namespace) -> dict:
    return {
        "comparison": "frozen_tuned_settings_paired_clean_confirmation",
        "selection_split": "validation",
        "selection_metric": "exact_joint_top1_accuracy",
        "checkpoint_rule": "strict highest validation accuracy; earliest tie",
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gradient_clip": args.gradient_clip,
        "scheduler": "none",
        "val_trials": args.val_trials,
        "validate_every": args.validate_every,
        "eval_trials": args.eval_trials,
        "eval_batch_size": args.eval_batch_size,
        "validation_seed_base": args.validation_seed_base,
        "test_seed_base": args.test_seed_base,
        "target_accuracy": args.target_accuracy,
        "also_accuracy_matched": args.also_accuracy_matched,
        "match_validate_every": args.match_validate_every,
        "target_tolerance": args.target_tolerance,
        "pair_tolerance": args.pair_tolerance,
        "matching_rule": "validation-only: within target band and pair gap; minimize gap then target distance then updates",
        "input_dim": base.INPUT_DIM,
        "input_channels": ["distance", "movement", "texture"],
        "hidden_dim": base.HIDDEN_DIM,
        "place_bins": experiment.PLACE_BINS,
        "surface_bins": experiment.SURFACE_BINS,
        "min_steps": base.MIN_STEPS,
        "max_steps": base.MAX_STEPS,
        "bin_size": base.BIN_SIZE,
        "wall_bin_size": base.WALL_BIN_SIZE,
        "dt": base.DT,
        "fov": base.FOV,
        "head_direction_step": base.HEAD_DIRECTION_STEP,
        "p_stay": base.P_STAY,
        "p_hd": base.P_HD,
        "evaluation_noise_std": 0.0,
        "same_training_batches": True,
        "same_recurrent_initialization": True,
        "fresh_training_all_seeds": True,
        "test_support_note": (
            "Legacy unsupported_joint_rate means a predicted combination was "
            "not observed in this finite test sample, not proven impossible."
        ),
    }


def build_pair(seed: int, device: torch.device):
    base.set_seed(seed)
    factorized = experiment.FactorizedDiscreteRNN().to(device)
    conjunction = base.ConjunctionRNN().to(device)
    conjunction.rnn.load_state_dict(factorized.rnn.state_dict())
    return factorized, conjunction


def recurrent_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in model.rnn.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def select_accuracy_match(selections, args):
    """Select a pair before seeing test scores; never force a match."""
    eligible = []
    for f in selections["factorized"].get("match_candidates", []):
        for c in selections["conjunction"].get("match_candidates", []):
            if any(abs(member["accuracy"] - args.target_accuracy) > args.target_tolerance + 1e-12
                   for member in (f, c)):
                continue
            gap = abs(f["accuracy"] - c["accuracy"])
            if gap <= args.pair_tolerance + 1e-12:
                rank = (gap, abs(f["accuracy"] - args.target_accuracy) +
                        abs(c["accuracy"] - args.target_accuracy),
                        f["update"] + c["update"], f["update"], c["update"])
                eligible.append((rank, f, c))
    if not eligible:
        return {"status": "unmatched", "reason": "No checkpoint pair met both validation tolerances",
                "candidate_counts": {name: len(s.get("match_candidates", []))
                                     for name, s in selections.items()}}
    _, f, c = min(eligible, key=lambda item: item[0])
    return {"status": "matched", "factorized": f, "conjunction": c,
            "validation_gap": abs(f["accuracy"] - c["accuracy"])}


def train_and_select(bridge, seed: int, args, device, seed_dir: Path):
    # Materialize validation once. Both models use the very same tensors, and
    # no validation batch is used for gradient updates.
    validation_generator = bridge.make_generator(
        args.val_trials, rseed=args.validation_seed_base + seed
    )
    val_x, _, val_joint, val_valid = base.unpack_batch(validation_generator(), device)
    generator = bridge.make_generator(args.batch_size, rseed=seed)
    factorized, conjunction = build_pair(seed, device)
    initialization_digest = recurrent_digest(factorized)
    assert initialization_digest == recurrent_digest(conjunction)
    models = {"factorized": factorized, "conjunction": conjunction}
    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        for name, model in models.items()
    }
    config = sweep.SweepConfig(
        args.learning_rate, args.batch_size, args.gradient_clip, "none"
    )
    selections = {
        name: {"best_accuracy": -1.0, "first_target_update": None,
               "first_target_trials": None, "first_target_valid_timesteps": None,
               "first_target_validation_accuracy": None, "match_candidates": []}
        for name in models
    }
    history = []
    timesteps_seen = 0
    start = time.perf_counter()

    for update in range(1, args.updates + 1):
        x, _, joint, valid = base.unpack_batch(generator(), device)
        training = {}
        best_check = (
            update == 1 or update == args.updates or update % args.validate_every == 0
        )
        should_validate = best_check or (args.also_accuracy_matched and
                                        update % args.match_validate_every == 0)
        for name, model in models.items():
            model.train()
            optimizers[name].zero_grad(set_to_none=True)
            if name == "factorized":
                output = model(x)
                loss, _, _ = experiment.factorized_loss(*output, joint, valid)
            else:
                output = model(x)
                loss = F.cross_entropy(output[valid], joint[valid])
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizers[name].step()
            if should_validate:
                with torch.no_grad():
                    if name == "factorized":
                        p, s, j = experiment.factorized_predictions(*output)
                    else:
                        p, s, j = experiment.joint_predictions(output)
                    training.update({
                        f"{name}_loss": float(loss.item()),
                        f"{name}_joint_accuracy": float((j[valid] == joint[valid]).float().mean()),
                        f"{name}_place_accuracy": float((p[valid] == joint[valid] // experiment.SURFACE_BINS).float().mean()),
                        f"{name}_surface_accuracy": float((s[valid] == joint[valid] % experiment.SURFACE_BINS).float().mean()),
                        f"{name}_gradient_norm_before_clip": float(norm),
                    })
        timesteps_seen += int(valid.sum())
        if not should_validate:
            continue
        row = {
            "seed": seed, "update": update,
            "training_trials_seen": update * args.batch_size,
            "training_valid_timesteps_seen": timesteps_seen,
            **training,
        }
        for name, model in models.items():
            metrics = sweep.validation_metrics(
                model, val_x, val_joint, val_valid, args.eval_batch_size, name
            )
            row.update({f"{name}_{key}": value for key, value in metrics.items()})
            selected = selections[name]
            accuracy = metrics["validation_joint_accuracy"]
            if best_check and accuracy > selected["best_accuracy"]:
                selected.update({
                    "best_accuracy": accuracy, "best_update": update,
                    "best_training_trials": update * args.batch_size,
                    "best_training_valid_timesteps": timesteps_seen,
                    "best_metrics": metrics,
                })
                sweep.save_best_checkpoint(
                    seed_dir / "checkpoints" / f"{name}_best.pt",
                    model, config, seed, update, metrics, timesteps_seen, name,
                )
            if args.also_accuracy_matched and abs(accuracy - args.target_accuracy) <= args.target_tolerance + 1e-12:
                relative_path = f"checkpoints/accuracy_candidates/{name}_update_{update}.pt"
                sweep.save_best_checkpoint(
                    seed_dir / relative_path, model, config, seed, update,
                    metrics, timesteps_seen, name,
                )
                selected["match_candidates"].append({
                    "update": update, "accuracy": accuracy, "checkpoint": relative_path,
                    "training_trials": update * args.batch_size,
                    "training_valid_timesteps": timesteps_seen,
                })
            if selected["first_target_update"] is None and accuracy >= args.target_accuracy:
                selected.update({
                    "first_target_update": update,
                    "first_target_trials": update * args.batch_size,
                    "first_target_valid_timesteps": timesteps_seen,
                    "first_target_validation_accuracy": accuracy,
                })
                # First crossing is a sample-efficiency diagnostic, not
                # automatically the accuracy-matched checkpoint.
                sweep.save_best_checkpoint(
                    seed_dir / "checkpoints" / f"{name}_first_target.pt",
                    model, config, seed, update, metrics, timesteps_seen, name,
                )
        row["elapsed_seconds"] = time.perf_counter() - start
        history.append(row)
        experiment.write_csv(seed_dir / "training_history.csv", history)
        print(
            f"seed={seed} update={update}/{args.updates} "
            f"val factorized={100 * row['factorized_validation_joint_accuracy']:.2f}% "
            f"conjunction={100 * row['conjunction_validation_joint_accuracy']:.2f}%",
            flush=True,
        )

    # Complete the full shared budget before restoring independently selected
    # checkpoints. The final update's weights are not necessarily the best.
    for name, model in models.items():
        checkpoint = torch.load(
            seed_dir / "checkpoints" / f"{name}_best.pt",
            map_location="cpu", weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    return factorized, conjunction, selections, initialization_digest, timesteps_seen


def run_seed(bridge, seed: int, args, device):
    seed_dir = args.output_dir / "seed_runs" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    completed = seed_dir / "completed.json"
    if completed.exists():
        record = json.loads(completed.read_text(encoding="utf-8"))
        if record["protocol"] != protocol(args):
            raise ValueError("Completed seed has different settings; use a new output directory")
        if not (seed_dir / "raw_evaluation.npz").exists():
            raise RuntimeError("Completed record has no raw evaluation file")
        if args.also_accuracy_matched and record["accuracy_matching"]["status"] == "matched" and not (seed_dir / "accuracy_matched_raw_evaluation.npz").exists():
            raise RuntimeError("Completed record has no accuracy-matched evaluation file")
        print(f"Skipping completed seed {seed}", flush=True)
        return record
    print(f"\n=== Fresh paired training: seed {seed} ===", flush=True)
    factorized, conjunction, selections, digest, timesteps = train_and_select(
        bridge, seed, args, device, seed_dir
    )
    matching = select_accuracy_match(selections, args) if args.also_accuracy_matched else None
    # First use of test data happens here, only after all checkpoint selection.
    metrics, arrays = experiment.evaluate_pair(
        bridge, factorized, conjunction, seed, args.eval_trials,
        args.eval_batch_size, device, test_seed_base=args.test_seed_base,
    )
    np.savez_compressed(seed_dir / "raw_evaluation.npz", **arrays)
    matched_metrics = None
    if matching and matching["status"] == "matched":
        for name, model in (("factorized", factorized), ("conjunction", conjunction)):
            checkpoint = torch.load(seed_dir / matching[name]["checkpoint"],
                                    map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
        matched_metrics, matched_arrays = experiment.evaluate_pair(
            bridge, factorized, conjunction, seed, args.eval_trials,
            args.eval_batch_size, device, test_seed_base=args.test_seed_base,
        )
        for key in ("true_joint", "true_xy"):
            if not np.array_equal(arrays[key], matched_arrays[key]):
                raise RuntimeError("Best and matched comparisons must use identical test trials")
        np.savez_compressed(seed_dir / "accuracy_matched_raw_evaluation.npz", **matched_arrays)
        for name in ("factorized", "conjunction"):
            for key in ("update", "accuracy", "training_trials", "training_valid_timesteps"):
                matched_metrics[f"{name}_selected_{key}"] = matching[name][key]
        print(f"seed={seed} validation match: "
              f"factorized={100 * matching['factorized']['accuracy']:.2f}% "
              f"conjunction={100 * matching['conjunction']['accuracy']:.2f}%", flush=True)
    elif matching:
        print(f"seed={seed} accuracy matching UNMATCHED: {matching['reason']}", flush=True)
    metrics.update({"completed_training_updates": args.updates,
                    "training_trials_seen": args.updates * args.batch_size,
                    "training_valid_timesteps_seen": timesteps})
    for name, selection in selections.items():
        for key in ("best_update", "best_accuracy", "best_training_trials",
                    "first_target_update", "first_target_trials", "first_target_valid_timesteps",
                    "first_target_validation_accuracy"):
            metrics[f"{name}_{key}"] = selection[key]
    record = {
        "seed": seed, "protocol": protocol(args), "metrics": metrics,
        "selections": selections, "initial_recurrent_sha256": digest,
        "accuracy_matching": matching, "accuracy_matched_metrics": matched_metrics,
        "training_seed": seed, "validation_seed": args.validation_seed_base + seed,
        "test_seed": args.test_seed_base + seed,
        "torch_version": str(torch.__version__), "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    # Written last: an interrupted seed has no completed marker and restarts.
    write_json(completed, record)
    print(
        f"seed={seed} TEST factorized={100 * metrics['factorized_joint_accuracy']:.2f}% "
        f"conjunction={100 * metrics['conjunction_joint_accuracy']:.2f}%",
        flush=True,
    )
    return record


def validation_plot(history: list[dict], path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    updates = sorted({int(row["update"]) for row in history})
    for name, color in (("factorized", "#3976AF"), ("conjunction", "#D64B40")):
        key = f"{name}_validation_joint_accuracy"
        for seed in sorted({int(row["seed"]) for row in history}):
            rows = [row for row in history if int(row["seed"]) == seed]
            ax.plot([int(row["update"]) for row in rows], [float(row[key]) for row in rows], color=color, alpha=0.2)
        means = [np.mean([float(row[key]) for row in history if int(row["update"]) == u]) for u in updates]
        ax.plot(updates, means, color=color, linewidth=2.5, label=name.capitalize())
    ax.set(xlabel="Training update", ylabel="Validation exact-joint top-1 accuracy",
           ylim=(0, 1), title="Frozen-settings confirmation: validation learning curves")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_test_comparison(rows, arrays, directory, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    experiment.write_csv(directory / "metrics_by_seed.csv", rows)
    paired = []
    for metric in ("joint_accuracy", "place_accuracy", "surface_accuracy"):
        f = np.asarray([row[f"factorized_{metric}"] for row in rows])
        c = np.asarray([row[f"conjunction_{metric}"] for row in rows])
        mean, half_width = experiment.mean_ci(c - f)
        paired.append({
            "metric": metric, "n_seeds": len(rows),
            "factorized_mean": float(f.mean()), "conjunction_mean": float(c.mean()),
            "paired_difference_percentage_points": 100 * mean,
            "paired_95ci_low_percentage_points": 100 * (mean - half_width) if len(rows) > 1 else None,
            "paired_95ci_high_percentage_points": 100 * (mean + half_width) if len(rows) > 1 else None,
        })
    experiment.write_csv(directory / "paired_test_differences.csv", paired)
    numeric_keys = [key for key in rows[0] if key != "seed" and all(
        isinstance(row[key], (float, int)) and np.isfinite(row[key]) for row in rows
    )]
    write_json(directory / "summary.json", {
        **metadata, "n_seeds": len(rows),
        "metrics_mean": {key: float(np.mean([row[key] for row in rows])) for key in numeric_keys},
        "metrics_seed_sd": {key: float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else None for key in numeric_keys},
        "paired_test_differences": paired,
        "uncertainty_note": "95% Student-t intervals across paired seeds; timesteps are not independent replicates.",
        "factorized_parameter_count": experiment.count_parameters(experiment.FactorizedDiscreteRNN()),
        "conjunction_parameter_count": experiment.count_parameters(base.ConjunctionRNN()),
    })
    experiment.grouped_accuracy_plot(rows, directory / "overall_accuracy.png")
    experiment.dumbbell_plot(rows, directory / "paired_accuracy_dumbbell.png")
    experiment.place_error_map(arrays, directory / "place_bin_error_map.png")
    experiment.surface_error_map(arrays, directory / "surface_bin_error_map.png")
    experiment.position_coordinate_error_map(arrays, directory / "position_coordinate_error_map.png")
    experiment.gaze_coordinate_error_spatial_map(arrays, directory / "gaze_coordinate_error_spatial_map.png")
    experiment.diagnostic_bars(rows, directory / "joint_error_diagnostics.png",
                               unobserved_label="Joint not seen\nin test sample")


def checkpoint_summary(records, directory, args, matched=False):
    """Presentation figure and per-seed CSV; all accuracy columns name their split."""
    rows = []
    for record in records:
        values = {}
        metrics = record["accuracy_matched_metrics"] if matched else record["metrics"]
        for name in ("factorized", "conjunction"):
            selection = record["accuracy_matching"][name] if matched else record["selections"][name]
            values[name] = {
                "selected_update": selection["update" if matched else "best_update"],
                "training_trials": selection["training_trials" if matched else "best_training_trials"],
                "validation_joint_accuracy": selection["accuracy" if matched else "best_accuracy"],
                "test_joint_accuracy": metrics[f"{name}_joint_accuracy"],
                "test_place_accuracy": metrics[f"{name}_place_accuracy"],
                "test_surface_accuracy": metrics[f"{name}_surface_accuracy"],
            }
        row = {"seed": record["seed"]}
        for name, members in values.items():
            row.update({f"{name}_{key}": value for key, value in members.items()})
        for key in values["factorized"]:
            difference = values["conjunction"][key] - values["factorized"][key]
            row[f"conjunction_minus_factorized_{key}" + ("_percentage_points" if "accuracy" in key else "")] = difference * (100 if "accuracy" in key else 1)
        rows.append(row)
    experiment.write_csv(directory / "checkpoint_comparison_by_seed.csv", rows)
    keys = list(values["factorized"])
    means = {name: [np.mean([row[f"{name}_{key}"] for row in rows]) for key in keys]
             for name in ("factorized", "conjunction")}
    cells = []
    for name, label in (("factorized", "Factorized"), ("conjunction", "Conjunction")):
        cells.append([label] + [f"{value:,.0f}" if i < 2 else f"{100 * value:.2f}%"
                                for i, value in enumerate(means[name])])
    differences = np.asarray(means["conjunction"]) - np.asarray(means["factorized"])
    cells.append(["Conj. minus fact."] + [f"{value:+,.0f}" if i < 2 else f"{100 * value:+.2f} pp"
                                          for i, value in enumerate(differences)])
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=["Model", "Selected\nupdate", "Trials through\ncheckpoint",
                     "Validation\njoint top-1", "Test\njoint top-1", "Test\nplace", "Test\nsurface"],
                     cellLoc="center", bbox=[0, 0.28, 1, 0.48])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D5DCE3")
        if row == 0:
            cell.set_facecolor("#E9EEF3")
        elif column == 0:
            cell.set_facecolor("#EDF4FB" if row == 1 else "#FBEFEE" if row == 2 else "#F3F3F3")
    title = "Validation-accuracy-matched checkpoints" if matched else "Best validation-selected checkpoints within budget"
    ax.set_title(f"{title} — means across {len(records)} paired seeds", fontsize=16, pad=16)
    note = (f"Both models completed {args.updates:,} updates ({args.updates * args.batch_size:,} trials) per seed. "
            "Selected update is not an early-stop instruction.\n"
            f"LR={args.learning_rate:g}; batch={args.batch_size}; gradient clip={args.gradient_clip:g}; scheduler=none. "
            "Accuracy differences are percentage points.")
    if matched:
        note += (f"\nValidation target: {100 * args.target_accuracy:.1f}% ± {100 * args.target_tolerance:.1f} pp; "
                 f"pair gap ≤ {100 * args.pair_tolerance:.1f} pp. Test accuracy is not forced to match.")
    fig.text(0.05, 0.06, note, fontsize=10, va="bottom")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.83, bottom=0.12)
    fig.savefig(directory / "checkpoint_comparison.png", dpi=180)
    plt.close(fig)


def aggregate(args):
    missing = [seed for seed in args.seeds if not (
        args.output_dir / "seed_runs" / f"seed_{seed}" / "completed.json"
    ).exists()]
    if missing:
        raise RuntimeError(f"Cannot aggregate: seeds {missing} are not complete")
    records, arrays, history = [], [], []
    for seed in args.seeds:
        directory = args.output_dir / "seed_runs" / f"seed_{seed}"
        record = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
        if record["protocol"] != protocol(args):
            raise ValueError(f"Protocol mismatch for seed {seed}")
        records.append(record)
        with np.load(directory / "raw_evaluation.npz") as archive:
            arrays.append({key: archive[key] for key in archive.files})
        with (directory / "training_history.csv").open(newline="", encoding="utf-8") as handle:
            history.extend(list(csv.DictReader(handle)))
    rows = [record["metrics"] for record in records]
    experiment.write_csv(args.output_dir / "training_history.csv", history)
    render_test_comparison(rows, arrays, args.output_dir, {
        "protocol": protocol(args), "seeds": args.seeds,
        "comparison_kind": "best_validation_checkpoint_within_budget",
    })
    checkpoint_summary(records, args.output_dir, args)
    if args.also_accuracy_matched:
        matched_records = [r for r in records if r["accuracy_matching"]["status"] == "matched"]
        matched_dir = args.output_dir / "accuracy_matched"
        matching_summary = {
            "protocol": protocol(args), "requested_seeds": args.seeds,
            "seeds": [r["seed"] for r in matched_records],
            "unmatched_seeds": [r["seed"] for r in records if r not in matched_records],
            "comparison_kind": "validation_accuracy_matched_not_test_accuracy_matched",
            "matching_by_seed": [{"seed": r["seed"], **r["accuracy_matching"]} for r in records],
        }
        write_json(matched_dir / "matching_status.json", matching_summary)
        if matched_records:
            matched_arrays = []
            for record in matched_records:
                path = args.output_dir / "seed_runs" / f"seed_{record['seed']}" / "accuracy_matched_raw_evaluation.npz"
                with np.load(path) as archive:
                    matched_arrays.append({key: archive[key] for key in archive.files})
            render_test_comparison([r["accuracy_matched_metrics"] for r in matched_records],
                                   matched_arrays, matched_dir, matching_summary)
            checkpoint_summary(matched_records, matched_dir, args, matched=True)
        else:
            write_json(matched_dir / "summary.json", {**matching_summary, "n_seeds": 0})
        print(f"Accuracy matching: {len(matched_records)}/{len(records)} seeds met tolerances", flush=True)
    validation_plot(history, args.output_dir / "validation_comparison.png")
    # Training metrics were measured on each current batch before its update;
    # validation curves above were measured after the update.
    numeric_history = [{key: float(value) for key, value in row.items()} for row in history]
    experiment.training_plot(numeric_history, args.output_dir / "training_comparison.png")
    print(f"Comparison figures and summary written to {args.output_dir.resolve()}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--train-only", action="store_true", help="Train/select/test seeds; defer combined plots")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--updates", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gradient-clip", type=float, default=3.0)
    parser.add_argument("--val-trials", type=int, default=1024)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--eval-trials", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--validation-seed-base", type=int, default=900000)
    parser.add_argument("--test-seed-base", type=int, default=2000000)
    parser.add_argument("--target-accuracy", type=float, default=0.83)
    parser.add_argument("--also-accuracy-matched", action="store_true",
                        help="Also test validation-matched checkpoints from the SAME full training runs")
    parser.add_argument("--match-validate-every", type=int, default=50)
    parser.add_argument("--target-tolerance", type=float, default=0.005,
                        help="Absolute accuracy tolerance: 0.005 = 0.5 percentage points")
    parser.add_argument("--pair-tolerance", type=float, default=0.005)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path("objective_results" if args.also_accuracy_matched else "objective_results_best_only")
    if args.quick:
        args.updates = 2
        args.batch_size = 8
        args.val_trials = 32
        args.eval_trials = 32
        args.eval_batch_size = 8
        args.validate_every = 1
        args.match_validate_every = 1
        args.seeds = [1]
        args.output_dir = args.output_dir.parent / f"{args.output_dir.name}_quick"
    for field in ("updates", "batch_size", "val_trials", "validate_every", "match_validate_every", "eval_trials", "eval_batch_size", "learning_rate", "gradient_clip"):
        if getattr(args, field) <= 0:
            parser.error(f"{field} must be positive")
    if args.eval_trials % args.eval_batch_size:
        parser.error("eval-trials must be divisible by eval-batch-size")
    if not 0 < args.target_accuracy <= 1:
        parser.error("target-accuracy must be in (0,1]")
    if not 0 <= args.target_tolerance <= 1 or not 0 <= args.pair_tolerance <= 1:
        parser.error("Accuracy tolerances must be in [0,1]")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 1:
        parser.error("seeds must be distinct positive integers")
    splits = [set(args.seeds), {args.validation_seed_base + s for s in args.seeds}, {args.test_seed_base + s for s in args.seeds}]
    if any(splits[i] & splits[j] for i in range(3) for j in range(i + 1, 3)):
        parser.error("Training, validation and test RNG seeds must be disjoint")
    if args.aggregate_only and args.train_only:
        parser.error("aggregate-only and train-only are mutually exclusive")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != protocol(args):
            raise ValueError("Output directory has different settings; use a fresh directory")
    else:
        write_json(config_path, protocol(args))
    if args.aggregate_only:
        aggregate(args)
        return
    device = base.choose_device(args.device)
    print(f"Device: {device}", flush=True)
    print("Initializing Julia RNNTrialStructures bridge...", flush=True)
    bridge = base.JuliaTrialBridge()
    print("Julia bridge ready.", flush=True)
    for seed in args.seeds:
        run_seed(bridge, seed, args, device)
    if not args.train_only:
        aggregate(args)


if __name__ == "__main__":
    main()

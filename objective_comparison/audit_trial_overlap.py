"""Replay trial streams without model training; audit complete-trial duplicates.

Run in the ORIGINAL cluster environment beside controlled_comparison.py.
Results are conditional on faithful replay: saved test truths and cumulative
training timestep counts must match. Historical input tensors were not saved,
so their exact identity cannot be independently certified retrospectively.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def digest_arrays(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        # Canonical byte order, shape and dtype; padding is excluded upstream.
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        h.update(str((a.shape, a.dtype.str)).encode("ascii"))
        h.update(a.tobytes())
    return h.digest()


def fingerprint_batch(raw):
    x, xy, weights, joint = (np.asarray(a) for a in raw)
    if x.ndim != 3 or xy.shape[:2] != (4, x.shape[1]):
        raise ValueError("Expected original (channels,time,batch) bridge arrays")
    x = x.astype(np.float32, copy=False)
    xy = xy.astype(np.float32, copy=False)
    joint = joint.astype(np.int64, copy=False)
    valid = weights[0] > 0.5
    full, coordinates = [], []
    for b in range(x.shape[2]):
        keep = valid[:, b]
        if not keep.any() or np.any(np.diff(keep.astype(int)) > 0):
            raise ValueError("Expected a nonempty contiguous valid prefix")
        xi, yi, ji = x[:, keep, b].T, xy[:, keep, b].T, joint[keep, b]
        full.append(digest_arrays(xi, yi, ji))
        coordinates.append(digest_arrays(yi, ji))
    # Match the evaluator's batch-major boolean flattening order exactly.
    truths = {
        "true_xy": xy.transpose(2, 1, 0)[valid.T],
        "true_joint": joint.T[valid.T],
    }
    return {"input_and_target": full, "position_gaze_sequence": coordinates}, truths, int(valid.sum())


def intersections(first_indices, query, examples_limit=5):
    examples = []
    count = 0
    for index, fingerprint in enumerate(query):
        if fingerprint in first_indices:
            count += 1
            if len(examples) < examples_limit:
                examples.append({"reference_trial_index_0based": first_indices[fingerprint],
                                 "query_trial_index_0based": index})
    return {"query_trials_with_exact_duplicate": count, "query_trials": len(query),
            "fraction": count / len(query) if query else None, "examples": examples}


def first_indices(hashes):
    result = {}
    for i, fingerprint in enumerate(hashes):
        result.setdefault(fingerprint, i)
    return result


def collect(generator, batches):
    hashes = {"input_and_target": [], "position_gaze_sequence": []}
    truths = {"true_xy": [], "true_joint": []}
    steps = 0
    for _ in range(batches):
        batch_hashes, batch_truths, count = fingerprint_batch(generator())
        steps += count
        for key in hashes:
            hashes[key].extend(batch_hashes[key])
        for key in truths:
            truths[key].append(batch_truths[key])
    return hashes, {k: np.concatenate(v) for k, v in truths.items()}, steps


def require_equal_truths(replayed, saved):
    for key in ("true_xy", "true_joint"):
        if not np.array_equal(replayed[key], saved[key]):
            raise RuntimeError(f"Historical test replay mismatch: {key}. No overlap conclusion is valid.")


def check_protocol(base, protocol):
    fields = {"input_dim": "INPUT_DIM", "min_steps": "MIN_STEPS", "max_steps": "MAX_STEPS",
              "bin_size": "BIN_SIZE", "wall_bin_size": "WALL_BIN_SIZE", "dt": "DT",
              "fov": "FOV", "head_direction_step": "HEAD_DIRECTION_STEP",
              "p_stay": "P_STAY", "p_hd": "P_HD"}
    for field, constant in fields.items():
        if protocol[field] != getattr(base, constant):
            raise ValueError(f"Current bridge constant differs from original protocol: {field}")
    if protocol["input_channels"] != ["distance", "movement", "texture"]:
        raise ValueError("Unsupported input-channel configuration")


def provenance(bridge, base):
    source = Path(str(bridge.jl.seval("pathof(RNNTrialStructures)")))
    files = {str(p.relative_to(source.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(source.parent.rglob("*.jl"))}
    return {"julia_version": str(bridge.jl.seval("string(VERSION)")),
            "installed_julia_source": str(source), "current_julia_source_sha256": files,
            "current_python_bridge_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
            "historical_source_hashes_available": False}


def audit_seed(bridge, base, directory, seed, max_updates=None):
    seed_dir = directory / "seed_runs" / f"seed_{seed}"
    record = json.loads((seed_dir / "completed.json").read_text())
    protocol = record["protocol"]
    check_protocol(base, protocol)
    expected_seeds = (seed, protocol["validation_seed_base"] + seed, protocol["test_seed_base"] + seed)
    if expected_seeds != tuple(record[k] for k in ("training_seed", "validation_seed", "test_seed")):
        raise ValueError("Recorded split seeds do not match the protocol")
    if len(set(expected_seeds)) != 3:
        raise ValueError("Split RNG seed numbers overlap")
    original_updates = protocol["updates"]
    updates = min(max_updates, original_updates) if max_updates else original_updates
    training_indices = {"input_and_target": {}, "position_gaze_sequence": {}}
    logged_counts = {}
    with (seed_dir / "training_history.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            logged_counts[int(row["update"])] = int(row["training_valid_timesteps_seen"])

    # Same order as original run. Julia default RNG may be shared by closures:
    # finish consuming one stream before constructing/reseeding another.
    validation, _, val_steps = collect(
        bridge.make_generator(protocol["val_trials"], rseed=expected_seeds[1]), 1)
    for selection in record["selections"].values():
        expected = selection["best_metrics"]["validation_valid_timesteps"]
        if val_steps != expected:
            raise RuntimeError("Historical validation timestep count does not reproduce")
    generator = bridge.make_generator(protocol["batch_size"], rseed=expected_seeds[0])
    train_steps = 0
    checks = 0
    start = time.monotonic()
    for update in range(1, updates + 1):
        hashes, _, steps = fingerprint_batch(generator())
        train_steps += steps
        offset = (update - 1) * protocol["batch_size"]
        for key, values in hashes.items():
            for i, fingerprint in enumerate(values):
                training_indices[key].setdefault(fingerprint, offset + i)
        if update in logged_counts:
            checks += 1
            if train_steps != logged_counts[update]:
                raise RuntimeError(f"Historical training timestep count mismatch at update {update}")
        if update == 1 or update % 250 == 0 or update == updates:
            print(f"seed={seed} replay={update}/{updates}; trials={update * protocol['batch_size']:,}; "
                  f"elapsed={time.monotonic()-start:.0f}s", flush=True)

    test_trials, test_bs = protocol["eval_trials"], protocol["eval_batch_size"]
    if test_trials % test_bs:
        raise ValueError("Test trial count is not divisible by evaluation batch size")
    testing, test_truths, test_steps = collect(
        bridge.make_generator(test_bs, rseed=expected_seeds[2]), test_trials // test_bs)
    with np.load(seed_dir / "raw_evaluation.npz", allow_pickle=False) as saved:
        require_equal_truths(test_truths, saved)

    checkpoint_limits = {"full_run": original_updates * protocol["batch_size"]}
    for name, selection in record["selections"].items():
        checkpoint_limits[f"best_{name}"] = selection["best_training_trials"]
    matching = record.get("accuracy_matching") or {}
    if matching.get("status") == "matched":
        for name in ("factorized", "conjunction"):
            checkpoint_limits[f"matched_{name}"] = matching[name]["training_trials"]
    findings = {}
    for key in training_indices:
        train = training_indices[key]
        findings[key] = {
            "training_unique_trials": len(train),
            "training_repeated_occurrences": updates * protocol["batch_size"] - len(train),
            "train_vs_validation": intersections(train, validation[key]),
            "train_vs_test": intersections(train, testing[key]),
            "validation_vs_test": intersections(first_indices(validation[key]), testing[key]),
            "test_duplicate_occurrences": len(testing[key]) - len(set(testing[key])),
            "test_duplicates_seen_before_checkpoint": {
                name: (sum(1 for h in testing[key] if h in train and train[h] < limit)
                       if limit <= updates * protocol["batch_size"] else None)
                for name, limit in checkpoint_limits.items()},
        }
    return {
        "seed": seed, "scope": "each model's own seed run; not cross-run pooling",
        "status": "replay_verified_against_saved_truths_and_counts",
        "complete_training_stream_audited": updates == original_updates,
        "updates_audited": updates, "original_updates": original_updates,
        "training_trials_audited": updates * protocol["batch_size"],
        "validation_trials": protocol["val_trials"], "test_trials": test_trials,
        "training_valid_timesteps": train_steps, "test_valid_timesteps": test_steps,
        "training_history_count_checks_passed": checks, "exact_saved_test_truth_match": True,
        "findings": findings,
        "limitations": [
            "Historical training inputs and source hashes were not saved. Replay is conditional on the original environment/code.",
            "Test coordinates/labels match exactly; original test sensory inputs were not retained for independent comparison.",
            "Zero complete-trial duplicates does not imply zero shared states/subpaths or no other form of leakage.",
            "SHA-256 collision risk is negligible but not mathematically zero.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("objective_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("trial_overlap_audit"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--quick", action="store_true", help="Audit only first two training updates; NOT a full overlap clearance")
    parser.add_argument("--max-updates", type=int,
                        help="Replay this many updates (diagnostic; omitted checkpoint counts remain null)")
    parser.add_argument("--matched-checkpoint-only", action="store_true",
                        help="Replay through the later of the two accuracy-matched checkpoints for each seed")
    parser.add_argument("--julia-repo", type=Path,
                        help="Require RNNTrialStructures to load from this checked-out repository")
    args = parser.parse_args()
    if sum((args.quick, args.max_updates is not None, args.matched_checkpoint_only)) > 1:
        parser.error("Choose only one of --quick, --max-updates, or --matched-checkpoint-only")
    if args.quick and args.output_dir == Path("trial_overlap_audit"):
        args.output_dir = Path("trial_overlap_audit_quick")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 1:
        parser.error("Choose distinct positive run seeds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Lazy import lets fingerprint/report tests run without Julia or PyTorch.
    # On Windows JuliaCall must initialise before PyTorch.
    import juliacall
    import controlled_comparison as base
    bridge = base.JuliaTrialBridge()
    if args.julia_repo:
        loaded = Path(str(bridge.jl.seval("pathof(RNNTrialStructures)"))).resolve()
        if args.julia_repo.resolve() not in loaded.parents:
            raise RuntimeError(f"Wrong Julia package loaded: {loaded}")
    metadata = provenance(bridge, base)
    completed_results = []
    for seed in args.seeds:
        destination = args.output_dir / f"seed_{seed}.json"
        if destination.exists():
            raise FileExistsError(f"Preserving existing audit: {destination}; use a new output directory")
        max_updates = 2 if args.quick else args.max_updates
        if args.matched_checkpoint_only:
            record = json.loads((args.results_dir / "seed_runs" / f"seed_{seed}" / "completed.json").read_text())
            matching = record.get("accuracy_matching") or {}
            if matching.get("status") != "matched":
                raise RuntimeError(f"Seed {seed} has no accuracy-matched checkpoint pair")
            max_updates = max(int(matching[name]["update"])
                              for name in ("factorized", "conjunction"))
        try:
            result = audit_seed(bridge, base, args.results_dir, seed, max_updates)
        except Exception as exc:
            result = {"seed": seed, "status": "FAILED_NO_VALID_OVERLAP_CONCLUSION", "error": str(exc)}
            destination.write_text(json.dumps({**result, "provenance": metadata}, indent=2))
            raise
        destination.write_text(json.dumps({**result, "provenance": metadata}, indent=2))
        completed_results.append(result)
        for mode, finding in result["findings"].items():
            print(f"seed={seed} {mode}: test trials also in audited training = "
                  f"{finding['train_vs_test']['query_trials_with_exact_duplicate']}/{result['test_trials']}; "
                  f"full stream={result['complete_training_stream_audited']}", flush=True)
    aggregate = {
        "status": "complete" if len(completed_results) == len(args.seeds) else "incomplete",
        "scope": "accuracy-matched training prefixes" if args.matched_checkpoint_only else "requested replay prefixes",
        "seeds": args.seeds,
        "all_saved_test_truths_reproduced_exactly": all(r["exact_saved_test_truth_match"] for r in completed_results),
        "results": {
            mode: {
                "train_vs_test_exact_duplicates": int(sum(
                    r["findings"][mode]["train_vs_test"]["query_trials_with_exact_duplicate"]
                    for r in completed_results)),
                "test_trials_checked": int(sum(r["test_trials"] for r in completed_results)),
                "train_vs_validation_exact_duplicates": int(sum(
                    r["findings"][mode]["train_vs_validation"]["query_trials_with_exact_duplicate"]
                    for r in completed_results)),
                "validation_trials_checked": int(sum(r["validation_trials"] for r in completed_results)),
                "validation_vs_test_exact_duplicates": int(sum(
                    r["findings"][mode]["validation_vs_test"]["query_trials_with_exact_duplicate"]
                    for r in completed_results)),
            }
            for mode in ("input_and_target", "position_gaze_sequence")
        },
        "interpretation": (
            "Exact complete-trajectory identity is tested. Shared individual states or short subpaths are expected "
            "in a finite arena and are not ruled out by a zero result."
        ),
        "provenance": metadata,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

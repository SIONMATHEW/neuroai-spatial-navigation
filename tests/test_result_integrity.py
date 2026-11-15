"""Integrity checks for the compact, versioned result artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_objective_protocol_is_the_reported_clean_run() -> None:
    protocol = read_json(ROOT / "configs" / "objective_comparison.json")
    assert protocol["updates"] == 8000
    assert protocol["batch_size"] == 256
    assert protocol["evaluation_noise_standard_deviation"] == 0.0
    assert protocol["paired_training_batches"] is True
    assert protocol["paired_recurrent_initialization"] is True


def test_best_within_budget_has_five_paired_seeds() -> None:
    path = RESULTS / "objective_comparison" / "best_within_budget"
    rows = read_csv(path / "metrics_by_seed.csv")
    summary = read_json(path / "summary.json")
    assert [int(row["seed"]) for row in rows] == [1, 2, 3, 4, 5]
    assert summary["n_seeds"] == 5
    assert summary["protocol"]["updates"] == 8000
    assert all(int(row["completed_training_updates"]) == 8000 for row in rows)


def test_accuracy_matching_was_validation_only() -> None:
    path = RESULTS / "objective_comparison" / "accuracy_matched"
    summary = read_json(path / "summary.json")
    status = read_json(path / "matching_status.json")
    assert summary["seeds"] == [2, 3, 4, 5]
    assert summary["unmatched_seeds"] == [1]
    assert "not_test_accuracy_matched" in summary["comparison_kind"]
    assert status["comparison_kind"] == summary["comparison_kind"]


def test_environment_results_have_two_conditions_for_each_seed() -> None:
    rows = read_csv(RESULTS / "environment_comparison" / "behavior_metrics.csv")
    observed = {(int(row["seed"]), row["environment"]) for row in rows}
    expected = {(seed, environment) for seed in range(1, 6) for environment in ("open", "occluded")}
    assert observed == expected
    assert all(int(row["checkpoint_update"]) == 4000 for row in rows)
    assert all(float(row["test_noise_sd"]) == 0.25 for row in rows)


def test_environment_hidden_counts_sum_to_256() -> None:
    rows = read_csv(RESULTS / "environment_comparison" / "network_summary.csv")
    category_columns = [
        "place-dominant_units",
        "gaze-dominant_units",
        "additive-mixed_units",
        "interaction-dependent_units",
        "poorly-explained_units",
        "unclassified_units",
    ]
    for row in rows:
        assert sum(int(row[column]) for column in category_columns) == 256


def test_synthetic_encoding_recovery_passed() -> None:
    manifest = read_json(RESULTS / "environment_comparison" / "analysis_manifest.json")
    recovery = manifest["synthetic_recovery"]
    assert recovery["passed"] is True
    assert recovery["median_gain_known_interaction_units"] > 0.5


def test_versioned_result_checksums() -> None:
    manifest = RESULTS / "SHA256SUMS.txt"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative_path = line.split("  ", maxsplit=1)
        payload = (RESULTS / relative_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected

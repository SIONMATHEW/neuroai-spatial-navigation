"""Checks that executable entry points and documentation stay internally consistent."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_entry_points_exist() -> None:
    expected = [
        "objective_comparison/compare_tuned_models.py",
        "objective_comparison/evaluate_frozen_hidden.py",
        "objective_comparison/evaluate_sic_bins.py",
        "objective_comparison/submit_tuned_comparison.sh",
        "environment_comparison/train_environments.py",
        "environment_comparison/evaluate_environments.py",
        "environment_comparison/evaluate_sic_bins.py",
        "environment_comparison/submit_environment_comparison.sh",
    ]
    assert all((ROOT / path).is_file() for path in expected)


def test_slurm_files_use_configurable_project_root() -> None:
    slurm_and_shell = list(ROOT.glob("objective_comparison/*.slurm"))
    slurm_and_shell += list(ROOT.glob("environment_comparison/*.slurm"))
    slurm_and_shell += [
        ROOT / "objective_comparison" / "submit_tuned_comparison.sh",
        ROOT / "environment_comparison" / "submit_environment_comparison.sh",
    ]
    for path in slurm_and_shell:
        text = path.read_text(encoding="utf-8")
        assert "NEUROAI_PROJECT" in text
        assert "rnn-objective-comparison/controlled_objective_comparison" not in text


def test_environment_configuration_matches_evaluation_manifest() -> None:
    config = json.loads((ROOT / "configs" / "environment_comparison.json").read_text())
    manifest = json.loads(
        (ROOT / "results" / "environment_comparison" / "analysis_manifest.json").read_text()
    )
    assert config["training"]["training_noise_sd"] == manifest["training_noise_sd"]
    assert config["evaluation"]["noise_sd"] == manifest["test_and_probe_noise_sd"]
    assert config["evaluation"]["test_trials"] == manifest["test_trials"]
    assert config["evaluation"]["probe_trials"] == manifest["probe_trials"]

"""CPU checks with synthetic trials; Julia/cluster execution is not exercised.

Run: python -m unittest test_sweep_objectives.py
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
import types
import unittest
from unittest.mock import patch
import uuid

# Import the real Python models without starting/downloading a Julia runtime.
sys.modules["juliacall"] = types.ModuleType("juliacall")
import torch
import sweep_factorized as sweep


@contextmanager
def synthetic_output_directory():
    # Python 3.13's secure tempfile directory ACLs can exclude the Windows
    # sandbox token. Ordinary workspace directories avoid that platform issue.
    parent = Path(__file__).resolve().parent
    folder = parent / f"sweep-test-{uuid.uuid4().hex}"
    folder.mkdir()
    try:
        yield str(folder)
    finally:
        if folder.resolve().parent != parent:
            raise RuntimeError("Refusing cleanup outside the test directory")
        shutil.rmtree(folder)


class SyntheticBridge:
    def make_generator(self, ntrials, rseed):
        rng = torch.Generator().manual_seed(rseed)

        def generate():
            x = torch.randn(ntrials, 3, sweep.base.INPUT_DIM, generator=rng)
            joint = torch.randint(sweep.experiment.JOINT_BINS, (ntrials, 3), generator=rng)
            valid = torch.ones_like(joint, dtype=torch.bool)
            valid[:, -1] = False
            joint[:, -1] = -1
            return x, None, joint, valid

        return generate


class SweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_validation_matches_direct_calculation_and_masks_padding(self):
        x, _, joint, valid = SyntheticBridge().make_generator(5, 7)()
        for objective in ("factorized", "conjunction"):
            with self.subTest(objective=objective):
                model = sweep.make_model(objective)
                model.eval()
                with torch.no_grad():
                    output = model(x)
                    if objective == "factorized":
                        p, s, j = sweep.experiment.factorized_predictions(*output)
                        loss, _, _ = sweep.experiment.factorized_loss(*output, joint, valid)
                    else:
                        p, s, j = sweep.experiment.joint_predictions(output)
                        loss = torch.nn.functional.cross_entropy(output[valid], joint[valid])
                actual = sweep.validation_metrics(model, x, joint, valid, 2, objective)
                self.assertAlmostEqual(actual["validation_loss"], loss.item(), places=5)
                for name, pred, truth in (
                    ("place", p, joint // sweep.experiment.SURFACE_BINS),
                    ("surface", s, joint % sweep.experiment.SURFACE_BINS),
                    ("joint", j, joint),
                ):
                    expected = (pred[valid] == truth[valid]).float().mean().item()
                    self.assertAlmostEqual(actual[f"validation_{name}_accuracy"], expected, places=6)
                self.assertEqual(actual["validation_valid_timesteps"], 10)

    def test_training_checkpoint_and_completed_run_reuse(self):
        for objective in ("factorized", "conjunction"):
            with self.subTest(objective=objective), synthetic_output_directory() as folder:
                args = argparse.Namespace(
                    objective=objective, output_dir=Path(folder), overwrite=False,
                    val_trials=4, validation_seed_base=900000, max_updates=2,
                    eval_every=1, eval_batch_size=2, min_delta=5e-4,
                    target_accuracy=0.83, stop_at_target=False, patience_evals=0,
                )
                config = sweep.SweepConfig(0.001, 3, 1.0, "none")
                with patch.object(sweep.base, "unpack_batch", side_effect=lambda raw, device: raw):
                    result = sweep.run_one_seed(SyntheticBridge(), config, 1, args, torch.device("cpu"))
                self.assertEqual(result["completed_updates"], 2)
                self.assertGreater(result["best_validation_loss"], 0)
                checkpoint = torch.load(result["checkpoint"], weights_only=True)
                expected_objective = (
                    "factorized_place_surface" if objective == "factorized"
                    else "joint_place_surface_conjunction"
                )
                self.assertEqual(checkpoint["objective"], expected_objective)
                self.assertEqual(checkpoint["selection_split"], "validation")
                model = sweep.make_model(objective)
                model.load_state_dict(checkpoint["model_state_dict"])
                # No generator must be called when a completed record exists.
                reused = sweep.run_one_seed(None, config, 1, args, torch.device("cpu"))
                self.assertEqual(reused, result)

    def test_serialized_config_indices_build_combined_ranking(self):
        with synthetic_output_directory() as folder:
            for index in (0, 1):
                argv = [
                    "sweep_conjunction.py", "--learning-rates", "0.001", "0.0005",
                    "--batch-sizes", "3", "--gradient-clips", "1",
                    "--max-updates", "2", "--eval-every", "1",
                    "--val-trials", "4", "--eval-batch-size", "2",
                    "--patience-evals", "0", "--device", "cpu",
                    "--config-index", str(index), "--output-dir", folder,
                ]
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(sweep.base, "JuliaTrialBridge", SyntheticBridge),
                    patch.object(sweep.base, "unpack_batch", side_effect=lambda raw, device: raw),
                ):
                    sweep.main(default_objective="conjunction")
            best = json.loads((Path(folder) / "best_config.json").read_text())
            self.assertEqual(best["objective"], "conjunction")
            self.assertEqual(best["completed_configs"], 2)
            self.assertEqual(len(list((Path(folder) / "runs").glob("*.json"))), 2)
            self.assertTrue((Path(folder) / "sweep_ranking.png").exists())

    def test_factorized_cli_default_is_unchanged(self):
        with patch.object(sys, "argv", ["sweep_factorized.py"]):
            args = sweep.parse_args()
        self.assertEqual(args.objective, "factorized")
        self.assertEqual(args.max_updates, 8000)
        self.assertEqual(args.batch_sizes, [64, 128, 256])


if __name__ == "__main__":
    unittest.main()

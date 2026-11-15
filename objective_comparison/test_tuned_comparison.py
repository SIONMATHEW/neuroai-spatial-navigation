"""Synthetic CPU checks; no real Julia/GPU execution is exercised."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from test_sweep_objectives import synthetic_output_directory
import torch
import compare_tuned_models as comparison


class SyntheticBridge:
    def __init__(self):
        self.requested_seeds = []

    def make_generator(self, ntrials, rseed):
        self.requested_seeds.append(rseed)
        rng = torch.Generator().manual_seed(rseed)

        def generate():
            x = torch.randn(ntrials, 3, comparison.base.INPUT_DIM, generator=rng)
            joint = torch.randint(comparison.experiment.JOINT_BINS, (ntrials, 3), generator=rng)
            place = joint // comparison.experiment.SURFACE_BINS
            surface = joint % comparison.experiment.SURFACE_BINS
            pc = torch.from_numpy(comparison.base.normalized_position_centres())
            sc = torch.from_numpy(comparison.base.normalized_surface_centres())
            xy = torch.cat([pc[place], sc[surface]], dim=-1)
            valid = torch.ones_like(joint, dtype=torch.bool)
            valid[:, -1] = False
            joint[:, -1] = -1
            return x, xy, joint, valid

        return generate


def tiny_args(folder, seeds=(1,)):
    argv = ["compare_tuned_models.py", "--updates", "2", "--batch-size", "4",
            "--val-trials", "8", "--validate-every", "1", "--eval-trials", "8",
            "--eval-batch-size", "4", "--device", "cpu", "--output-dir", folder,
            "--seeds", *map(str, seeds)]
    with patch.object(sys, "argv", argv):
        return comparison.parse_args()


class TunedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_identical_recurrent_initialization_and_output_shapes(self):
        f, c = comparison.build_pair(1, torch.device("cpu"))
        self.assertEqual(comparison.recurrent_digest(f), comparison.recurrent_digest(c))
        x = torch.randn(2, 3, comparison.base.INPUT_DIM)
        p, s = f(x)
        self.assertEqual(tuple(p.shape), (2, 3, 21))
        self.assertEqual(tuple(s.shape), (2, 3, 24))
        self.assertEqual(tuple(c(x).shape), (2, 3, 504))

    def test_validation_selects_different_updates_and_restores_weights(self):
        with synthetic_output_directory() as folder:
            args = tiny_args(folder)
            bridge = SyntheticBridge()

            def metrics(accuracy):
                return {"validation_joint_accuracy": accuracy,
                        "validation_place_accuracy": accuracy,
                        "validation_surface_accuracy": accuracy,
                        "validation_loss": 1.0, "validation_valid_timesteps": 16}

            # Factorized peaks at update 1; conjunction at update 2.
            scores = [metrics(0.9), metrics(0.1), metrics(0.1), metrics(0.9)]
            with (
                patch.object(comparison.base, "unpack_batch", side_effect=lambda raw, device: raw),
                patch.object(comparison.sweep, "validation_metrics", side_effect=scores),
            ):
                f, c, selected, _, timesteps = comparison.train_and_select(
                    bridge, 1, args, torch.device("cpu"), Path(folder)
                )
            self.assertEqual(selected["factorized"]["best_update"], 1)
            self.assertEqual(selected["conjunction"]["best_update"], 2)
            self.assertEqual(timesteps, 16)
            self.assertEqual(bridge.requested_seeds, [900001, 1])
            for name, model in (("factorized", f), ("conjunction", c)):
                checkpoint = torch.load(Path(folder) / "checkpoints" / f"{name}_best.pt", weights_only=True)
                for key, tensor in model.state_dict().items():
                    self.assertTrue(torch.equal(tensor, checkpoint["model_state_dict"][key]))

    def test_test_separation_reuse_and_two_seed_aggregation(self):
        with synthetic_output_directory() as folder:
            args = tiny_args(folder, seeds=(1, 2))
            comparison.write_json(Path(folder) / "config.json", comparison.protocol(args))
            bridge = SyntheticBridge()
            with patch.object(comparison.base, "unpack_batch", side_effect=lambda raw, device: raw):
                for seed in args.seeds:
                    record = comparison.run_seed(bridge, seed, args, torch.device("cpu"))
                    self.assertEqual(record["test_seed"], 2000000 + seed)
                    self.assertEqual(record["metrics"]["training_trials_seen"], 8)
                    reused = comparison.run_seed(None, seed, args, torch.device("cpu"))
                    self.assertEqual(reused["selections"], record["selections"])
            self.assertEqual(bridge.requested_seeds, [900001, 1, 2000001, 900002, 2, 2000002])
            comparison.aggregate(args)
            summary = json.loads((Path(folder) / "summary.json").read_text())
            self.assertEqual(summary["n_seeds"], 2)
            self.assertEqual(len(summary["paired_test_differences"]), 3)
            self.assertEqual(len(list(Path(folder).glob("*.png"))), 10)

    def test_missing_seed_is_not_silently_aggregated(self):
        with synthetic_output_directory() as folder:
            args = tiny_args(folder)
            with self.assertRaisesRegex(RuntimeError, "not complete"):
                comparison.aggregate(args)

    def test_dual_comparison_uses_same_training_and_test_trials(self):
        with synthetic_output_directory() as folder:
            args = tiny_args(folder)
            args.also_accuracy_matched = True
            args.match_validate_every = 1
            bridge = SyntheticBridge()

            def metrics(accuracy):
                return {"validation_joint_accuracy": accuracy,
                        "validation_place_accuracy": accuracy,
                        "validation_surface_accuracy": accuracy,
                        "validation_loss": 1.0, "validation_valid_timesteps": 16}

            # Matching takes update 1 for factorized, update 2 for conjunction;
            # best-budget selection independently takes update 2 for both.
            scores = [metrics(0.830), metrics(0.5), metrics(0.95), metrics(0.831)]
            with (
                patch.object(comparison.base, "unpack_batch", side_effect=lambda raw, device: raw),
                patch.object(comparison.sweep, "validation_metrics", side_effect=scores),
            ):
                record = comparison.run_seed(bridge, 1, args, torch.device("cpu"))
            matching = record["accuracy_matching"]
            self.assertEqual(matching["status"], "matched")
            self.assertEqual(matching["factorized"]["update"], 1)
            self.assertEqual(matching["conjunction"]["update"], 2)
            self.assertEqual(record["selections"]["factorized"]["best_update"], 2)
            self.assertEqual(bridge.requested_seeds, [900001, 1, 2000001, 2000001])
            directory = Path(folder) / "seed_runs" / "seed_1"
            with (
                comparison.np.load(directory / "raw_evaluation.npz") as best,
                comparison.np.load(directory / "accuracy_matched_raw_evaluation.npz") as matched,
            ):
                self.assertTrue(comparison.np.array_equal(best["true_joint"], matched["true_joint"]))
                self.assertTrue(comparison.np.array_equal(best["true_xy"], matched["true_xy"]))
            comparison.aggregate(args)
            summary = json.loads((Path(folder) / "accuracy_matched" / "summary.json").read_text())
            self.assertEqual(summary["n_seeds"], 1)
            self.assertEqual(summary["unmatched_seeds"], [])
            self.assertEqual(len(list((Path(folder) / "accuracy_matched").glob("*.png"))), 8)
            import csv
            with (Path(folder) / "accuracy_matched" / "checkpoint_comparison_by_seed.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row["conjunction_minus_factorized_selected_update"]), 1)
            self.assertEqual(float(row["conjunction_minus_factorized_training_trials"]), 4)
            self.assertAlmostEqual(float(row["conjunction_minus_factorized_validation_joint_accuracy_percentage_points"]), 0.1)

    def test_match_requires_pair_gap_and_target_band(self):
        with synthetic_output_directory() as folder:
            args = tiny_args(folder)
            selections = {
                "factorized": {"match_candidates": [{"accuracy": 0.825, "update": 10}]},
                "conjunction": {"match_candidates": [{"accuracy": 0.835, "update": 20}]},
            }
            self.assertEqual(comparison.select_accuracy_match(selections, args)["status"], "unmatched")
            selections["conjunction"]["match_candidates"][0]["accuracy"] = 0.827
            self.assertEqual(comparison.select_accuracy_match(selections, args)["status"], "matched")
            selections["factorized"]["match_candidates"][0]["accuracy"] = 0.9
            selections["conjunction"]["match_candidates"][0]["accuracy"] = 0.9
            self.assertEqual(comparison.select_accuracy_match(selections, args)["status"], "unmatched")

    def test_quick_unmatched_is_reported_not_fabricated(self):
        with synthetic_output_directory() as folder:
            args = tiny_args(folder)
            args.also_accuracy_matched = True
            with patch.object(comparison.base, "unpack_batch", side_effect=lambda raw, device: raw):
                record = comparison.run_seed(SyntheticBridge(), 1, args, torch.device("cpu"))
            self.assertEqual(record["accuracy_matching"]["status"], "unmatched")
            comparison.aggregate(args)
            summary = json.loads((Path(folder) / "accuracy_matched" / "summary.json").read_text())
            self.assertEqual(summary["n_seeds"], 0)
            self.assertEqual(summary["unmatched_seeds"], [1])

    def test_defaults_and_disjoint_seed_validation(self):
        with patch.object(sys, "argv", ["compare_tuned_models.py"]):
            args = comparison.parse_args()
        self.assertEqual(args.seeds, [1, 2, 3, 4, 5])
        self.assertEqual(args.updates, 8000)
        self.assertEqual(args.batch_size, 256)
        self.assertEqual(args.gradient_clip, 3)
        self.assertEqual(args.test_seed_base, 2000000)
        with patch.object(sys, "argv", ["compare_tuned_models.py", "--test-seed-base", "900000"]):
            with self.assertRaises(SystemExit):
                comparison.parse_args()


if __name__ == "__main__":
    unittest.main()

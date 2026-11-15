"""Offline synthetic checks only; do not establish real-data non-overlap."""
import unittest
import numpy as np
import audit_trial_overlap as audit


class AuditTests(unittest.TestCase):
    def raw(self):
        x = np.arange(36 * 4 * 2, dtype=np.float32).reshape(36, 4, 2)
        y = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)
        w = np.ones_like(y)
        w[:, 3, 0] = 0
        j = np.arange(8, dtype=np.int64).reshape(4, 2)
        return x, y, w, j

    def test_padding_is_excluded(self):
        raw = self.raw()
        hashes, _, count = audit.fingerprint_batch(raw)
        changed = tuple(a.copy() for a in raw)
        changed[0][:, 3, 0] = 1000
        changed[1][:, 3, 0] = -999
        changed[3][3, 0] = -1
        self.assertEqual(hashes, audit.fingerprint_batch(changed)[0])
        self.assertEqual(count, 7)

    def test_inputs_and_coordinates_distinguished(self):
        raw = self.raw()
        first = audit.fingerprint_batch(raw)[0]
        raw[0][0, 0, 0] += 1
        changed = audit.fingerprint_batch(raw)[0]
        self.assertNotEqual(first["input_and_target"][0], changed["input_and_target"][0])
        self.assertEqual(first["position_gaze_sequence"], changed["position_gaze_sequence"])

    def test_flatten_order_matches_saved_evaluation(self):
        _, truths, _ = audit.fingerprint_batch(self.raw())
        np.testing.assert_array_equal(truths["true_joint"], [0, 2, 4, 1, 3, 5, 7])

    def test_duplicate_counts_and_earliest_index(self):
        train = audit.first_indices([b"a", b"b", b"a"])
        result = audit.intersections(train, [b"a", b"a", b"c"])
        self.assertEqual(result["query_trials_with_exact_duplicate"], 2)
        self.assertEqual(result["examples"][0]["reference_trial_index_0based"], 0)

    def test_truth_replay_mismatch_stops_clearance(self):
        _, truths, _ = audit.fingerprint_batch(self.raw())
        audit.require_equal_truths(truths, truths)
        bad = {k: a.copy() for k, a in truths.items()}
        bad["true_joint"][0] += 1
        with self.assertRaises(RuntimeError):
            audit.require_equal_truths(truths, bad)

    def test_length_changes_fingerprint(self):
        self.assertNotEqual(audit.digest_arrays(np.ones((3, 2), np.float32)),
                            audit.digest_arrays(np.ones((2, 3), np.float32)))


if __name__ == "__main__":
    unittest.main()

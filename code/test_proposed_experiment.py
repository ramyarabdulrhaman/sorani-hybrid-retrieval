import json
import os
import tempfile
import unittest

import numpy as np

import proposed_experiment as experiment


class ProposedExperimentTest(unittest.TestCase):
    def test_resume_log_returns_only_completed_fold_records(self):
        records = [
            {"partition_seed": 42, "seed": 1, "fold": 0, "mode": "raw"},
            {
                "partition_seed": 42,
                "seed": 1,
                "fold": 0,
                "FINAL_selected": {
                    "val": 98.0, "mode": "raw", "lr": 1e-5, "epochs": 2, "w": [0.6, 0.1, 0.3]
                },
                "checkpoint_dir": "results/checkpoints/partition_42/fold_0/seed_1",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "selection.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(records, f)

            loaded, completed = experiment.load_resume_log(path)

        self.assertEqual(loaded, records)
        self.assertEqual(list(completed), [(42, 1, 0)])

    def test_resume_log_deduplicates_identical_completed_runs(self):
        selected = {
            "val": 98.0, "mode": "raw", "lr": 1e-5, "epochs": 2, "w": [0.6, 0.1, 0.3]
        }
        completed = {"partition_seed": 42, "seed": 1, "fold": 0, "FINAL_selected": selected}
        records = [{"mode": "raw"}, completed, completed.copy()]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "selection.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(records, f)

            normalized, completed_runs = experiment.load_resume_log(path)

        self.assertEqual(normalized, records[:2])
        self.assertEqual(list(completed_runs), [(42, 1, 0)])

    def test_resume_log_rejects_conflicting_completed_runs(self):
        selected = {
            "val": 98.0, "mode": "raw", "lr": 1e-5, "epochs": 2, "w": [0.6, 0.1, 0.3]
        }
        conflicting = {
            "partition_seed": 42, "seed": 1, "fold": 0,
            "FINAL_selected": {**selected, "val": 99.0},
        }
        records = [
            {"partition_seed": 42, "seed": 1, "fold": 0, "FINAL_selected": selected},
            conflicting,
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "selection.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(records, f)

            with self.assertRaisesRegex(ValueError, "Conflicting duplicate completed run"):
                experiment.load_resume_log(path)

    def test_validation_candidates_match_test_size_and_keep_gold_first(self):
        queries = ["qirmizi car", "green tree"]
        passages = [
            "qirmizi car story",
            "green tree story",
            "qirmizi car other",
            "green tree other",
            "unrelated text",
            "another unrelated passage",
        ]
        validation = np.array([0, 1])
        hard_pool = np.array([2, 3, 4, 5])

        candidates = experiment.validation_candidate_indices(
            queries, passages, validation, hard_pool)

        self.assertEqual(candidates.shape, (2, 4))
        np.testing.assert_array_equal(candidates[:, :2], np.tile(validation, (2, 1)))
        self.assertTrue(set(candidates[:, 2:].ravel()).issubset(set(hard_pool)))
        for row in candidates:
            self.assertEqual(len(set(row)), len(row))

    def test_validation_candidates_reject_overlapping_pools(self):
        with self.assertRaises(ValueError):
            experiment.validation_candidate_indices(
                ["query"], ["passage"], np.array([0]), np.array([0]))

    def test_rank_uses_per_query_gold_position(self):
        scores = np.array([[0.5, 0.8, 0.3], [0.2, 0.1, 0.9]])
        ranks = experiment.rank_of_gold(scores, np.array([0, 2]))
        np.testing.assert_array_equal(ranks, [2, 1])

    def test_candidate_embedding_scores_match_explicit_dot_products(self):
        queries = np.array([[1.0, 0.0], [0.0, 1.0]])
        passages = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        candidates = np.array([[0, 2], [1, 0]])
        scores = experiment.score_candidates(queries, passages, candidates)
        expected = np.array([[1.0, -1.0], [1.0, 0.0]])
        np.testing.assert_allclose(scores, expected)


if __name__ == "__main__":
    unittest.main()

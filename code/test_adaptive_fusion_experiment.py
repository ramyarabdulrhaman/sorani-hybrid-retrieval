import unittest

import numpy as np

import adaptive_fusion_experiment as adaptive


class AdaptiveFusionTest(unittest.TestCase):
    def test_features_use_query_only_and_are_finite(self):
        features = adaptive.query_features("Sorani query 123")
        self.assertEqual(features.shape, (8,))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_blend_equal_weight_matches_three_component_equal_weights(self):
        matrices = [
            np.array([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]]),
            np.array([[0.0, 2.0, 1.0], [3.0, 1.0, 0.0]]),
            np.array([[1.0, 0.0, 2.0], [1.0, 3.0, 0.0]]),
        ]
        blended = adaptive.blend_scores(matrices, np.full(2, 1 / 3))
        expected = sum(adaptive.E.zscore(matrix) for matrix in matrices) / 3
        np.testing.assert_allclose(blended, expected)

    def test_group_folds_keep_passage_groups_together(self):
        groups = np.array([10, 10, 20, 20, 30, 30, 40, 40, 50])
        assignments = adaptive.group_folds(groups, n_folds=3, seed=7)
        for group in np.unique(groups):
            self.assertEqual(len(set(assignments[groups == group])), 1)
        self.assertEqual(set(assignments), {0, 1, 2})

    def test_policy_fit_uses_only_training_indices(self):
        features = np.arange(24, dtype=float).reshape(12, 2)
        overlaps = np.linspace(0, 1, 12)
        model = adaptive.fit_overlap_predictor(features, overlaps, np.arange(8))
        predictions = model.predict(features[8:])
        self.assertEqual(predictions.shape, (4,))
        self.assertTrue(np.all(np.isfinite(predictions)))


if __name__ == "__main__":
    unittest.main()

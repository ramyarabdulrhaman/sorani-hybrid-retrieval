import unittest

import numpy as np

import belebele_experiment as experiment
import belebele_robustness as robustness


class BelebeleExperimentTest(unittest.TestCase):
    def test_pessimistic_gold_ties(self):
        scores = np.array([[2.0, 2.0, 1.0], [0.0, 3.0, 1.0]])
        np.testing.assert_array_equal(experiment.ranks(scores, np.array([0, 2])), [2, 2])

    def test_weight_selection_requires_finetuned_component(self):
        components = [np.array([[2.0, 1.0]]), np.array([[1.0, 2.0]]),
                      np.array([[1.0, 0.0]])]
        weights, score = experiment.select_weights(components, np.array([0]), 1)
        self.assertGreaterEqual(weights[1], 0.1)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertEqual(score, 100.0)

    def test_cluster_bootstrap_of_equal_systems(self):
        scores = np.array([1.0, 0.5, 0.25, 0.0])
        self.assertEqual(experiment.cluster_ci(scores, scores, np.array([1, 1, 2, 2])), [0, 0])

    def test_dataset_and_split(self):
        rows, passages, gold = experiment.load_dataset()
        self.assertEqual((len(rows), len(passages)), (900, 488))
        perm = np.random.default_rng(experiment.SEED).permutation(len(passages))
        validation = np.isin(gold, perm[:len(perm) // 2])
        self.assertEqual(int(validation.sum()), 453)
        self.assertTrue(set(gold[validation]).isdisjoint(set(gold[~validation])))

    def test_grouped_cross_fit(self):
        gold = np.array([0, 0, 1, 2, 2, 3, 4, 5])
        folds = robustness.group_folds(gold, 6)
        for group in np.unique(gold):
            self.assertEqual(len(set(folds[gold == group])), 1)
        self.assertEqual(set(folds), {0, 1, 2})
        scores = np.full((len(gold), 6), 0.1)
        scores[np.arange(len(gold)), gold] = 1.0
        models = {seed: scores for seed in robustness.TRAIN_SEEDS}
        ranks, selection = robustness.evaluate(scores, scores, scores, models, gold, folds)
        self.assertEqual(len(selection), 9)
        self.assertTrue(all((r == 1).all() for r in ranks.values()))
        summary, comparisons = robustness.summarize(ranks, gold)
        self.assertEqual(summary[robustness.PROPOSED]["MRR@10"], 100)
        self.assertEqual(comparisons[robustness.BASELINE]["mean_MRR@10_difference"], 0)


if __name__ == "__main__":
    unittest.main()

import unittest

import morphology_intervention as intervention
import proposed_experiment as experiment
import real_experiments as retrieval


class MorphologyInterventionTest(unittest.TestCase):
    def test_intervention_splits_suffix_and_preserves_no_suffix_text(self):
        word = "کتابەکەمان"
        transformed = experiment.split_text(word)
        self.assertNotEqual(transformed, word)
        self.assertEqual(experiment.split_text("سورانی"), "سورانی")

    def test_tokenization_summary_is_finite(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            experiment.MODEL_ID, local_files_only=True)
        stats = intervention.tokenize_stats(tokenizer, ["Sorani sentence"])
        self.assertEqual(stats["texts"], 1)
        self.assertGreaterEqual(stats["mean_pieces_per_text"], 0)
        self.assertGreaterEqual(stats["mean_pieces_per_word"], 0)

    def test_ranks_use_candidate_local_gold_order(self):
        import numpy as np

        scores = np.array([[0.9, 0.1], [0.2, 0.8]])
        ranks = retrieval.rank_of_gold(scores)
        np.testing.assert_array_equal(ranks, [1, 1])


if __name__ == "__main__":
    unittest.main()

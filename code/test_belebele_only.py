import unittest

import numpy as np

import belebele_experiment as B
import belebele_only_experiment as E


class BelebeleOnlyTest(unittest.TestCase):
    def test_passage_partition_has_no_overlap(self):
        rows, passages, gold = B.load_dataset()
        groups = E.partition(gold, len(passages))
        sets = [set(gold[groups == i]) for i in range(3)]
        self.assertEqual([len(s) for s in sets], [244, 122, 122])
        self.assertTrue(all(not sets[i] & sets[j] for i in range(3) for j in range(i)))
        self.assertEqual([int((groups == i).sum()) for i in range(3)], [452, 229, 219])
        questions = [set(rows[j]["question"].strip() for j in np.flatnonzero(groups == i))
                     for i in range(3)]
        self.assertTrue(all(not questions[i] & questions[j] for i in range(3) for j in range(i)))

import numpy as np

import unittest

import scipy.stats as st
from src.utilities import data_generation as dg


class TestDataGeneration(unittest.TestCase):
    def test_cond_prob(self):
        test = dg.gummy_worm_dataset()
        delta = 0.25

        samples, labels = test.generate_data(50000)

        combined = list(zip(samples, labels))

        test_samples = np.array([
            [5, 2.5],
            [8, 6],
            [10, 10.5],
        ])
        samples_near_test_samples = [[comb for comb in combined if np.linalg.norm(comb[0] - test_sample) <= delta] for test_sample in test_samples]

        samples_near_test_samples_0 = [list(filter(lambda tup: tup[1] == 0, near_samples)) for near_samples in samples_near_test_samples]
        samples_near_test_samples_1 = [list(filter(lambda tup: tup[1] == 1, near_samples)) for near_samples in samples_near_test_samples]
        true_prob0 = np.array([test.cond_prob(s, 0) for s in test_samples])
        true_prob1 = np.array([test.cond_prob(s, 1) for s in test_samples])

        num_near_test_samples_0 = np.array(list(map(lambda l: len(l), samples_near_test_samples_0)))
        num_near_test_samples_1 = np.array(list(map(lambda l: len(l), samples_near_test_samples_1)))

        rel_freq0 = (1.0 * num_near_test_samples_0) / (num_near_test_samples_0 + num_near_test_samples_1)
        rel_freq1 = (1.0 * num_near_test_samples_1) / (num_near_test_samples_0 + num_near_test_samples_1)

        print(true_prob0, " : ", rel_freq0)
        print(true_prob1, " : ", rel_freq1)

        for result, expected in zip(true_prob0, rel_freq0):
            self.assertAlmostEqual(expected, result, delta=0.01)
        for result, expected in zip(true_prob1, rel_freq1):
            self.assertAlmostEqual(expected, result, delta=0.01)
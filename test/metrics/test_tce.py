import unittest

import numpy as np

from src.metrics.tce import tce


class TestTCE(unittest.TestCase):
    def test_tce__is_zero(self):
        pred_prob = np.array([[0.0, 1]] * 10000)
        true_labels = np.array([1] * 10000)

        result = tce(pred_prob, true_labels)
        self.assertAlmostEqual(0, result)


    def test_tce__is_one(self):
        pred_prob = np.array([[0.0, 1]] * 10000)
        true_labels = np.array([0] * 10000)

        result = tce(pred_prob, true_labels)
        self.assertAlmostEqual(100.0, result)


    def test_tce__is_a_half(self):
        # When is it not 0.0 or 100.0 ??? #
        pred_prob = np.array([[0.001, 0.998]] * 10000)
        true_labels = np.array([1] * 10000)

        result = tce(pred_prob, true_labels)
        self.assertAlmostEqual(50.0, result)
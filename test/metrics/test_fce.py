import unittest

import numpy as np

from src.metrics.fce import fce


class TestFCE(unittest.TestCase):
    def test_fce__is_zero(self):
        pred_prob = np.array([[0.0, 1], [0.0, 1]])
        true_labels = np.array([1, 1])

        result = fce(pred_prob, true_labels, n_bins=1)
        self.assertAlmostEqual(0, result)

        pred_prob = np.array([[0.0, 1]] * 1000)
        true_labels = np.array([1] * 1000)

        result = fce(pred_prob, true_labels, n_bins=10)
        self.assertAlmostEqual(0, result)


    def test_fce__is_one(self):
        pred_prob = np.array([[0.0, 1], [0.0, 1]])
        true_labels = np.array([0, 0])

        result = fce(pred_prob, true_labels, n_bins=1)
        self.assertAlmostEqual(1, result)

        pred_prob = np.array([[0.0, 1]] * 1000)
        true_labels = np.array([0] * 1000)

        result = fce(pred_prob, true_labels, n_bins=10)
        self.assertAlmostEqual(1.0, result)


    def test_fce__is_a_half(self):
        pred_prob = np.array([[0.0, 1], [0.0, 1]])
        true_labels = np.array([0, 1])

        result = fce(pred_prob, true_labels, n_bins=1)
        self.assertAlmostEqual(0.5, result)

        pred_prob = np.array([[0.0, 1]] * 1000)
        true_labels = np.array([0, 1] * 500)

        result = fce(pred_prob, true_labels, n_bins=10)
        self.assertAlmostEqual(0.5, result)
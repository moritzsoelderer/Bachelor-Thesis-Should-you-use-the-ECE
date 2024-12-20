import unittest

import numpy as np

from src.metrics.ace import ace


class TestACE(unittest.TestCase):
    def test_ace__is_zero(self):
        pred_prob = np.array([[0.0, 1], [0.0, 1]])
        true_labels = np.array([1, 1])

        result = ace(pred_prob, true_labels, n_ranges=1)
        self.assertAlmostEqual(0, result)

        pred_prob = np.array([[0.0, 1]] * 10000)
        true_labels = np.array([1] * 10000)

        result = ace(pred_prob, true_labels, n_ranges=100)
        self.assertAlmostEqual(0, result)


    def test_ace__is_one(self):
        pred_prob = np.array([[0.0, 1], [0.0, 1]])
        true_labels = np.array([0, 0])

        result = ace(pred_prob, true_labels, n_ranges=1)
        self.assertAlmostEqual(1, result)

        pred_prob = np.array([[0.0, 1]] * 10000)
        true_labels = np.array([0] * 10000)

        result = ace(pred_prob, true_labels, n_ranges=100)
        self.assertAlmostEqual(1.0, result)


    def test_ace__is_a_half(self):
        pred_prob = np.array([[0.0, 1], [0.0, 1]])
        true_labels = np.array([0, 1])

        result = ace(pred_prob, true_labels, n_ranges=1)
        self.assertAlmostEqual(0.5, result)

        pred_prob = np.array([[0.0, 1]] * 10000)
        true_labels = np.array([0, 1] * 5000)

        result = ace(pred_prob, true_labels, n_ranges=100)
        self.assertAlmostEqual(0.5, result)

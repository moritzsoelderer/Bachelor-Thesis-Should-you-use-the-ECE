import unittest

import numpy as np

from src.metrics.true_ece import true_ece


class TestTrueEce(unittest.TestCase):
    def test_true_ece__is_zero(self):
        pred_prob = np.array([[0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.7, 0.3], [0.7, 0.3], [0.7, 0.3]])
        true_prob = np.array([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2]])

        result = true_ece(pred_prob, true_prob)
        self.assertAlmostEqual(0, result)

    def test_true_ece__is_one(self):
        pred_prob = np.array(
            [[1, 0.0], [1, 0.0], [1, 0.0], [1, 0.0], [1, 0.0], [1, 0.0], [1, 0.0], [1, 0.0], [1, 0.0]])
        true_prob = np.array(
            [[0.0, 1], [0.0, 1], [0.0, 1], [0.0, 1], [0.0, 1], [0.0, 1], [0.0, 1], [0.0, 1], [0.0, 1]])

        result = true_ece(pred_prob, true_prob)
        self.assertAlmostEqual(1, result)


    def test_true_ece__is_a_half(self):
        pred_prob = np.array(
            [[1, 0.0], [1, 0.0], [1, 0.0], [1, 0.0], [0.0, 1], [0.0, 1], [0.0, 1], [0.0, 1]])
        true_prob = np.array(
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        result = true_ece(pred_prob, true_prob)
        self.assertAlmostEqual(.5, result)
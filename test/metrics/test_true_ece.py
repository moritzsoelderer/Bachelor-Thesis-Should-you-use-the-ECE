import unittest

import numpy as np

from src.metrics.true_ece import true_ece_binned


class TestTrueEce(unittest.TestCase):
    def test_true_ece__is_zero(self):
        p_pred = np.array([[0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.7, 0.3], [0.7, 0.3], [0.7, 0.3]])
        p_true = np.array([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2]])

        result, _ = true_ece_binned(p_pred, p_true, np.linspace(0, 1, 15))
        self.assertAlmostEqual(0, result)

    def test_true_ece__is_one(self):
        p_pred = np.array(
            [[1, 0.0], [1, 0.0], [1, 0.0], [1, 0.0], [1, 0.0], [1, 0.0], [1, 0.0], [1, 0.0], [1, 0.0]])
        p_true = np.array(
            [[0.0, 1], [0.0, 1], [0.0, 1], [0.0, 1], [0.0, 1], [0.0, 1], [0.0, 1], [0.0, 1], [0.0, 1]])

        result, _ = true_ece_binned(p_pred, p_true, np.linspace(0, 1, 15))
        self.assertAlmostEqual(1, result)


    def test_true_ece__is_a_half(self):
        p_pred = np.array(
            [[1, 0.0], [1, 0.0], [1, 0.0], [1, 0.0], [0.0, 1], [0.0, 1], [0.0, 1], [0.0, 1]])
        p_true = np.array(
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        result, _ = true_ece_binned(p_pred, p_true, np.linspace(0, 1, 15))
        self.assertAlmostEqual(.5, result)
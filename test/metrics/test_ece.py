import unittest

import numpy as np

from src.metrics.ece import ece


class TestECE(unittest.TestCase):
    def test_ece__is_zero(self):
        p_pred = np.array([[0.0, 1], [0.0, 1]])
        y_true = np.array([1, 1])

        result = ece(p_pred, y_true, n_bins=1)
        self.assertAlmostEqual(0, result)

        p_pred = np.array([[0.0, 1]] * 10000)
        y_true = np.array([1] * 10000)

        result = ece(p_pred, y_true, n_bins=100)
        self.assertAlmostEqual(0, result)


    def test_ece__is_one(self):
        p_pred = np.array([[0.0, 1], [0.0, 1]])
        y_true = np.array([0, 0])

        result = ece(p_pred, y_true, n_bins=1)
        self.assertAlmostEqual(1, result)

        p_pred = np.array([[0.0, 1]] * 10000)
        y_true = np.array([0] * 10000)

        result = ece(p_pred, y_true, n_bins=100)
        self.assertAlmostEqual(1.0, result)


    def test_ece__is_a_half(self):
        p_pred = np.array([[0.0, 1], [0.0, 1]])
        y_true = np.array([0, 1])

        result = ece(p_pred, y_true, n_bins=1)
        self.assertAlmostEqual(0.5, result)

        p_pred = np.array([[0.0, 1]] * 10000)
        y_true = np.array([0, 1] * 5000)

        result = ece(p_pred, y_true, n_bins=100)
        self.assertAlmostEqual(0.5, result)
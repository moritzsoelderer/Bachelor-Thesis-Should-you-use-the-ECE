import unittest

import numpy as np

from src.metrics.fce import fce


class TestFCE(unittest.TestCase):
    def test_fce__is_zero(self):
        p_pred = np.array([[0.0, 1], [0.0, 1]])
        y_true = np.array([1, 1])

        result = fce(p_pred, y_true, n_bins=1)
        self.assertAlmostEqual(0, result)

        p_pred = np.array([[0.0, 1]] * 1000)
        y_true = np.array([1] * 1000)

        result = fce(p_pred, y_true, n_bins=10)
        self.assertAlmostEqual(0, result)


    def test_fce__is_one(self):
        p_pred = np.array([[0.0, 1], [0.0, 1]])
        y_true = np.array([0, 0])

        result = fce(p_pred, y_true, n_bins=1)
        self.assertAlmostEqual(1, result)

        p_pred = np.array([[0.0, 1]] * 1000)
        y_true = np.array([0] * 1000)

        result = fce(p_pred, y_true, n_bins=10)
        self.assertAlmostEqual(1.0, result)


    def test_fce__is_a_half(self):
        p_pred = np.array([[0.0, 1], [0.0, 1]])
        y_true = np.array([0, 1])

        result = fce(p_pred, y_true, n_bins=1)
        self.assertAlmostEqual(0.5, result)

        p_pred = np.array([[0.0, 1]] * 1000)
        y_true = np.array([0, 1] * 500)

        result = fce(p_pred, y_true, n_bins=10)
        self.assertAlmostEqual(0.5, result)
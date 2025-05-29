import unittest

import numpy as np

from src.metrics.ksce import ksce


class TestKSCE(unittest.TestCase):
    def test_ksce__is_zero(self):
        p_pred = np.array([[0.0, 1], [0.0, 1]])
        y_true = np.array([1, 1])

        result = ksce(p_pred, y_true)
        self.assertAlmostEqual(0, result)

        p_pred = np.array([[0.0, 1]] * 10000)
        y_true = np.array([1] * 10000)

        result = ksce(p_pred, y_true)
        self.assertAlmostEqual(0, result)


    def test_ksce__is_one(self):
        p_pred = np.array([[0.0, 1], [0.0, 1]])
        y_true = np.array([0, 0])

        result = ksce(p_pred, y_true)
        self.assertAlmostEqual(1, result)

        p_pred = np.array([[0.0, 1]] * 10000)
        y_true = np.array([0] * 10000)

        result = ksce(p_pred, y_true)
        self.assertAlmostEqual(1.0, result)


    def test_ksce__is_a_half(self):
        p_pred = np.array([[0.0, 1], [0.0, 1]])
        y_true = np.array([0, 1])

        result = ksce(p_pred, y_true)
        self.assertAlmostEqual(0.5, result)

        p_pred = np.array([[0.0, 1]] * 10000)
        y_true = np.array([0, 1] * 5000)

        result = ksce(p_pred, y_true)
        self.assertAlmostEqual(0.5, result)

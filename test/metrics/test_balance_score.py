import unittest

import numpy as np

from src.metrics.balance_score import balance_score


class TestBalanceScore(unittest.TestCase):
    def test_balance_score__is_zero(self):
        p_pred = np.array([[0.0, 1], [0.0, 1]])
        y_true = np.array([1, 1])

        result = balance_score(p_pred, y_true)
        self.assertAlmostEqual(0, result)

        p_pred = np.array([[0.0, 1]] * 10000)
        y_true = np.array([1] * 10000)

        result = balance_score(p_pred, y_true)
        self.assertAlmostEqual(0, result)


    def test_balance_score__is_a_half(self):
        p_pred = np.array([[0.5, 0.5], [0.5, 0.5]])
        y_true = np.array([1, 1])

        result = balance_score(p_pred, y_true)
        self.assertAlmostEqual(0.5, result)

        p_pred = np.array([[0.5, 0.5]] * 10000)
        y_true = np.array([1] * 10000)

        result = balance_score(p_pred, y_true)
        self.assertAlmostEqual(0.5, result)



    def test_balance_score__is_minus_one(self):
        p_pred = np.array([[0.0, 1], [0.0, 1]])
        y_true = np.array([0, 0])

        result = balance_score(p_pred, y_true)
        self.assertAlmostEqual(-1.0, result)

        p_pred = np.array([[0.0, 1]] * 10000)
        y_true = np.array([0] * 10000)

        result = balance_score(p_pred, y_true)
        self.assertAlmostEqual(-1.0, result)


    def test_balance_score__is_minus_a_half(self):
        p_pred = np.array([[0.0, 1], [0.0, 1]])
        y_true = np.array([0, 1])

        result = balance_score(p_pred, y_true)
        self.assertAlmostEqual(-0.5, result)

        p_pred = np.array([[0.0, 1]] * 10000)
        y_true = np.array([0, 1] * 5000)

        result = balance_score(p_pred, y_true)
        self.assertAlmostEqual(-0.5, result)
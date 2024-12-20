import unittest

import numpy as np

from src.metrics.ksce import ksce


class TestKSCE(unittest.TestCase):
    def test_ksce__is_zero(self):
        pred_prob = np.array([[0.0, 1], [0.0, 1]])
        true_labels = np.array([1, 1])

        result = ksce(pred_prob, true_labels)
        self.assertAlmostEqual(0, result)

        pred_prob = np.array([[0.0, 1]] * 10000)
        true_labels = np.array([1] * 10000)

        result = ksce(pred_prob, true_labels)
        self.assertAlmostEqual(0, result)


    def test_ksce__is_one(self):
        pred_prob = np.array([[0.0, 1], [0.0, 1]])
        true_labels = np.array([0, 0])

        result = ksce(pred_prob, true_labels)
        self.assertAlmostEqual(1, result)

        pred_prob = np.array([[0.0, 1]] * 10000)
        true_labels = np.array([0] * 10000)

        result = ksce(pred_prob, true_labels)
        self.assertAlmostEqual(1.0, result)


    def test_ksce__is_a_half(self):
        pred_prob = np.array([[0.0, 1], [0.0, 1]])
        true_labels = np.array([0, 1])

        result = ksce(pred_prob, true_labels)
        self.assertAlmostEqual(0.5, result)

        pred_prob = np.array([[0.0, 1]] * 10000)
        true_labels = np.array([0, 1] * 5000)

        result = ksce(pred_prob, true_labels)
        self.assertAlmostEqual(0.5, result)

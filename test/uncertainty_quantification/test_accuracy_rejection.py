import unittest

import numpy.testing as npt
from sklearn.model_selection import train_test_split

from src.experiments.util import train_svm
from src.uncertainty_quantification.accuracy_rejection import accuracy_rejection, plot_accuracy_rejection
from src.utilities.datasets import gummy_worm_dataset


class TestAccuracyRejection(unittest.TestCase):

    def test_accuracy_rejection(self):
        # Expected values were once empirically determined - this serves primarly as a regression test
        expected_accuracies = [0.934375, 0.96611111, 0.98078125, 0.98732143, 0.99145833,
                               0.99275, 0.9953125,  0.99583333, 0.996875, 0.99875]
        expected_rates = [0, 0.11111111, 0.22222222, 0.33333333, 0.44444444, 0.55555556,
                          0.66666667, 0.77777778, 0.88888889, 1]

        samples, labels = gummy_worm_dataset().generate_data(10000)
        X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=42)

        pred_prob = train_svm(X_train, y_train).predict_proba(X_test)
        rejection_accuracies, rejection_rates = accuracy_rejection(
            y_test, pred_prob, steps=10, strategy="entropy"
        )
        plot_accuracy_rejection(rejection_accuracies, rejection_rates)
        npt.assert_allclose(expected_accuracies, rejection_accuracies, rtol=1e-7, atol=1e-9)
        npt.assert_allclose(expected_rates, rejection_rates, rtol=1e-7, atol=1e-9)

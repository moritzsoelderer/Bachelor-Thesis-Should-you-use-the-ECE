import unittest

import numpy as np

import src.data_generation.datasets


class TestDataGeneration(unittest.TestCase):

    def test_cond_prob(self):
        test = src.data_generation.datasets.gummy_worm_dataset()
        delta = 0.25

        X, labels = test.generate_data(50000)

        combined = list(zip(X, labels))

        X_test = np.array([
            [5, 2.5],
            [8, 6],
            [10, 10.5],
        ])
        samples_near_test_samples = [[comb for comb in combined if np.linalg.norm(comb[0] - test_sample) <= delta] for test_sample in X_test]

        samples_near_test_samples_0 = [list(filter(lambda tup: tup[1] == 0, near_samples)) for near_samples in samples_near_test_samples]
        samples_near_test_samples_1 = [list(filter(lambda tup: tup[1] == 1, near_samples)) for near_samples in samples_near_test_samples]
        p_true_0 = np.array([test.cond_prob(s, 0) for s in X_test])
        p_true_1 = np.array([test.cond_prob(s, 1) for s in X_test])

        num_near_test_samples_0 = np.array(list(map(lambda l: len(l), samples_near_test_samples_0)))
        num_near_test_samples_1 = np.array(list(map(lambda l: len(l), samples_near_test_samples_1)))

        rel_freq0 = (1.0 * num_near_test_samples_0) / (num_near_test_samples_0 + num_near_test_samples_1)
        rel_freq1 = (1.0 * num_near_test_samples_1) / (num_near_test_samples_0 + num_near_test_samples_1)

        print(p_true_0, " : ", rel_freq0)
        print(p_true_1, " : ", rel_freq1)

        for result, expected in zip(p_true_0, rel_freq0):
            self.assertAlmostEqual(expected, result, delta=0.01)
        for result, expected in zip(p_true_1, rel_freq1):
            self.assertAlmostEqual(expected, result, delta=0.01)


    def test_data_generation_is_idempotent(self):
        datasets = [
            src.data_generation.datasets.gummy_worm_dataset,
            src.data_generation.datasets.imbalanced_gummy_worm_dataset,
            src.data_generation.datasets.sad_clown_dataset,
            src.data_generation.datasets.imbalanced_sad_clown_dataset
        ]
        for dataset in datasets:
            with self.subTest(dataset):
                data1 = dataset()
                samples1, labels1 = data1.generate_data(2500)

                data2 = dataset()
                samples2, labels2 = data2.generate_data(2500)

                np.testing.assert_array_equal(samples1, samples2)
                np.testing.assert_array_equal(labels1, labels2)

import numpy as np

from src.utilities.experiment_utils import train_svm, plot_bin_count_histogram
from src.metrics.true_ece import true_ece_binned
from src.utilities.data_generation import DataGeneration
from src.utilities.datasets import gummy_worm_dataset_hard


def calculate_true_ece_and_plot_bin_count(dataset: DataGeneration, bins: int, samplesize: int):
    samples, labels = dataset.generate_data(int(samplesize/dataset.n_features))
    prob1 = dataset.cond_prob(samples, k=1)
    svm = train_svm(samples, labels)
    pred_prob = svm.predict_proba(samples)
    true_prob = [[p, 1-p] for p in prob1]
    true_ece, bin_counts = true_ece_binned(pred_prob, true_prob, np.linspace(0, 1, bins))

    plot_bin_count_histogram(bin_counts, "Bin Count {True ECE: " + str(true_ece) + ", Samples: " + str(samplesize) + "}")
    dataset.scatter2d(show=True)


if __name__ == "__main__":
    dg = gummy_worm_dataset_hard()
    calculate_true_ece_and_plot_bin_count(dg, 100, 10000)
from src.metrics.true_ece import true_ece_binned
from src.utilities.data_generation import DataGeneration
from src.utilities.datasets import gummy_worm_dataset


def calculate_true_ece_and_plot_bin_count(dataset: DataGeneration, bins: int, samplesize: int):

    true_ece, bin_counts = true_ece_binned(dataset, samplesize)


if __name__ == "__main__":
    dg = gummy_worm_dataset()
    samples, labels = dg.generate_data(2)
    print("Samples: ", samples)
    true_prob = dg.cond_prob(samples)
    print(true_prob)
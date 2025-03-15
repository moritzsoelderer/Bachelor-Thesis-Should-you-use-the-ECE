import copy
import logging
from datetime import datetime

import numpy as np
from joblib import delayed, Parallel
from matplotlib import pyplot as plt

from src.experiments.util import predict_sklearn, predict_tf, EMPTY_METRIC_MEANS, EMPTY_METRIC_STD_DEVS
from src.experiments.varying_sample_size_dataset_family_util import generate_train_test_split, \
    calculate_true_ece_on_dists_and_grid, plot_bin_count_histogram, process_model, flatten_results, persist_to_pickle, \
    train_models
from src.utilities import utils, datasets

datetime_start = datetime.now()

logging.basicConfig(
    level=logging.INFO,
    format="MS-Varying Sample Size -- %(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"./logs/varying_sample_size_dataset_family/{datetime_start.strftime('%Y%m%d_%H%M%S')}.log"),  # Log to file
        logging.StreamHandler()
    ]
)


# Declare Metavariables #
dataset_size = 40000
min_samples = 100
max_samples = dataset_size/2
num_steps = 200
subsample_sizes = np.linspace(min_samples, max_samples, num_steps, dtype=np.int64)

# Adjust depending on dataset
data_generations = datasets.gummy_worm_dataset_family()
num_dists = data_generations[0].get_n_distributions()
sample_dim = data_generations[0].n_features

samples_per_distribution = int(dataset_size/num_dists)

true_ece_sample_size = 400000
true_ece_samples_grid = utils.sample_uniformly_within_bounds([-5, -5], [15, 15], true_ece_sample_size) # [-15, -15, -20], [15, 15, 10], 20000000 for sad clown dataset
samples_per_distribution_true_ece_dists = int(true_ece_sample_size/num_dists)


def main():
    # Generate Dataset #
    logging.info("Generating Dataset...")

    true_ece_samples_dists = [data_generation.generate_data(samples_per_distribution_true_ece_dists, overwrite=False)[0] for data_generation in data_generations]
    true_probabilities_grid = [[[1 - (p := data_generation.cond_prob(s, k=1)), p] for s in true_ece_samples_grid] for data_generation in data_generations]

    sampless_and_labelss = [
        generate_train_test_split(
            data_generation, samples_per_distribution, 0.5, 0.5, 42
        )
        for data_generation in data_generations
    ]
    train_samples = [samples_and_labels[0] for samples_and_labels in sampless_and_labelss]
    test_samples = [samples_and_labels[1] for samples_and_labels in sampless_and_labelss]
    train_labels = [samples_and_labels[2] for samples_and_labels in sampless_and_labelss]
    test_labels = [samples_and_labels[3] for samples_and_labels in sampless_and_labelss]

    logging.info("Training Models")
    svms, neural_networks, logistic_regressions, random_forests = train_models(train_samples, train_labels,  sample_dim)

    models = {
        "SVM": (svms, predict_sklearn),
        "Neural Network": (neural_networks, predict_tf),
        "Logistic Regression": (logistic_regressions, predict_sklearn),
        "Random Forest": (random_forests, predict_sklearn)
    }

    for model_name, model_pred_fun_tuple in models.items():
        logging.info("Model: %s", model_name)

        filename_absolute = f"{data_generations[0].title}__{model_name}__{data_generations[0].title} Family__AbsoluteValues__{datetime_start.strftime('%Y%m%d_%H%M%S')}"
        savePath = './data/varying_sample_size_dataset_family/' + filename_absolute + '.pkl'

        estimators = model_pred_fun_tuple[0]

        means = copy.deepcopy(EMPTY_METRIC_MEANS)
        std_devs = copy.deepcopy(EMPTY_METRIC_STD_DEVS)


        # Sanity Check
        assert len(data_generations) == len(true_ece_samples_dists) == len(estimators) == len(true_probabilities_grid)
        assert len(true_probabilities_grid) == len(test_samples) == len(test_labels)

        ### True ECE Calculation
        true_ece_values_15_bins = [
            calculate_true_ece_on_dists_and_grid(
                data_generations[index], true_ece_samples_dists[index],
                estimator, model_pred_fun_tuple[1], true_ece_samples_grid,
                true_probabilities_grid[index], 15) for index, estimator in enumerate(estimators)
        ]
        true_ece_values_100_bins = [
            calculate_true_ece_on_dists_and_grid(
                data_generations[index], true_ece_samples_dists[index],
                estimator, model_pred_fun_tuple[1], true_ece_samples_grid,
                true_probabilities_grid[index], 100) for index, estimator in enumerate(estimators)
        ]

        means["True ECE Dists (Binned - 15 Bins)"] = [np.mean([vals[2] for vals in true_ece_values_15_bins])] * num_steps
        means["True ECE Dists (Binned - 100 Bins)"] = [np.mean([vals[2] for vals in true_ece_values_100_bins])] * num_steps
        means["True ECE Grid (Binned - 15 Bins)"] = [np.mean([vals[0] for vals in true_ece_values_15_bins])] * num_steps
        means["True ECE Grid (Binned - 100 Bins)"] = [np.mean([vals[0] for vals in true_ece_values_100_bins])] * num_steps

        std_devs["True ECE Dists (Binned - 15 Bins)"] = [np.std([vals[2] for vals in true_ece_values_15_bins])] * num_steps
        std_devs["True ECE Dists (Binned - 100 Bins)"] = [np.std([vals[2] for vals in true_ece_values_100_bins])] * num_steps
        std_devs["True ECE Grid (Binned - 15 Bins)"] = [np.std([vals[0] for vals in true_ece_values_15_bins])] * num_steps
        std_devs["True ECE Grid (Binned - 100 Bins)"] = [np.std([vals[0] for vals in true_ece_values_100_bins])] * num_steps

        ### Plot bin counts
        plot_bin_count_histogram(true_ece_values_100_bins[0][1], "Bin Counts True ECE Grid (Binned - 100 Bins)")
        plot_bin_count_histogram(true_ece_values_15_bins[0][1], "Bin Counts True ECE Grid (Binned - 15 Bins)")
        plot_bin_count_histogram(true_ece_values_100_bins[0][3], "Bin Counts True ECE Dists (Binned - 100 Bins)")
        plot_bin_count_histogram(true_ece_values_15_bins[0][3], "Bin Counts True ECE Dists (Binned - 15 Bins)")


        ### Execution
        logging.info(
            "Executing Varying Test Sample Size on %s on %s family with %s training sample shape, %s max. test sample shape and %s datasets in total... (this might take some time)",
            model_name, data_generations[0].title, sampless_and_labelss[0][0].shape, sampless_and_labelss[0][1].shape, len(data_generations)
        )

        results = Parallel(n_jobs=-1, verbose=10)(  # n_jobs=-1 uses all available CPUs
            delayed(process_model)(estimators, test_samples, test_labels, subsample_size, model_pred_fun_tuple[1])
            for subsample_size in subsample_sizes
        )

        results = sorted(results, key=lambda x: x["Subsample Size"], reverse=False)

        logging.debug("Results Subsample Size: %s, %s : %s, %s", results[0], results[1], results[-2], results[-1])

        means, std_devs = flatten_results(results, means, std_devs)

        # Persist Values #
        persist_to_pickle(true_ece_samples_dists, true_ece_samples_grid, true_probabilities_grid, means, std_devs, savePath)


        # Plotting Absolute Mean and Std Deviation #
        logging.info("Plotting...")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        for metric in means.keys():
            metric_means = np.array(means[metric])
            print("Metric", metric)
            ax.plot(subsample_sizes, metric_means, label=metric)
            ax.fill_between(subsample_sizes, metric_means - np.array(std_devs[metric]), metric_means + np.array(std_devs[metric]), alpha=0.2)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
        plt.xlabel('Sample Size', fontsize=12)
        plt.ylabel('Metric Values', fontsize=12)
        plt.title(f'Varying Sample Size - {model_name}, {data_generations[0].title} Family', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.savefig("./plots/varying_sample_size_dataset_family/" + filename_absolute + ".png")
        plt.show()

        # Plotting Relative Mean and Std Deviation #
        logging.info("   Plotting...")
        filename_relative = f"{data_generations[0].title}__{model_name}__{data_generations[0].title} Family__RelativeValues__{datetime_start.strftime('%Y%m%d_%H%M%S')}"
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        for metric in means.keys():
            relative_means = np.array(means[metric]) - means["True ECE Grid (Binned - 100 Bins)"]
            ax.plot(subsample_sizes, relative_means, label=metric)
            ax.fill_between(subsample_sizes, relative_means - np.array(std_devs[metric]), relative_means + np.array(std_devs[metric]), alpha=0.2)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
        plt.xlabel('Sample Size', fontsize=12)
        plt.ylabel('Metric Values (Relative to True ECE Grid (Binned - 100 Bins))', fontsize=12)
        plt.title(f'Varying Sample Size - {model_name}, {data_generations[0].title} Family', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.savefig("./plots/varying_sample_size_dataset_family/" + filename_relative + ".png")
        plt.show()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    main()

import copy
import logging
from datetime import datetime

import numpy as np
from joblib import delayed, Parallel

from src.experiments.util import EMPTY_METRIC_MEANS, EMPTY_METRIC_STD_DEVS, DATASETS
from src.experiments.varying_test_sample_size_dataset_family.varying_sample_size_dataset_family_util import generate_train_test_split, \
    calculate_true_ece_on_dists_and_grid, plot_bin_count_histogram, process_model, flatten_results, persist_to_pickle, \
    train_models, plot_experiment
from src.utilities import utils
from src.utilities.data_generation import DataGeneration


def run(dataset_name, dataset_size, min_samples, max_samples, num_steps, true_ece_sample_size):
    ### Config
    datetime_start = datetime.now()
    logging.basicConfig(
        level=logging.INFO,
        format="MS-Varying Test Sample Size on Dataset Family -- %(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f"./logs/{datetime_start.strftime('%Y%m%d_%H%M%S')}.log"),
            # Log to file
            logging.StreamHandler()
        ]
    )

    logging.info(
        f"Received command line arguments: dataset_name={dataset_name}, dataset_size={dataset_size}, "
        f"min_samples={min_samples}, max_samples{max_samples}, num_steps={num_steps}, true_ece"
        f"_sample_size={true_ece_sample_size}"
    )

    subsample_sizes = np.linspace(min_samples, max_samples, num_steps, dtype=np.int64)
    dataset_info = DATASETS[dataset_name]
    data_generations : list[DataGeneration] = dataset_info[0]()
    num_dists = data_generations[0].get_n_distributions()

    # Generate Dataset #
    logging.info("Generating Dataset...")

    true_ece_samples_dists = [
        data_generation.generate_data(
            n_examples=int(true_ece_sample_size / num_dists), overwrite=False)
        [0] for data_generation in data_generations
    ]
    true_ece_samples_grid = utils.sample_uniformly_within_bounds(
        locs=dataset_info[1][0], scales=dataset_info[1][1], size=true_ece_sample_size
    )
    true_probabilities_grid = [
        [
            [1 - (p := data_generation.cond_prob(s, k=1)), p]
            for s in true_ece_samples_grid
        ]
        for data_generation in data_generations
    ]

    sampless_and_labelss = [
        generate_train_test_split(
            data_generation, int(dataset_size / num_dists), 0.5, 0.5, 42
        )
        for data_generation in data_generations
    ]
    train_samples = [samples_and_labels[0] for samples_and_labels in sampless_and_labelss]
    test_samples = [samples_and_labels[1] for samples_and_labels in sampless_and_labelss]
    train_labels = [samples_and_labels[2] for samples_and_labels in sampless_and_labelss]
    test_labels = [samples_and_labels[3] for samples_and_labels in sampless_and_labelss]

    models = train_models(train_samples, train_labels,  sample_dim=data_generations[0].n_features)

    for model_name, model_pred_fun_tuple in models.items():
        logging.info("Model: %s", model_name)

        filename_absolute = f"{data_generations[0].title}__{model_name}__{data_generations[0].title} Family__AbsoluteValues__{datetime_start.strftime('%Y%m%d_%H%M%S')}"
        filename_relative = f"{data_generations[0].title}__{model_name}__{data_generations[0].title} Family__RelativeValues__{datetime_start.strftime('%Y%m%d_%H%M%S')}"
        savePath = './data/' + filename_absolute + '.pkl'

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

        ### Data Persistence
        persist_to_pickle(true_ece_samples_dists, true_ece_samples_grid, true_probabilities_grid, means, std_devs, savePath)

        ### Plots
        plot_experiment(model_name, data_generations[0].title, means, std_devs, subsample_sizes, filename_absolute, filename_relative)

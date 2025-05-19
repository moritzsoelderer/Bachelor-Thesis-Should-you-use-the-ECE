import copy
import logging
from datetime import datetime

import numpy as np
from joblib import delayed, Parallel

from src.utilities.experiment_utils import EMPTY_METRIC_DICT, DATASETS, plot_bin_count_histogram
from src.experiments.varying_test_sample_size_dataset_family.varying_sample_size_dataset_family_util import \
    train_models, calculate_true_ece_on_dists_and_grid
from src.experiments.varying_test_sample_size_train_test_split_seeds.varying_sample_size_train_test_split_seeds_util import \
    generate_train_test_split, process_model, flatten_results, persist_to_pickle, plot_experiment
from src.utilities import utils
from src.data_generation.data_generation import DataGeneration


def run(dataset_name, dataset_size, min_sample_size, max_sample_size, num_steps, true_ece_sample_size):
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
        f"min_sample_size={min_sample_size}, max_sample_size{max_sample_size}, num_steps={num_steps}, true_ece"
        f"_sample_size={true_ece_sample_size}"
    )

    subsample_sizes = np.linspace(min_sample_size, max_sample_size, num_steps, dtype=np.int64)
    dataset_info = DATASETS[dataset_name]
    data_generation: DataGeneration = dataset_info[0]()
    num_dists = data_generation.get_n_distributions()
    train_test_split_seeds = [42, 8, 103, 34, 97, 3, 175, 163, 90, 11, 19, 82, 77, 131, 67, 49, 60, 22, 144, 89]

    # Generate Dataset #
    logging.info("Generating Dataset...")

    X_true_ece_dists, _ = data_generation.generate_data(
            n_examples=int(true_ece_sample_size / num_dists), overwrite=False)

    X_true_ece_grid = utils.sample_uniformly_within_bounds(
        locs=dataset_info[1][0], scales=dataset_info[1][1], size=true_ece_sample_size
    )

    p_true_grid = [
        [1 - (p := data_generation.cond_prob(s, k=1)), p]
        for s in X_true_ece_grid
    ]

    Xs_and_y_trues = [
        generate_train_test_split(
            data_generation, int(dataset_size / num_dists), 0.5, 0.5, random_state=seed
        )
        for seed in train_test_split_seeds
    ]
    X_train = [X_and_y_true[0] for X_and_y_true in Xs_and_y_trues]
    X_test = [X_and_y_true[1] for X_and_y_true in Xs_and_y_trues]
    y_true_train = [X_and_y_true[2] for X_and_y_true in Xs_and_y_trues]
    y_true_test = [X_and_y_true[3] for X_and_y_true in Xs_and_y_trues]

    models = train_models(X_train, y_true_train,  sample_dim=data_generation.n_features)

    for model_name, model_pred_fun_tuple in models.items():
        logging.info("Model: %s", model_name)

        filename_absolute = f"{data_generation.title}__{model_name}__20_TrainTestSplits__AbsoluteValues__{datetime_start.strftime('%Y%m%d_%H%M%S')}"
        filename_relative = f"{data_generation.title}__{model_name}__20_TrainTestSplits__RelativeValues__{datetime_start.strftime('%Y%m%d_%H%M%S')}"
        savePath = './data/' + filename_absolute + '.pkl'

        estimators = model_pred_fun_tuple[0]

        means = copy.deepcopy(EMPTY_METRIC_DICT)
        std_devs = copy.deepcopy(EMPTY_METRIC_DICT)


        # Sanity Check
        #assert len(X_true_ece_dists) == len(estimators) == len(p_true_grid)
        #assert len(p_true_grid) == len(X_test) == len(y_true_test)

        ### True ECE Calculation
        true_ece_values_15_bins = [
            calculate_true_ece_on_dists_and_grid(
                data_generation, X_true_ece_dists,
                estimator, model_pred_fun_tuple[1], X_true_ece_grid,
                p_true_grid, 15) for index, estimator in enumerate(estimators)
        ]
        true_ece_values_100_bins = [
            calculate_true_ece_on_dists_and_grid(
                data_generation, X_true_ece_dists,
                estimator, model_pred_fun_tuple[1], X_true_ece_grid,
                p_true_grid, 100) for index, estimator in enumerate(estimators)
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
            "Executing Varying Test Sample Size on %s on %s with %s training sample shape, %s max. test sample shape and %s train-test-splits in total... (this might take some time)",
            model_name, data_generation.title, Xs_and_y_trues[0][0].shape, Xs_and_y_trues[0][1].shape, len(estimators)
        )

        results = Parallel(n_jobs=-1, verbose=10)(  # n_jobs=-1 uses all available CPUs
            delayed(process_model)(estimators, X_test, y_true_test, subsample_size, model_pred_fun_tuple[1])
            for subsample_size in subsample_sizes
        )
        results = sorted(results, key=lambda x: x["Subsample Size"], reverse=False)

        logging.debug("Results Subsample Size: %s, %s : %s, %s", results[0], results[1], results[-2], results[-1])

        means, std_devs = flatten_results(results, means, std_devs)

        ### Data Persistence
        persist_to_pickle(estimators, X_true_ece_dists, X_true_ece_grid, p_true_grid, means, std_devs, subsample_sizes, savePath)

        ### Plots
        plot_experiment(model_name, data_generation.title, means, std_devs, subsample_sizes, filename_absolute, filename_relative)

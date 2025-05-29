import copy
import logging
from datetime import datetime

import numpy as np
from joblib import delayed, Parallel

from src.utilities.experiment_utils import EMPTY_METRIC_DICT, DATASETS, plot_bin_count_histogram
from src.experiments.varying_bins_dataset_family.varying_bins_dataset_family_util import \
    generate_train_test_split, \
    calculate_true_ece_on_dists_and_grid, process_model, flatten_results, persist_to_pickle, \
    train_models, plot_experiment
from src.utilities import utils
from src.data_generation.data_generation import DataGeneration


def run(dataset_name, dataset_size, min_bin_size, max_bin_size, num_steps, train_test_split_seed, true_ece_sample_size, model_names):
    ### Config
    datetime_start = datetime.now()
    logging.basicConfig(
        level=logging.INFO,
        format="MS-Varying Bins on Dataset Family -- %(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f"./logs/{datetime_start.strftime('%Y%m%d_%H%M%S')}.log"),
            # Log to file
            logging.StreamHandler()
        ]
    )

    logging.info(
        f"Received command line arguments: dataset_name={dataset_name}, dataset_size={dataset_size}, "
        f"min_bin_size={min_bin_size}, max_bin_size={max_bin_size}, num_steps={num_steps}, train_test_split_seed={train_test_split_seed}, "
        f"true_ece_sample_size={true_ece_sample_size}, model_names={model_names}"
    )

    binss = np.linspace(min_bin_size, max_bin_size, num_steps, dtype=np.uint32)
    dataset_info = DATASETS[dataset_name]
    data_generations: list[DataGeneration] = dataset_info[0]()
    num_dists = data_generations[0].get_n_distributions()

    # Generate Dataset #
    logging.info("Generating Dataset...")

    X_true_ece_dists = [
        data_generation.generate_data(
            n_examples=int(true_ece_sample_size / num_dists), overwrite=False)
        [0] for data_generation in data_generations
    ]
    X_true_ece_grid = utils.sample_uniformly_within_bounds(
        locs=dataset_info[1][0], scales=dataset_info[1][1], size=true_ece_sample_size
    )
    p_true_grid = [
        [
            [1 - (p := data_generation.cond_prob(s, k=1)), p]
            for s in X_true_ece_grid
        ]
        for data_generation in data_generations
    ]

    Xs_and_y_trues = [
        generate_train_test_split(
            data_generation, int(dataset_size / num_dists), 0.5, 0.5, train_test_split_seed
        )
        for data_generation in data_generations
    ]

    X_train = np.array([X_and_y_true[0] for X_and_y_true in Xs_and_y_trues])
    X_test = np.array([X_and_y_true[1] for X_and_y_true in Xs_and_y_trues])
    y_true_train = np.array([X_and_y_true[2] for X_and_y_true in Xs_and_y_trues])
    y_true_test = np.array([X_and_y_true[3] for X_and_y_true in Xs_and_y_trues])

    models = train_models(X_train, y_true_train,  sample_dim=data_generations[0].n_features, models=model_names)

    for model_name, model_pred_fun_tuple in models.items():
        logging.info("Model: %s", model_name)

        filename_absolute = f"{data_generations[0].title}__{model_name}__{data_generations[0].title} Family__AbsoluteValues__{datetime_start.strftime('%Y%m%d_%H%M%S')}"
        filename_relative = f"{data_generations[0].title}__{model_name}__{data_generations[0].title} Family__RelativeValues__{datetime_start.strftime('%Y%m%d_%H%M%S')}"
        savePath = './data/' + filename_absolute + '.pkl'

        estimators = model_pred_fun_tuple[0]

        means = copy.deepcopy(EMPTY_METRIC_DICT)
        std_devs = copy.deepcopy(EMPTY_METRIC_DICT)


        # Sanity Check
        assert len(data_generations) == len(X_true_ece_dists) == len(estimators) == len(p_true_grid)
        assert len(p_true_grid) == len(X_test) == len(y_true_test)

        pred_fun = model_pred_fun_tuple[1]
        predicted_probabilitiess = [pred_fun(estimator, X_test[i]) for i, estimator in enumerate(estimators)]

        ### Execution
        logging.info(
            "Executing Varying Bins on %s on %s family with %s training sample shape, min_bin_size %s, max_bin_size %s and %s datasets in total... (this might take some time)",
            model_name, data_generations[0].title, Xs_and_y_trues[0][0].shape, min_bin_size, max_bin_size, len(data_generations)
        )

        results = Parallel(n_jobs=-1, verbose=10)(  # n_jobs=-1 uses all available CPUs
            delayed(process_model)(predicted_probabilitiess, y_true_test, bins)
            for bins in binss
        )
        results = sorted(results, key=lambda x: x["bins"], reverse=False)

        logging.debug("Results Bins: %s, %s : %s, %s", results[0], results[1], results[-2], results[-1])

        means, std_devs = flatten_results(results, means, std_devs)

        def calc_true_ece(index, estimator, bins):
            return calculate_true_ece_on_dists_and_grid(
                data_generations[index], X_true_ece_dists[index],
                estimator, model_pred_fun_tuple[1], X_true_ece_grid,
                p_true_grid[index], bins)

        logging.info("Calculating True ECE with 15 bins")
        true_ece_values_15_bins = np.array(Parallel(n_jobs=-1)(
            delayed(calc_true_ece)(index, estimator, 15) for index, estimator in enumerate(estimators)
        ))
        logging.info("Calculating True ECE with 100 bins")
        true_ece_values_100_bins = np.array(Parallel(n_jobs=-1)(
            delayed(calc_true_ece)(index, estimator, 100) for index, estimator in enumerate(estimators)
        ))

        logging.info("Calculating True ECE Means and Std Dev")

        grid_15 = true_ece_values_15_bins[:, 0]
        dists_15 = true_ece_values_15_bins[:, 1]
        grid_100 = true_ece_values_100_bins[:, 0]
        dists_100 = true_ece_values_100_bins[:, 1]

        means.update({
            "True ECE Dists (Binned - 15 Bins)": [np.mean(dists_15)] * num_steps,
            "True ECE Dists (Binned - 100 Bins)": [np.mean(dists_100)] * num_steps,
            "True ECE Grid (Binned - 15 Bins)": [np.mean(grid_15)] * num_steps,
            "True ECE Grid (Binned - 100 Bins)": [np.mean(grid_100)] * num_steps
        })

        std_devs.update({
            "True ECE Dists (Binned - 15 Bins)": [np.std(dists_15)] * num_steps,
            "True ECE Dists (Binned - 100 Bins)": [np.std(dists_100)] * num_steps,
            "True ECE Grid (Binned - 15 Bins)": [np.std(grid_15)] * num_steps,
            "True ECE Grid (Binned - 100 Bins)": [np.std(grid_100)] * num_steps
        })

        logging.info("Persisting...")
        ### Data Persistence
        persist_to_pickle(estimators, X_true_ece_dists, X_true_ece_grid, p_true_grid, means, std_devs, binss, savePath)

        ### Plots
        plot_experiment(model_name, data_generations[0].title, means, std_devs, binss, filename_absolute, filename_relative)

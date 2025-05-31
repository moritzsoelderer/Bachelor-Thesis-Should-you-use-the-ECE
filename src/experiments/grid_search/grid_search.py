import copy
import logging
import pickle
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from _search_modified import GridSearchWithEstimatorOutput
from src.experiments.grid_search.grid_search_utils import neural_network_info, logistic_regression_info, \
    random_forest_info, svm_info, sort_by_key_index, remove_every_nth_element, process_model, plot_absolute_metrics, \
    plot_relative_metrics
from src.utilities.experiment_utils import DATASETS, EMPTY_METRIC_DICT
from src.utilities import utils


def run(dataset_name, dataset_size, num_folds, train_test_split_seed, test_size, true_ece_sample_size):
    ### Config
    datetime_start = datetime.now()
    logging.basicConfig(
        level=logging.INFO,
        format="MS-Grid Search -- %(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f"./logs/{datetime_start.strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

    logging.info(
        f"Received command line arguments: dataset_name={dataset_name}, dataset_size={dataset_size}, "
        f"num_folds={num_folds}, train_test_split_seed={train_test_split_seed}, test_size={test_size} and "
        f"true_sample_size={true_ece_sample_size}"
    )
    
    dataset_info = DATASETS[dataset_name]
    data_generation = dataset_info[0]()
    num_dists = data_generation.get_n_distributions()
    X, labels = data_generation.generate_data(n_examples=int(dataset_size/num_dists))

    model_infos = {
        "SVM": svm_info,
        "Neural Network": neural_network_info(data_generation.n_features),
        "Logistic Regression": logistic_regression_info,
        "Random Forest": random_forest_info
    }

    # Prepare data and obtain true probabilities
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=train_test_split_seed)
    p_true = np.array(
        [[data_generation.cond_prob(x, k=0), data_generation.cond_prob(x, k=1)] for x in X_test])

    X_true_ece_dists, _ = data_generation.generate_data(
        n_examples=int(true_ece_sample_size / num_dists), overwrite=False
    )
    X_true_ece_grid = utils.sample_uniformly_within_bounds(
        locs=dataset_info[1][0], scales=dataset_info[1][1], size=true_ece_sample_size
    )
    p_true_dists = np.array([[1 - (p := data_generation.cond_prob(s, k=1)), p] for s in X_true_ece_dists])
    p_true_grid = np.array([[1 - (p := data_generation.cond_prob(s, k=1)), p] for s in X_true_ece_grid])

    # Perform grid search for all model classes and parameter grids
    for model_name, model_info in model_infos.items():
        logging.info(f"Model: {model_name}")

        model, parameter_grid = model_info()

        logging.info(" Performing GridSearch")
        grid_search = GridSearchWithEstimatorOutput(  # Use modified GridSearch that outputs all estimators
            estimator=model, param_grid=parameter_grid, scoring='accuracy', cv=num_folds, n_jobs=-2, verbose=3
        )
        grid_search.fit(X_train, y_train)

        accuracies = grid_search.cv_results_["mean_test_score"]
        estimators = remove_every_nth_element(grid_search.cv_results_["estimator"], num_folds)
        logging.info(f"Estimators: {estimators}")

        metric_values = copy.deepcopy(EMPTY_METRIC_DICT)

        logging.info(f"Length Accuracies and Estimators: {len(estimators)}")

        # Calculate metrics on grid search estimators
        results = Parallel(n_jobs=-2, verbose=10)(
            delayed(process_model)(
                accuracy=accuracies[index],
                estimator=estimator,
                X_test=X_test.copy(),
                y_test=y_test.copy(),
                p_true=p_true.copy(),
                X_dists=X_true_ece_dists,
                X_grid=X_true_ece_grid,
                p_dists_true=p_true_dists,
                p_grid_true=p_true_grid
            )
            for index, estimator in enumerate(estimators)
        )

        # Persisting Results #
        logging.info("DEBUG: Persisting Results...")
        num_estimators = len(estimators)
        dataset_name = data_generation.title
        pickle_filename = (f"{dataset_name}__{model_name}__Samples_{dataset_size}__Estimators_{num_estimators}__Folds_"
                           f"{num_folds}__AbsoluteValues__{datetime_start.strftime('%Y%m%d_%H%M%S')}")
        with open('./data/' + pickle_filename + '.pkl', 'wb') as file:
            pickle.dump(results, file)

        logging.debug(f"Results: {results}")
        logging.debug(f"Length Results: {len(results)}")
        for result in results:
            for metric_name in metric_values.keys():
                metric_values[metric_name].append(result[metric_name])

        logging.info(f"Metric Values: {metric_values}")
        logging.info(f"Length Metric Values: {len(metric_values)}")
        logging.info(" Plotting...")

        # -1 equals Accuracy here, False ascending, 0 True ECE Grid (Binned 15 Bins) and True descending
        indices_and_sort_order = [(-1, False), (0, True)]
        for index, sort_order in indices_and_sort_order:
            metric_values_sorted = sort_by_key_index(metric_values, index, reverse=sort_order)
            sorted_by = list(metric_values.keys())[index]
            plot_absolute_metrics(
                dataset_name, model_name, sorted_by, num_estimators, metric_values_sorted,
                datetime_start, dataset_size, num_folds
            )
            plot_relative_metrics(
                dataset_name, model_name, sorted_by, num_estimators,
                metric_values_sorted, datetime_start, dataset_size, num_folds
            )
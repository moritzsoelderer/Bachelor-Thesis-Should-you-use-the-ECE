import logging
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.utilities.experiment_utils import train_neural_network, train_svm, train_logistic_regression, train_random_forest, \
    predict_tf, predict_sklearn
from src.metrics.ace import ace
from src.metrics.balance_score import balance_score
from src.metrics.ece import ece
from src.metrics.fce import fce
from src.metrics.ksce import ksce
from src.metrics.tce import tce
from src.metrics.true_ece import true_ece_binned


def process_model(predicted_probabilitiess, y_true_tests, bins):
    # Declare State Variables #
    metric_values = {
        "Accuracy": [],
        "ECE": [],
        "Balance Score": [],
        "FCE": [],
        "KSCE": [],
        "TCE": [],
        "ACE": []
    }

    means = {
        "Accuracy": [],
        "ECE": [],
        "Balance Score": [],
        "FCE": [],
        "KSCE": [],
        "TCE": [],
        "ACE": []
    }

    std_devs = {
        "Accuracy": [],
        "ECE": [],
        "Balance Score": [],
        "FCE": [],
        "KSCE": [],
        "TCE": [],
        "ACE": []
    }

    for index, predicted_probabilities in enumerate(predicted_probabilitiess):
        y_test = y_true_tests[index]

        # Predict Probabilities #
        y_pred = np.argmax(predicted_probabilities, axis=1)
        logging.debug("Predicted Probabilities Shape: %s", predicted_probabilities.shape)

        # Calculate Metric Values #
        logging.info("Evaluating Metrics...")
        metric_values["ECE"].append(ece(predicted_probabilities, y_test, bins))
        metric_values["FCE"].append(fce(predicted_probabilities, y_test, bins))
        metric_values["KSCE"].append(ksce(predicted_probabilities, y_test))
        metric_values["TCE"].append(tce(predicted_probabilities, y_test, n_bin=bins) / 100.0)
        metric_values["ACE"].append(ace(predicted_probabilities, y_test, bins))
        metric_values["Balance Score"].append(np.abs(balance_score(predicted_probabilities, y_test)))
        metric_values["Accuracy"].append(accuracy_score(y_test, y_pred))

    logging.info("Calculating Means and Std Deviations...")
    for metric, scores in metric_values.items():
        means[metric] = np.mean(scores)
        std_devs[metric] = np.std(scores)

    logging.info("DEBUG: Result: %s", {
        "bins": bins,
        "means": means,
        "std_devs": std_devs
    })
    return {
        "bins": bins,
        "means": means,
        "std_devs": std_devs
    }


def calculate_true_ece_on_dists_and_grid(data_generation, X_dist, estimator, predict_proba_fun, X_grid, p_true_grid,
                                         n_bins):
    p_true_dist_1 = data_generation.cond_prob(X_dist, k=1)
    p_true_dist = np.column_stack((1 - p_true_dist_1, p_true_dist_1))

    dist_predictions = predict_proba_fun(estimator, X_dist)
    grid_predictions = predict_proba_fun(estimator, X_grid)

    dist_true_ece, dist_bin_count = true_ece_binned(dist_predictions, p_true_dist, np.linspace(0, 1, n_bins + 1))
    grid_true_ece, grid_bin_count = true_ece_binned(grid_predictions, p_true_grid, np.linspace(0, 1, n_bins + 1))

    return grid_true_ece, grid_bin_count, dist_true_ece, dist_bin_count


def generate_train_test_split(data_generation, samples_per_distribution, test_size, train_size, random_state):
    X, y_true = data_generation.generate_data(samples_per_distribution)

    logging.debug("X Shape: %s", X.shape)
    logging.debug("y_true Shape: %s", y_true.shape)

    X_train, X_test, y_true_train, y_true_test = train_test_split(X, y_true, test_size=test_size,
                                                                  train_size=train_size, random_state=random_state)

    logging.debug("X_train Shape: %s", X_train.shape)
    logging.debug("X_test Shape: %s", X_test.shape)

    return X_train, X_test, y_true_train, y_true_test


def train_models(X_train, y_true_train, sample_dim):
    length = len(X_train)

    assert length == len(y_true_train)

    logging.info("Training Models")
    logging.info("Training SVM")
    svms = [train_svm(X_train[index], y_true_train[index]) for index in range(length)]

    logging.info("Training Neural Network")
    neural_networks = [train_neural_network(X_train[index], y_true_train[index], sample_dim) for index in range(length)]

    logging.info("Training Logistic Regression")
    logistic_regressions = [train_logistic_regression(X_train[index], y_true_train[index]) for index in range(length)]

    logging.info("Training Random Forest")
    random_forests = [train_random_forest(X_train[index], y_true_train[index]) for index in range(length)]

    return {
        "SVM": (svms, predict_sklearn),
        "Neural Network": (neural_networks, predict_tf),
        "Logistic Regression": (logistic_regressions, predict_sklearn),
        "Random Forest": (random_forests, predict_sklearn)
    }


def flatten_results(results, means, std_devs):
    for result in results:
        for metric, mean in result["means"].items():
            means[metric].append(mean)
        for metric, std in result["std_devs"].items():
            std_devs[metric].append(std)

    logging.debug("Means ECE: %s", means["ECE"])
    logging.debug("Means ACE: %s", means["ACE"])
    logging.debug("Std Devs ECE: %s", std_devs["ECE"])
    logging.debug("Std Devs ACE: %s", std_devs["ACE"])

    return means, std_devs


def persist_to_pickle(estimators, X_true_ece_dists, X_true_ece_grid, p_true_grid, means, std_devs, binss,
                      savePath):
    pickle_object = {
        "Estimators": estimators,
        "True ECE Samples Dists": X_true_ece_dists,
        "True ECE Samples Grid": X_true_ece_grid,
        "True Probabilities Grid": p_true_grid,
        "Means": means,
        "Std Devs": std_devs,
        "Binss": binss
    }
    with open(savePath, 'wb') as file:
        pickle.dump(pickle_object, file)


def plot_experiment(model_name, dataset_title, means, std_devs, binss, filename_absolute, filename_relative):
    # Plotting Absolute Mean and Std Deviation #
    logging.info("Plotting...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for metric in means.keys():
        metric_means = np.array(means[metric])
        print("Metric", metric)
        ax.plot(binss, metric_means, label=metric)
        ax.fill_between(binss, metric_means - np.array(std_devs[metric]),
                        metric_means + np.array(std_devs[metric]), alpha=0.2)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
    plt.xlabel('Sample Size', fontsize=12)
    plt.ylabel('Metric Values', fontsize=12)
    plt.title(f'Varying Bins - {model_name}, {dataset_title} Family', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.savefig("./plots/" + filename_absolute + ".png")
    plt.show(block=False)

    # Plotting Relative Mean and Std Deviation #
    logging.info("   Plotting...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for metric in means.keys():
        relative_means = np.array(means[metric]) - means["True ECE Grid (Binned - 100 Bins)"]
        ax.plot(binss, relative_means, label=metric)
        ax.fill_between(binss, relative_means - np.array(std_devs[metric]),
                        relative_means + np.array(std_devs[metric]), alpha=0.2)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
    plt.xlabel('Sample Size', fontsize=12)
    plt.ylabel('Metric Values (Relative to True ECE Grid (Binned - 100 Bins))', fontsize=12)
    plt.title(f'Varying Bins - {model_name}, {dataset_title} Family', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.savefig("./plots/" + filename_relative + ".png")
    plt.show(block=False)
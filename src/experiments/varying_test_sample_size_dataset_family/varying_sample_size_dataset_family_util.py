import logging
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.experiments.util import train_neural_network, train_svm, train_logistic_regression, train_random_forest, \
    predict_tf, predict_sklearn
from src.metrics.ace import ace
from src.metrics.balance_score import balance_score
from src.metrics.ece import ece
from src.metrics.fce import fce
from src.metrics.ksce import ksce
from src.metrics.tce import tce
from src.metrics.true_ece import true_ece_binned


def process_model(estimators, test_samples, test_labelss, subsample_size, predict_proba_fun):
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

    for index, estimator in enumerate(estimators):
        test_sample = test_samples[index]
        test_labels = test_labelss[index]

        # Filter by Subsample Indices #
        X_test = test_sample[:subsample_size]
        y_test = test_labels[:subsample_size]

        # Predict Probabilities #
        predicted_probabilities = predict_proba_fun(estimator, X_test)
        y_pred = np.argmax(predicted_probabilities, axis=1)
        logging.debug("Predicted Probabilities Shape: %s", predicted_probabilities.shape)

        # Calculate Metric Values #
        logging.info("Evaluating Metrics...")
        metric_values["ECE"].append(ece(predicted_probabilities, y_test, 15))
        metric_values["FCE"].append(fce(predicted_probabilities, y_test, 15))
        metric_values["KSCE"].append(ksce(predicted_probabilities, y_test))
        metric_values["TCE"].append(tce(predicted_probabilities, y_test, n_bin=15) / 100.0)
        metric_values["ACE"].append(ace(predicted_probabilities, y_test, 15))
        metric_values["Balance Score"].append(np.abs(balance_score(predicted_probabilities, y_test)))
        metric_values["Accuracy"].append(accuracy_score(y_test, y_pred))

    logging.info("Calculating Means and Std Deviations...")
    for metric, scores in metric_values.items():
        means[metric] = np.mean(scores)
        std_devs[metric] = np.std(scores)

    logging.info("DEBUG: Result: %s", {
        "Subsample Size": subsample_size,
        "means": means,
        "std_devs": std_devs
    })
    return {
        "Subsample Size": subsample_size,
        "means": means,
        "std_devs": std_devs
    }


def calculate_true_ece_on_dists_and_grid(data_generation, dist_samples, estimator, predict_proba_fun, grid_samples, grid_true_prob, n_bins):
    dist_true_prob = [[1 - (p := data_generation.cond_prob(s, k=1)), p] for s in dist_samples]

    dist_predictions = predict_proba_fun(estimator, dist_samples)
    grid_predictions = predict_proba_fun(estimator, grid_samples)

    dist_true_ece, dist_bin_count = true_ece_binned(dist_predictions, dist_true_prob, np.linspace(0, 1, n_bins + 1))
    grid_true_ece, grid_bin_count = true_ece_binned(grid_predictions, grid_true_prob, np.linspace(0, 1, n_bins + 1))

    return grid_true_ece, grid_bin_count, dist_true_ece, dist_bin_count


def generate_train_test_split(data_generation, samples_per_distribution, test_size, train_size, random_state):
    sample, labels = data_generation.generate_data(samples_per_distribution)

    logging.debug("Sample Shape: %s", sample.shape)
    logging.debug("Labels Shape: %s", labels.shape)

    train_sample, test_sample, train_labels, test_labels = train_test_split(sample, labels, test_size=test_size,
                                                                            train_size=train_size, random_state=random_state)

    logging.debug("Train Sample Shape: %s", train_sample.shape)
    logging.debug("Test Sample Shape: %s", test_sample.shape)

    return train_sample, test_sample, train_labels, test_labels


def train_models(train_samples, train_labels, sample_dim):
    length = len(train_samples)

    assert length == len(train_labels)

    logging.info("Training Models")
    logging.info("Training SVM")
    svms = [train_svm(train_samples[index], train_labels[index]) for index in range(length)]

    logging.info("Training Neural Network")
    neural_networks = [train_neural_network(train_samples[index], train_labels[index], sample_dim) for index in range(length)]

    logging.info("Training Logistic Regression")
    logistic_regressions = [train_logistic_regression(train_samples[index], train_labels[index]) for index in range(length)]

    logging.info("Training Random Forest")
    random_forests = [train_random_forest(train_samples[index], train_labels[index]) for index in range(length)]

    return {
        "SVM": (svms, predict_sklearn),
        "Neural Network": (neural_networks, predict_tf),
        "Logistic Regression": (logistic_regressions, predict_sklearn),
        "Random Forest": (random_forests, predict_sklearn)
    }


def flatten_results(results, means, std_devs):
    # Store Metric Values #
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


def persist_to_pickle(estimators, true_ece_samples_dists, true_ece_samples_grid, true_probabilities_grid, means, std_devs, subsample_sizes, savePath):
    pickle_object = {
        "Estimators": estimators,
        "True ECE Samples Dists": true_ece_samples_dists,
        "True ECE Samples Grid": true_ece_samples_grid,
        "True Probabilities Grid": true_probabilities_grid,
        "Means": means,
        "Std Devs": std_devs,
        "Subsample Sizes": subsample_sizes
    }
    with open(savePath, 'wb') as file:
        pickle.dump(pickle_object, file)
        
    
def plot_experiment(model_name, dataset_title, means, std_devs, subsample_sizes, filename_absolute, filename_relative):
    # Plotting Absolute Mean and Std Deviation #
    logging.info("Plotting...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for metric in means.keys():
        metric_means = np.array(means[metric])
        print("Metric", metric)
        ax.plot(subsample_sizes, metric_means, label=metric)
        ax.fill_between(subsample_sizes, metric_means - np.array(std_devs[metric]),
                        metric_means + np.array(std_devs[metric]), alpha=0.2)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
    plt.xlabel('Sample Size', fontsize=12)
    plt.ylabel('Metric Values', fontsize=12)
    plt.title(f'Varying Sample Size - {model_name}, {dataset_title} Family', fontsize=14, fontweight='bold')
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
        ax.plot(subsample_sizes, relative_means, label=metric)
        ax.fill_between(subsample_sizes, relative_means - np.array(std_devs[metric]),
                        relative_means + np.array(std_devs[metric]), alpha=0.2)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
    plt.xlabel('Sample Size', fontsize=12)
    plt.ylabel('Metric Values (Relative to True ECE Grid (Binned - 100 Bins))', fontsize=12)
    plt.title(f'Varying Sample Size - {model_name}, {dataset_title} Family', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.savefig("./plots/" + filename_relative + ".png")
    plt.show(block=False)
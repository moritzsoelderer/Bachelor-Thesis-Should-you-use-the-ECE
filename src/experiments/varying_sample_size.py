import copy
import logging
import pickle
from datetime import datetime

import numpy as np
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.experiments.util import train_neural_network, train_logistic_regression, train_random_forest, \
    predict_sklearn, predict_tf, EMPTY_METRIC_DICT, plot_probability_masks, train_svm
from src.metrics.ace import ace
from src.metrics.balance_score import balance_score
from src.metrics.ece import ece
from src.metrics.fce import fce
from src.metrics.ksce import ksce
from src.metrics.tce import tce
from src.metrics.true_ece import true_ece_binned
from src.utilities import utils, datasets

datetime_start = datetime.now()

logging.basicConfig(
    level=logging.INFO,
    format="MS-Varying Sample Size -- %(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"./logs/varying_sample_size/{datetime_start.strftime('%Y%m%d_%H%M%S')}.log"),  # Log to file
        logging.StreamHandler()
    ]
)


def process_model(test_sample, subsample_size, iteration_counter, labels, fun):
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

    for iteration in range(iteration_counter):
        # Prepare Subsample Indices #
        subsample_indices = np.random.choice(test_sample.shape[0], int(subsample_size), replace=True)

        # Filter by Subsample Indices #
        X_test = test_sample[subsample_indices]
        y_test = labels[subsample_indices]

        # Predict Probabilities #
        predicted_probabilities = fun[1](fun[0], X_test)
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


# Declare Metavariables #
dataset_size = 40000
iteration_counter = 20
min_samples = 100
max_samples = dataset_size/2
num_steps = 200
subsample_sizes = np.linspace(min_samples, max_samples, num_steps, dtype=np.int64)

# Adjust depending on dataset
dataset = datasets.gummy_worm_dataset
num_dists = 4
true_ece_samples_grid = utils.sample_uniformly_within_bounds([0, -3], [15, 15], 200000) # [-15, -15, -20], [15, 15, 10], 20000000 for sad clown dataset
samples_per_distribution = int(dataset_size/num_dists)  # / 6 for sad clown dataset
samples_per_distribution_true_ece_dists = int(200000/num_dists)
sample_dim = 2   # 3 for sad clown dataset

def main():
    # Generate Dataset #
    logging.info("Generating Dataset...")

    data_generation = dataset()
    true_ece_samples_dists, _ = data_generation.generate_data(samples_per_distribution_true_ece_dists, overwrite=False)
    true_probabilities_dists = np.array([[1 - (p := data_generation.cond_prob(s, k=1)), p] for s in true_ece_samples_dists])
    true_probabilities_grid = np.array([[1 - (p := data_generation.cond_prob(s, k=1)), p] for s in true_ece_samples_grid])

    sample, labels = data_generation.generate_data(samples_per_distribution)

    logging.debug("Sample Shape: %s", sample.shape)
    logging.debug("Labels Shape: %s", labels.shape)

    train_sample, test_sample, train_labels, test_labels = train_test_split(sample, labels, test_size=0.5, train_size=0.5, random_state=42)

    logging.debug("Train Sample Shape: %s", train_sample.shape)
    logging.debug("Test Sample Shape: %s", test_sample.shape)

    logging.info("Training Models")

    logging.info("Training SVM")
    svm = train_svm(train_sample, train_labels)

    logging.info("Training Neural Network")
    neural_network = train_neural_network(train_sample, train_labels, sample_dim)

    logging.info("Training Logistic Regression")
    logistic_regression = train_logistic_regression(train_sample, train_labels)

    logging.info("Training Random Forest")
    random_forest = train_random_forest(train_sample, train_labels)

    models = {
        "SVM": (svm, predict_sklearn),
        "Neural Network": (neural_network, predict_tf),
        "Logistic Regression": (logistic_regression, predict_sklearn),
        "Random Forest": (random_forest, predict_sklearn)
    }

    # Plot Dataset #
    scatter_plot = data_generation.scatter2d(0, 1, "Feature 1", "Feature 2", np.array(['#111111', '#FF5733']))
    scatter_plot.show()

    for model_name, model_pred_fun_tuple in models.items():
        logging.info("Model: %s", model_name)

        means = copy.deepcopy(EMPTY_METRIC_DICT)
        std_devs = copy.deepcopy(EMPTY_METRIC_DICT)

        # Calculate True ECE of model (approximation)
        predictions_dists = model_pred_fun_tuple[1](model_pred_fun_tuple[0], true_ece_samples_dists)
        predictions_grid = model_pred_fun_tuple[1](model_pred_fun_tuple[0], true_ece_samples_grid)

        # Plot probability masks
        save_path = './plots/varying_sample_size/'
        filename_probabilities_grid = f"{data_generation.title}__{model_name}__Iterations_{iteration_counter}__Grid"
        filename_probabilities_dists = f"{data_generation.title}__{model_name}__Iterations_{iteration_counter}__Dists"
        plot_probability_masks(true_ece_samples_grid, true_probabilities_grid, predictions_grid, filename_probabilities_grid, datetime_start, save_path=save_path)
        plot_probability_masks(true_ece_samples_dists, true_probabilities_dists, predictions_dists, filename_probabilities_dists, datetime_start, save_path=save_path)

        # True ECE does not deviate
        means["True ECE Dists (Binned)"] = [true_ece_binned(predictions_dists, true_probabilities_dists, np.linspace(0, 1, 100))] * num_steps
        means["True ECE Dists (Binned - 15 Bins)"] = [true_ece_binned(predictions_dists, true_probabilities_dists, np.linspace(0, 1, 15))] * num_steps
        means["True ECE Grid (Binned)"] = [true_ece_binned(predictions_grid, true_probabilities_grid, np.linspace(0, 1, 100))] * num_steps

        # True ECE does not deviate
        std_devs["True ECE Dists (Binned)"] = [0] * num_steps
        std_devs["True ECE Dists (Binned - 15 Bins)"] = [0] * num_steps
        std_devs["True ECE Grid (Binned)"] = [0] * num_steps

        logging.info(
            "Executing Varying Test Sample Size on %s on %s with %s training sample shape, %s max. test sample shape and %s iterations per test subsample size... (this might take some time)",
            model_name, data_generation.title, train_sample.shape, test_sample.shape, iteration_counter
        )

        results = Parallel(n_jobs=-1, verbose=10)(  # n_jobs=-1 uses all available CPUs
            delayed(process_model)(test_sample.copy(), subsample_size, iteration_counter, test_labels.copy(), model_pred_fun_tuple)
            for subsample_size in subsample_sizes
        )

        results = sorted(results, key=lambda x: x["Subsample Size"], reverse=False)

        logging.debug("Results Subsample Size: %s, %s : %s, %s", results[0], results[1], results[-2], results[-1])

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

        # Persist Values #
        pickle_object = {
            "True ECE Samples Dists": true_ece_samples_dists,
            "True Probabilities Dists": true_probabilities_dists,
            "True ECE Dists Predicted Probabilities": predictions_dists,
            "True ECE Samples Grid": true_ece_samples_grid,
            "True Probabilities Grid": true_probabilities_grid,
            "True ECE Grid Predicted Probabilities": predictions_grid,
            "Means": means,
            "Std Devs": std_devs
        }
        filename_absolute = f"{data_generation.title}__{model_name}__Iterations_{iteration_counter}__AbsoluteValues__{datetime_start.strftime('%Y%m%d_%H%M%S')}"
        with open('./data/varying_sample_size/' + filename_absolute + '.pkl', 'wb') as file:
            pickle.dump(pickle_object, file)

        # Plotting Absolute Mean and Std Deviation #
        logging.info("Plotting...")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        for metric in means.keys():
            metric_means = np.array(means[metric])
            ax.plot(subsample_sizes, metric_means, label=metric)
            ax.fill_between(subsample_sizes, metric_means - np.array(std_devs[metric]), metric_means + np.array(std_devs[metric]), alpha=0.2)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
        plt.xlabel('Sample Size', fontsize=12)
        plt.ylabel('Metric Values', fontsize=12)
        plt.title(f'Varying Sample Size - {model_name}, Iterations: {iteration_counter}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.savefig("./plots/varying_sample_size/" + filename_absolute + ".png")
        plt.show(block=False)

        # Plotting Relative Mean and Std Deviation #
        logging.info("   Plotting...")
        filename_relative = f"{data_generation.title}__{model_name}__Iterations_{iteration_counter}__RelativeValues__{datetime_start.strftime('%Y%m%d_%H%M%S')}"
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        for metric in means.keys():
            relative_means = np.array(means[metric]) - means["True ECE Grid (Binned)"]
            ax.plot(subsample_sizes, relative_means, label=metric)
            ax.fill_between(subsample_sizes, relative_means - np.array(std_devs[metric]), relative_means + np.array(std_devs[metric]), alpha=0.2)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
        plt.xlabel('Sample Size', fontsize=12)
        plt.ylabel('Metric Values (Relative to True ECE Grid (Binned))', fontsize=12)
        plt.title(f'Varying Sample Size - {model_name}, Iterations: {iteration_counter}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.savefig("./plots/varying_sample_size/" + filename_relative + ".png")
        plt.show(block=False)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    main()

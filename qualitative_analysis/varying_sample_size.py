from datetime import datetime

import numpy as np
import scipy.stats as st
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import data_generation as dg
from metrics.ace import ace
from metrics.balance_score import balance_score
from metrics.ece import ece
from metrics.fce import fce
from metrics.ksce import ksce
from metrics.tce import tce
from metrics.true_ece import true_ece
from qualitative_analysis import util


def predict_svm(X, y):
    svm_model = SVC(kernel='linear', probability=True)  # Enable probability estimation
    svm_model.fit(X, y)
    return svm_model.predict_proba(X)


def predict_neural_network(X, y):
    y_categorical = tf.keras.utils.to_categorical(y)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(50, activation="tanh"))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X.reshape(-1, 2), y_categorical, epochs=15, batch_size=1000)
    return model.predict(X.reshape(-1, 2))


def predict_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model.predict_proba(X)


def predict_random_forest(X, y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model.predict_proba(X)


# Declare Metavariables #
iteration_counter = 20
subsample_sizes = [100, 1000, 5000, 10000, 20000, 50000, 80000]
models = {
    "SVM": predict_svm,
    "Neural Network": predict_neural_network,
    "Logistic Regression": predict_logistic_regression,
    "Random Forest": predict_random_forest
}

# Generate Dataset #
data_generation = util.gummy_worm_dataset()
sample, labels = data_generation.generate_data(25000)

# Plot Dataset #
scatter_plot = data_generation.scatter2d(0, 1, "feature 1", "feature 2", np.array(['#111111', '#FF5733']))
scatter_plot.show()

print("----------------------------------------")
for model_name, predict in models.items():
    print("Model: ", model_name)
    for subsample_size in subsample_sizes:
        print("  Subsample Size: ", subsample_size)
        # Declare State Variables #
        metric_values = {
            "true_ece": [],
            "ece": [],
            "balance_score": [],
            "fce": [],
            "ksce": [],
            "tce": [],
            "ace": []
        }

        # Train Model
        for iteration in range(iteration_counter):
            print("   Iteration: ", iteration + 1, "/", iteration_counter)

            # Prepare Subsample and True Probabilities #
            subsample_indices = np.random.choice(sample.shape[0], subsample_size, replace=False)

            X = sample[subsample_indices]
            y = labels[subsample_indices]

            true_probabilities = np.array(
                [[data_generation.cond_prob(x, k=0), data_generation.cond_prob(x, k=1)] for x in X])

            # Train Model
            print("      Training Model...")
            predicted_probabilities = predict(X, y)
            print(predicted_probabilities)

            # Calculate Metric Values #
            print("      Evaluating Metrics...")
            true_ece_score = true_ece(predicted_probabilities, true_probabilities)
            ece_score = ece(predicted_probabilities, y, 15)
            balance_score_score = np.abs(balance_score(predicted_probabilities, y))
            fce_score = fce(predicted_probabilities, y, 15)
            ksce_score = ksce(predicted_probabilities, y)
            tce_score = tce(predicted_probabilities, y, n_bin=15) / 100.0
            ace_score = ace(predicted_probabilities, y, 15)

            # Store Metric Values #
            metric_values["ece"].append(ece_score)
            metric_values["fce"].append(fce_score)
            metric_values["ksce"].append(ksce_score)
            metric_values["tce"].append(tce_score)
            metric_values["ace"].append(ace_score)
            metric_values["true_ece"].append(true_ece_score)
            metric_values["balance_score"].append(balance_score_score)

        # Calculating Means and Std Deviations #
        print("   Calculating Means and Std Deviations...")
        means = {}
        std_devs = {}

        for metric, scores in metric_values.items():
            relative_scores = np.array(scores) - metric_values["true_ece"]
            means[metric] = np.mean(relative_scores)
            std_devs[metric] = np.std(relative_scores)

        function_names = list(means.keys())
        mean_values = list(means.values())
        std_devs_values = list(std_devs.values())

        # Plotting Mean and Std Deviation #
        print("   Plotting Errorbar Plot...")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.errorbar(function_names, mean_values, yerr=std_devs_values, capsize=5, fmt='o', color='red',
                     ecolor='black', linestyle='None', markersize=8, label="Mean ± STD")
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Metric Values (Relative to True ECE)', fontsize=12)
        plt.title(f'Metric Behaviour - {model_name} - Samples: {subsample_size}, Iterations: {iteration_counter}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.tight_layout()
        ax.grid(True, linestyle='--', alpha=0.6)

        filename = f"{model_name}__Samples_{subsample_size}__Iterations_{iteration_counter}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig("./plots/varying_sample_size/" + filename)
        plt.show()

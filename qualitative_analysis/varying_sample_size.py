from datetime import datetime

import numpy as np
import tensorflow as tf
import pickle
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from metrics.ace import ace
from metrics.balance_score import balance_score
from metrics.ece import ece
from metrics.fce import fce
from metrics.ksce import ksce
from metrics.tce import tce
from metrics.true_ece import true_ece
from qualitative_analysis import util

# predict distinction for tensorflow and sklearn
predict_sklearn = lambda model, X_test: model.predict_proba(X_test)
predict_tf = lambda model, X_test: model.predict(X_test)

def train_svm(X_train, y_train):
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)
    return svm_model

def train_neural_network(X_train, y_train):
    y_categorical = tf.keras.utils.to_categorical(y_train)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(50, activation="tanh"))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train.reshape(-1, 2), y_categorical, epochs=15, batch_size=1000)
    return model


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model


def process_model(test_sample, subsample_size, iteration_counter, true_probabilities, labels, fun):
    # Declare State Variables #
    metric_values = {
        "True ECE": [],
        "ECE": [],
        "Balance Score": [],
        "FCE": [],
        "KSCE": [],
        "TCE": [],
        "ACE": []
    }

    means = {
        "True ECE": [],
        "ECE": [],
        "Balance Score": [],
        "FCE": [],
        "KSCE": [],
        "TCE": [],
        "ACE": []
    }

    std_devs = {
        "True ECE": [],
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
        true_probabilities_test = true_probabilities[subsample_indices]

        # Predict Probabilities #
        predicted_probabilities = fun[1](fun[0], X_test)
        print("      DEBUG: Predicted Probabilities Shape: ", predicted_probabilities.shape)

        # Calculate Metric Values #
        print("      Evaluating Metrics...")
        metric_values["ECE"].append(ece(predicted_probabilities, y_test, 15))
        metric_values["FCE"].append(fce(predicted_probabilities, y_test, 15))
        metric_values["KSCE"].append(ksce(predicted_probabilities, y_test))
        metric_values["TCE"].append(tce(predicted_probabilities, y_test, n_bin=15) / 100.0)
        metric_values["ACE"].append(ace(predicted_probabilities, y_test, 15))
        metric_values["True ECE"].append(true_ece(predicted_probabilities, true_probabilities_test))
        metric_values["Balance Score"].append(np.abs(balance_score(predicted_probabilities, y_test)))

    print("   Calculating Means and Std Deviations...")
    for metric, scores in metric_values.items():
        means[metric] = np.mean(scores)
        std_devs[metric] = np.std(scores)

    print("   DEBUG: Result: ", {
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
dataset_size = 1000
iteration_counter = 20
min_samples = 100
max_samples = dataset_size/2
step_size = 100
subsample_sizes = np.arange(min_samples, max_samples + step_size, step_size)

def main():
    # Generate Dataset #
    print("Generating Dataset")
    data_generation = util.gummy_worm_dataset()
    sample, labels = data_generation.generate_data(int(dataset_size/4))
    print("DEBUG: Sample Shape: ", sample.shape)
    print("DEBUG: Labels Shape: ", labels.shape)

    # randomly split dataset into two halfs (train and test)
    train_sample, test_sample, train_labels, test_labels = train_test_split(sample, labels, test_size=0.5, train_size=0.5, random_state=42)
    print("DEBUG: Train Sample Shape: ", train_sample.shape)
    print("DEBUG: Test Sample Shape: ", test_sample.shape)
    # Calculate true probabilties for test set (training set is not needed) #
    print("Calculating True Probabilities")
    true_probabilities = np.array(
        [[data_generation.cond_prob(x, k=0), data_generation.cond_prob(x, k=1)] for x in test_sample])
    print("DEBUG: True Probabilities Shape: ", true_probabilities.shape)

    # Instantiate and train models #
    print("Training Models")
    print(" Training SVM")
    svm = train_svm(train_sample, train_labels)
    print(" Training Neural Network")
    neural_network = train_neural_network(train_sample, train_labels)
    print(" Training Logistic Regression")
    logistic_regression = train_logistic_regression(train_sample, train_labels)
    print(" Training Random Forest")
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

    print("----------------------------------------")
    for model_name, fun in models.items():
        print("Model: ", model_name)

        means = {
            "True ECE": [],
            "ECE": [],
            "Balance Score": [],
            "FCE": [],
            "KSCE": [],
            "TCE": [],
            "ACE": []
        }

        std_devs = {
            "True ECE": [],
            "ECE": [],
            "Balance Score": [],
            "FCE": [],
            "KSCE": [],
            "TCE": [],
            "ACE": []
        }

        # Train Model
        results = Parallel(n_jobs=-1, verbose=10)(  # n_jobs=-1 uses all available CPUs
            delayed(process_model)(test_sample.copy(), subsample_size, iteration_counter, true_probabilities, test_labels.copy(), fun)
            for subsample_size in subsample_sizes
        )

        results = sorted(results, key=lambda x: x["Subsample Size"], reverse=False)

        print("DEBUG: Results Subsample Size: ", results[0], results[1], " : ", results[-2], results[-1])

        # Persist Values #
        filename = f"{model_name}__Iterations_{iteration_counter}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open('./data/varying_sample_size/' + filename + '.pkl', 'wb') as file:
            pickle.dump(results, file)

        # Store Metric Values #
        for result in results:
            for metric, mean in result["means"].items():
                means[metric].append(mean)
            for metric, std in result["std_devs"].items():
                std_devs[metric].append(std)

        # Plotting Mean and Std Deviation #
        print("   Plotting...")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        for metric in means.keys():
            relative_means = np.array(means[metric]) - means["True ECE"]
            ax.plot(subsample_sizes, relative_means, label=metric)
            ax.fill_between(subsample_sizes, relative_means - np.array(std_devs[metric]), relative_means + np.array(std_devs[metric]), alpha=0.2)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
        plt.xlabel('Sample Size', fontsize=12)
        plt.ylabel('Metric Values (Relative to True ECE)', fontsize=12)
        plt.title(f'Varying Sample Size - {model_name}, Iterations: {iteration_counter}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.savefig("./plots/varying_sample_size/" + filename + ".png")
        plt.show()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    main()

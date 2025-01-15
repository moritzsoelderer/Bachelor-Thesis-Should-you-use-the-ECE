import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.metrics.ace import ace
from src.metrics.balance_score import balance_score
from src.metrics.ece import ece
from src.metrics.fce import fce
from src.metrics.ksce import ksce
from src.metrics.tce import tce
from src.metrics.true_ece import true_ece, true_ece_binned
from src.utilities import utils, datasets

# predict distinction for tensorflow and sklearn
predict_sklearn = lambda model, X_test: model.predict_proba(X_test)
predict_tf = lambda model, X_test: model.predict(X_test)

def train_svm(X_train, y_train):
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)
    return svm_model

def train_neural_network(X_train, y_train, sample_dim):
    y_categorical = tf.keras.utils.to_categorical(y_train)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(50, activation="tanh"))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train.reshape(-1, sample_dim), y_categorical, epochs=15, batch_size=1000)
    return model


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model


def process_model(test_sample, subsample_size, iteration_counter, labels, fun):
    # Declare State Variables #
    metric_values = {
        "ECE": [],
        "Balance Score": [],
        "FCE": [],
        "KSCE": [],
        "TCE": [],
        "ACE": []
    }

    means = {
        "ECE": [],
        "Balance Score": [],
        "FCE": [],
        "KSCE": [],
        "TCE": [],
        "ACE": []
    }

    std_devs = {
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
        print("      DEBUG: Predicted Probabilities Shape: ", predicted_probabilities.shape)

        # Calculate Metric Values #
        print("      Evaluating Metrics...")
        metric_values["ECE"].append(ece(predicted_probabilities, y_test, 15))
        metric_values["FCE"].append(fce(predicted_probabilities, y_test, 15))
        metric_values["KSCE"].append(ksce(predicted_probabilities, y_test))
        metric_values["TCE"].append(tce(predicted_probabilities, y_test, n_bin=15) / 100.0)
        metric_values["ACE"].append(ace(predicted_probabilities, y_test, 15))
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
dataset_size = 40000
iteration_counter = 20
min_samples = 100
max_samples = dataset_size/2
num_steps = 200
subsample_sizes = np.linspace(min_samples, max_samples, num_steps, dtype=np.int64)

# Adjust depending on dataset
dataset = datasets.gummy_worm_dataset
true_ece_samples = utils.sample_uniformly_within_bounds([0, -3], [15, 15], 200000) # other locs and scales for sad clown dataset
samples_per_distribution = int(dataset_size/4)  # / 6 for sad clown dataset
sample_dim = 2   # 3 for sad clown dataset

def main():
    datetime_start = datetime.now()
    # Generate Dataset #
    print("Generating Dataset")
    data_generation = dataset()
    sample, labels = data_generation.generate_data(samples_per_distribution)
    print("DEBUG: Sample Shape: ", sample.shape)
    print("DEBUG: Labels Shape: ", labels.shape)

    # randomly split dataset into two halfs (train and test)
    train_sample, test_sample, train_labels, test_labels = train_test_split(sample, labels, test_size=0.5, train_size=0.5, random_state=42)
    print("DEBUG: Train Sample Shape: ", train_sample.shape)
    print("DEBUG: Test Sample Shape: ", test_sample.shape)
    true_probabilities = np.array([[1 - (p := data_generation.cond_prob(s, k=1)), p] for s in true_ece_samples])

    # Instantiate and train models #
    print("Training Models")
    print(" Training SVM")
    svm = train_svm(train_sample, train_labels)
    print(" Training Neural Network")
    neural_network = train_neural_network(train_sample, train_labels, sample_dim)
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
            "True ECE (Binned)": [],
            "ECE": [],
            "Balance Score": [],
            "FCE": [],
            "KSCE": [],
            "TCE": [],
            "ACE": [],
            "True ECE": []
        }

        std_devs = {
            "True ECE (Binned)": [],
            "ECE": [],
            "Balance Score": [],
            "FCE": [],
            "KSCE": [],
            "TCE": [],
            "ACE": [],
            "True ECE": []
        }

        # Calculate True ECE of model (approximation)
        predictions = fun[1](fun[0], true_ece_samples)

        # Plot probability masks
        filename_probabilities = f"{data_generation.title}__{model_name}__Iterations_{iteration_counter}"
        utils.plot_samples_probability_mask(true_ece_samples, predictions,
                                            colorbar_label='Predicted Probability (Positive Class)',
                                            title='Predicted Probabilities (Positive Class)',
                                            save_path='./plots/varying_sample_size/' + filename_probabilities +
                                                      '__Predicted_Probabilities__' + datetime_start.strftime('%Y%m%d_%H%M%S') + '.png')
        utils.plot_samples_probability_mask(true_ece_samples, true_probabilities,
                                            colorbar_label='True Probability (Positive Class)',
                                            title='True Probabilities (Positive Class)',
                                            save_path='./plots/varying_sample_size/' + filename_probabilities +
                                                      '__True_Probabilities__' + datetime_start.strftime(
                                                '%Y%m%d_%H%M%S') + '.png')
        utils.plot_samples_probability_mask(true_ece_samples, np.abs(predictions - true_probabilities),
                                            colorbar_label='Probability Difference (Positive Class)',
                                            title='Difference Predicted and True Probabilities (Positive Class)',
                                            save_path='./plots/varying_sample_size/' + filename_probabilities +
                                                      '__Probabilitiy_Difference__' + datetime_start.strftime(
                                                '%Y%m%d_%H%M%S') + '.png')

        # True ECE does not deviate
        means["True ECE"] = [true_ece(predictions, true_probabilities)] * num_steps
        means["True ECE (Binned)"] = [true_ece_binned(predictions, true_probabilities, np.linspace(0, 1, 100))] * num_steps

        # True ECE does not deviate
        std_devs["True ECE"] = [0] * num_steps
        std_devs["True ECE (Binned)"] = [0] * num_steps

        # Calculate other Metrics
        results = Parallel(n_jobs=-1, verbose=10)(  # n_jobs=-1 uses all available CPUs
            delayed(process_model)(test_sample.copy(), subsample_size, iteration_counter, test_labels.copy(), fun)
            for subsample_size in subsample_sizes
        )

        results = sorted(results, key=lambda x: x["Subsample Size"], reverse=False)

        print("DEBUG: Results Subsample Size: ", results[0], results[1], " : ", results[-2], results[-1])

        # Store Metric Values #
        for result in results:
            for metric, mean in result["means"].items():
                means[metric].append(mean)
            for metric, std in result["std_devs"].items():
                std_devs[metric].append(std)

        print("DEBUG: Means ECE: ", means["ECE"])
        print("DEBUG: Means ACE: ", means["ACE"])
        print("DEBUG: Std Devs ECE: ", std_devs["ECE"])
        print("DEBUG: Std Devs ACE: ", std_devs["ACE"])

        # Persist Values #
        pickle_object = {
            "True Probabilities": true_probabilities,
            "True ECE Predicted Probabilities": predictions,
            "Means": means,
            "Std Devs": std_devs
        }
        filename_absolute = f"{data_generation.title}__{model_name}__Iterations_{iteration_counter}__AbsoluteValues__{datetime_start.strftime('%Y%m%d_%H%M%S')}"
        with open('./data/varying_sample_size/' + filename_absolute + '.pkl', 'wb') as file:
            pickle.dump(pickle_object, file)

        # Plotting Absolute Mean and Std Deviation #
        print("   Plotting...")
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
        plt.show()

        # Plotting Relative Mean and Std Deviation #
        print("   Plotting...")
        filename_relative = f"{data_generation.title}__{model_name}__Iterations_{iteration_counter}__RelativeValues__{datetime_start.strftime('%Y%m%d_%H%M%S')}"
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        for metric in means.keys():
            relative_means = np.array(means[metric]) - means["True ECE (Binned)"]
            ax.plot(subsample_sizes, relative_means, label=metric)
            ax.fill_between(subsample_sizes, relative_means - np.array(std_devs[metric]), relative_means + np.array(std_devs[metric]), alpha=0.2)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
        plt.xlabel('Sample Size', fontsize=12)
        plt.ylabel('Metric Values (Relative to True ECE)', fontsize=12)
        plt.title(f'Varying Sample Size - {model_name}, Iterations: {iteration_counter}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.savefig("./plots/varying_sample_size/" + filename_relative + ".png")
        plt.show()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    main()

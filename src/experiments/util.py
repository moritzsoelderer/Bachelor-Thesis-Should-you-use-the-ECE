from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf

from src.utilities import utils
from src.data_generation import datasets

EMPTY_METRIC_DICT = {
    "True ECE Grid (Binned - 100 Bins)": [],
    "True ECE Grid (Binned - 15 Bins)": [],
    "ECE": [],
    "Balance Score": [],
    "FCE": [],
    "KSCE": [],
    "TCE": [],
    "ACE": [],
    "True ECE Dists (Binned - 100 Bins)": [],
    "True ECE Dists (Binned - 15 Bins)": [],
    "Accuracy": []
}

DATASETS = {
    "gummy_worm": (datasets.gummy_worm_dataset, ([-5, -5], [15, 15])),
    "gummy_worm_imbalanced": (datasets.imbalanced_gummy_worm_dataset, ([-5, -5], [15, 15])),
    "gummy_worm_family": (datasets.gummy_worm_dataset_family, ([-5, -5], [15, 15])),
    "sad_clown": (datasets.sad_clown_dataset, ([], [])),
    "sad_clown_imbalanced": (datasets.imbalanced_sad_clown_dataset, ([], [])),  # bounds to be added for sad clown
    "exclamation_mark": (datasets.exclamation_mark_dataset, ([-5, -5], [15, 15])),
    "exclamation_mark_family": (datasets.exclamation_mark_dataset_family, ([-5, -5], [15, 15]))
}

# predict distinction for tensorflow and sklearn
predict_sklearn = lambda model, X_test: model.predict_proba(X_test)
predict_tf = lambda model, X_test: model.predict(X_test, verbose=0)


def train_svm(X_train, y_train):
    svm_model = SVC(probability=True)
    svm_model.fit(X_train, y_train)
    return svm_model


def train_neural_network(X_train, y_train, sample_dim):
    y_categorical = tf.keras.utils.to_categorical(y_train)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(50, activation="tanh"))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train.reshape(-1, sample_dim), y_categorical, epochs=15, batch_size=1000, verbose=0)
    return model


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model


def plot_probability_masks(X, p_true, predictions, filename=None, date_time=datetime.now(), show=True, save_path=None):
    formatted_date_time = date_time.strftime('%Y%m%d_%H%M%S')
    if save_path is not None:
        p_pred_path = save_path + filename + '__Predicted_Probabilities__' + formatted_date_time + '.png'
        p_true_path = save_path + filename + '__True_Probabilities__' + formatted_date_time + '.png'
        prob_diff_path = save_path + filename + '__Probabilitiy_Difference__' + formatted_date_time + '.png'
    else:
        p_pred_path = None
        p_true_path = None
        prob_diff_path = None

    p_pred_plot = utils.plot_samples_probability_mask(X, predictions,
                                        colorbar_label='Predicted Probability (Positive Class)',
                                        title='Predicted Probabilities (Positive Class)',
                                        save_path=p_pred_path,
                                        show=show)
    p_true_plot = utils.plot_samples_probability_mask(X, p_true,
                                        colorbar_label='True Probability (Positive Class)',
                                        title='True Probabilities (Positive Class)',
                                        save_path=p_true_path,
                                        show=show)
    prob_diff_plot = utils.plot_samples_probability_mask(X, np.abs(predictions - p_true),
                                        colorbar_label='Probability Difference (Positive Class)',
                                        title='Difference Predicted and True Probabilities (Positive Class)',
                                        save_path=prob_diff_path,
                                        show=show)

    return p_pred_plot, p_true_plot, prob_diff_plot


def plot_bin_count_histogram(bin_count, title):
    bin_numbers = range(len(bin_count))
    plt.bar(bin_numbers, bin_count, width=0.8, color='red', alpha=0.7)
    plt.xlabel("Bin Number")
    plt.ylabel("Sample Count")
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show(block=False)


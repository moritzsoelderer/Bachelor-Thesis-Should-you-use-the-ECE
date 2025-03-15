import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf

from src.utilities import utils, datasets

EMPTY_METRIC_MEANS = {
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

EMPTY_METRIC_STD_DEVS = {
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
    "sad_clown_imbalanced": (datasets.imbalanced_sad_clown_dataset, ([], []))  # bounds to be added for sad clown
}

# predict distinction for tensorflow and sklearn
predict_sklearn = lambda model, X_test: model.predict_proba(X_test)
predict_tf = lambda model, X_test: model.predict(X_test)


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


def plot_probability_masks(samples, true_prob, predictions, filename, date_time, show=True, save_path=None):
    formatted_date_time = date_time.strftime('%Y%m%d_%H%M%S')
    if save_path is not None:
        pred_prob_path = save_path + filename + '__Predicted_Probabilities__' + formatted_date_time + '.png'
        true_prob_path = save_path + filename + '__True_Probabilities__' + formatted_date_time + '.png'
        prob_diff_path = save_path + filename + '__Probabilitiy_Difference__' + formatted_date_time + '.png'
    else:
        pred_prob_path = None
        true_prob_path = None
        prob_diff_path = None

    pred_prob_plot = utils.plot_samples_probability_mask(samples, predictions,
                                        colorbar_label='Predicted Probability (Positive Class)',
                                        title='Predicted Probabilities (Positive Class)',
                                        save_path=pred_prob_path,
                                        show=show)
    true_prob_plot = utils.plot_samples_probability_mask(samples, true_prob,
                                        colorbar_label='True Probability (Positive Class)',
                                        title='True Probabilities (Positive Class)',
                                        save_path=true_prob_path,
                                        show=show)
    prob_diff_plot = utils.plot_samples_probability_mask(samples, np.abs(predictions - true_prob),
                                        colorbar_label='Probability Difference (Positive Class)',
                                        title='Difference Predicted and True Probabilities (Positive Class)',
                                        save_path=prob_diff_path,
                                        show=show)

    return pred_prob_plot, true_prob_plot, prob_diff_plot
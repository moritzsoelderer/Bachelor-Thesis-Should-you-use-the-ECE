import logging

import numpy as np
from keras import Input, Sequential
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import Adam, SGD
from matplotlib import pyplot as plt
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.metrics.ace import ace
from src.metrics.balance_score import balance_score
from src.metrics.ece import ece
from src.metrics.fce import fce
from src.metrics.ksce import ksce
from src.metrics.tce import tce
from src.metrics.true_ece import true_ece_binned


def svm_info():
    svm_model = SVC()  # Enable probability estimation

    param_grid = [
        {
            'probability': [True],
            'kernel': ['linear'],
            'C': [0.01, 0.1, 10],
            'class_weight': [None, 'balanced'],
            'shrinking': [True, False]
        },
        {
            'probability': [True],
            'kernel': ['poly'],
            'C': [0.01, 0.1, 10],
            'degree': [2, 3],  # Polynomial degree
            'gamma': ['scale', 'auto'],  # 'scale' and 'auto' are common gamma values
            'coef0': [0, 1],  # Independent term for the polynomial kernel
            'class_weight': [None, 'balanced']
        },
        {
            'probability': [True],
            'kernel': ['rbf'],
            'C': [0.01, 0.1, 10],
            'gamma': ['scale', 'auto'],
            'class_weight': [None, 'balanced'],
            'shrinking': [True, False]
        },
        {
            'probability': [True],
            'kernel': ['sigmoid'],
            'C': [0.01, 0.1, 10],
            'gamma': ['scale', 'auto'],
            'coef0': [0, 1],  # Independent term in the sigmoid kernel
            'class_weight': [None, 'balanced'],
            'shrinking': [True, False]
        },
    ]

    return svm_model, param_grid

def get_create_neural_network_fun(sample_dim: int):
    if sample_dim == 2:
        return create_neural_network_2d
    elif sample_dim == 3:
        return create_neural_network_3d


def create_neural_network_2d(optimizer='adam', activation='relu', neurons=12, layers=1, dropout_rate=0.2,
                          learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(2,)))
    model.add(Dense(neurons, activation=activation))

    for _ in range(layers - 1):
        model.add(Dense(neurons, activation=activation))

    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    if optimizer == 'adam':
        optimizer_instance = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer_instance = SGD(learning_rate=learning_rate)
    else:
        optimizer_instance = optimizer

    model.compile(optimizer=optimizer_instance, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_neural_network_3d(optimizer='adam', activation='relu', neurons=12, layers=1, dropout_rate=0.2,
                          learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(3,)))
    model.add(Dense(neurons, activation=activation))

    for _ in range(layers - 1):
        model.add(Dense(neurons, activation=activation))

    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    if optimizer == 'adam':
        optimizer_instance = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer_instance = SGD(learning_rate=learning_rate)
    else:
        optimizer_instance = optimizer

    model.compile(optimizer=optimizer_instance, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def neural_network_info(sample_dim: int):
    def fun():
        model = KerasClassifier(model=get_create_neural_network_fun(sample_dim))
        param_grid = {
            'batch_size': [32, 64],
            'epochs': [15],
            'model__optimizer': ['adam', 'sgd'],
            'model__activation': ['relu', 'sigmoid'],
            'model__neurons': [10, 20],  # Number of neurons in each layer
            'model__layers': [2, 3],  # Number of hidden layers
            'model__dropout_rate': [0.2, 0.4],  # Dropout rate for regularization
            'model__learning_rate': [0.01, 0.1]  # Learning rate for Adam and SGD optimizers
        }
        return model, param_grid
    return fun


def logistic_regression_info():
    model = LogisticRegression()
    param_grid = [
        {
            'solver': ['liblinear'],
            'penalty': ['l1'],
            'C': [0.1, 10, 100],
            'class_weight': [None, 'balanced'],
            'dual': [False],
            'tol': [1e-4, 1e-2],
            'max_iter': [500, 1000]
        },
        {
            'solver': ['liblinear'],
            'penalty': ['l2'],
            'C': [0.1, 10, 100],
            'class_weight': [None, 'balanced'],
            'dual': [True],
            'tol': [1e-4, 1e-2],
            'max_iter': [500, 1000]
        },
        {
            'solver': ['lbfgs'],
            'penalty': ['l2'],
            'C': [0.1, 10, 100],
            'class_weight': [None, 'balanced'],
            'tol': [1e-4, 1e-2],
            'max_iter': [500, 1000]
        },
        {
            'solver': ['saga'],
            'penalty': ['elasticnet'],
            'C': [0.1, 10, 100],
            'class_weight': [None, 'balanced'],
            'l1_ratio': [0.1, 0.6],  # Only valid for elasticnet
            'tol': [1e-4, 1e-2],
            'max_iter': [500, 1000],
        },
        {
            'solver': ['saga'],
            'penalty': ['l2'],
            'C': [0.1, 10, 100],
            'class_weight': [None, 'balanced'],
            'tol': [1e-4, 1e-2],
            'max_iter': [500, 1000],
        },
        {
            'solver': ['newton-cg'],
            'penalty': ['l2'],
            'C': [0.1, 10, 100],
            'class_weight': [None, 'balanced'],
            'tol': [1e-4, 1e-2],
            'max_iter': [500, 1000],
        },
    ]
    return model, param_grid


def random_forest_info():
    model = RandomForestClassifier()
    param_grid = {
        'n_estimators': [100],  # Number of trees
        'max_depth': [None, 10, 20],  # Maximum depth of each tree
        'min_samples_split': [2, 5],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [2, 4],  # Minimum number of samples required to be at a leaf node
        'max_features': [2, 'sqrt', 'log2'],  # The number of features to consider for a split
        'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
        'criterion': ['gini', 'entropy'],  # The function to measure the quality of a split
    }
    return model, param_grid


def sort_by_key_index(metric_values: dict, keyIndex: int, reverse: bool):
    combined = list(zip(
        metric_values["True ECE Grid (Binned - 100 Bins)"],
        metric_values["True ECE Grid (Binned - 15 Bins)"],
        metric_values["ECE"],
        metric_values["Balance Score"],
        metric_values["FCE"],
        metric_values["KSCE"],
        metric_values["TCE"],
        metric_values["ACE"],
        metric_values["True ECE Dists (Binned - 100 Bins)"],
        metric_values["True ECE Dists (Binned - 15 Bins)"],
        metric_values["Accuracy"]
    ))
    combined_sorted = sorted(combined, key=lambda x: x[keyIndex], reverse=reverse)

    metric_values_sorted = {
        "True ECE Grid (Binned - 100 Bins)": [item[0] for item in combined_sorted],
        "True ECE Grid (Binned - 15 Bins)": [item[1] for item in combined_sorted],
        "ECE": [item[2] for item in combined_sorted],
        "Balance Score": [item[3] for item in combined_sorted],
        "FCE": [item[4] for item in combined_sorted],
        "KSCE": [item[5] for item in combined_sorted],
        "TCE": [item[6] for item in combined_sorted],
        "ACE": [item[7] for item in combined_sorted],
        "True ECE Dists (Binned - 100 Bins)": [item[8] for item in combined_sorted],
        "True ECE Dists (Binned - 15 Bins)": [item[9] for item in combined_sorted],
        "Accuracy": [item[10] for item in combined_sorted]
    }
    return metric_values_sorted


def remove_every_nth_element(lst, d):
    return [lst[i] for i in range(len(lst)) if i % d == 1]

def plot_absolute_metrics(dataset_name, model_name, sorted_by, num_estimators, metric_values_sorted, datetime_now, sample_size, num_folds):
    # Plotting Absolute Metrics #
    plt.figure(figsize=(18, 6), dpi=150)
    plt.title("Grid Search " + model_name + " - Estimators: " + str(num_estimators) + ", Folds: " + str(num_folds) + ", Sample Size: " + str(sample_size), fontsize=14, fontweight='bold')
    plt.xlabel(model_name + 's' + " (Sorted by: " + sorted_by + ")", fontsize=12)
    plt.ylabel("Metrics", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    x_values = range(1, num_estimators + 1)
    plt.plot(x_values, metric_values_sorted['True ECE Grid (Binned - 100 Bins)'], label="True ECE Grid (Binned - 100 Bins)")
    plt.plot(x_values, metric_values_sorted['True ECE Grid (Binned - 15 Bins)'], label="True ECE Grid (Binned - 15 Bins)")
    plt.plot(x_values, metric_values_sorted['ECE'], label="ECE")
    plt.plot(x_values, metric_values_sorted['Balance Score'], label="Balance Score")
    plt.plot(x_values, metric_values_sorted['FCE'], label="FCE")
    plt.plot(x_values, metric_values_sorted['KSCE'], label="KSCE")
    plt.plot(x_values, metric_values_sorted['TCE'], label="TCE")
    plt.plot(x_values, metric_values_sorted['ACE'], label="ACE")
    plt.plot(x_values, metric_values_sorted['True ECE Dists (Binned - 100 Bins)'], label="True ECE Dists (Binned - 100 Bins)")
    plt.plot(x_values, metric_values_sorted['True ECE Dists (Binned - 15 Bins)'], label="True ECE Dists (Binned - 15 Bins)")
    plt.plot(x_values, metric_values_sorted['Accuracy'], label="Accuracy")

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Position outside the top-right corner
    plt.tight_layout(pad=1.12)
    filename = f"{dataset_name}__{model_name}__Samples_{sample_size}__Estimators_{x_values[-1]}__Folds_{num_folds}__AbsoluteValues__SortedBy_{sorted_by}__{datetime_now.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig("./plots/" + filename)
    plt.show(block=False)

def plot_relative_metrics(dataset_name, model_name, sorted_by, num_estimators, metric_values_sorted, datetime_now, sample_size, num_folds):
    # Plotting Relative Metrics #
    plt.figure(figsize=(18, 6))
    plt.title("Grid Search " + model_name + " (Relative Values) " + "- Estimators: " + str(num_estimators) + ", Folds: " + str(num_folds) + ", Sample Size: " + str(sample_size), fontsize=14, fontweight='bold')
    plt.xlabel(model_name + 's' + " (Sorted by: " + sorted_by + ")", fontsize=12)
    plt.ylabel("Metrics (Relative to True ECE Grid (Binned - 100 Bins))", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    x_values = range(1, num_estimators + 1)
    true_ece_array = np.array(metric_values_sorted['True ECE Grid (Binned - 100 Bins)'])
    plt.plot(x_values, metric_values_sorted['True ECE Grid (Binned - 100 Bins)'] - true_ece_array, label="True ECE Grid (Binned - 100 Bins)")
    plt.plot(x_values, metric_values_sorted['True ECE Grid (Binned - 15 Bins)'] - true_ece_array, label="True ECE Grid (Binned - 15 Bins)")
    plt.plot(x_values, metric_values_sorted['ECE'] - true_ece_array, label="ECE")
    plt.plot(x_values, metric_values_sorted['Balance Score'] - true_ece_array, label="Balance Score")
    plt.plot(x_values, metric_values_sorted['FCE'] - true_ece_array, label="FCE")
    plt.plot(x_values, metric_values_sorted['KSCE'] - true_ece_array, label="KSCE")
    plt.plot(x_values, metric_values_sorted['TCE'] - true_ece_array, label="TCE")
    plt.plot(x_values, metric_values_sorted['ACE'] - true_ece_array, label="ACE")
    plt.plot(x_values, metric_values_sorted['True ECE Dists (Binned - 100 Bins)'] - true_ece_array, label="True ECE Dists (Binned - 100 Bins)")
    plt.plot(x_values, metric_values_sorted['True ECE Dists (Binned - 15 Bins)'] - true_ece_array, label="True ECE Dists (Binned - 15 Bins)")
    plt.plot(x_values, metric_values_sorted['Accuracy'], label="Accuracy")

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Position outside the top-right corner
    plt.tight_layout(pad=1.12)
    filename = f"{dataset_name}__{model_name}__Samples_{sample_size}__Estimators_{x_values[-1]}__Folds_{num_folds}__RelativeValues__SortedBy_{sorted_by}__{datetime_now.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig("./plots/" + filename)
    plt.show(block=False)

def process_model(accuracy, estimator, X_test, y_test, p_true, X_dists, X_grid, p_dists_true, p_grid_true):
    logging.info(f"Predicting with model: {estimator}")
    logging.info(f"{X_test.shape}, {y_test.shape}, {p_true.shape}")

    pred_prob = estimator.predict_proba(X_test)
    pred_prob_dists = estimator.predict_proba(X_dists)
    pred_prob_grid = estimator.predict_proba(X_grid)

    # Evaluate metrics
    true_ece_dists_15bins, _ = true_ece_binned(pred_prob_dists, p_dists_true, np.linspace(0, 1, 15))
    true_ece_dists_100bins, _ = true_ece_binned(pred_prob_dists, p_dists_true, np.linspace(0, 1, 100))

    true_ece_grid_15bins, _ = true_ece_binned(pred_prob_grid, p_grid_true, np.linspace(0, 1, 15))
    true_ece_grid_100bins, _ = true_ece_binned(pred_prob_grid, p_grid_true, np.linspace(0, 1, 100))

    ece_score = ece(pred_prob, y_test, 15)
    balance_score_score = np.abs(balance_score(pred_prob, y_test))
    fce_score = fce(pred_prob, y_test, 15)
    ksce_score = ksce(pred_prob, y_test)
    tce_score = tce(pred_prob, y_test, n_bin=15) / 100.0
    ace_score = ace(pred_prob, y_test, 15)

    # Store metric values
    return {
        "True ECE Grid (Binned - 100 Bins)": true_ece_grid_100bins,
        "True ECE Grid (Binned - 15 Bins)": true_ece_grid_15bins,
        "ECE": ece_score,
        "Balance Score": balance_score_score,
        "FCE": fce_score,
        "KSCE": ksce_score,
        "TCE": tce_score,
        "ACE": ace_score,
        "True ECE Dists (Binned - 100 Bins)": true_ece_dists_100bins,
        "True ECE Dists (Binned - 15 Bins)": true_ece_dists_15bins,
        "Accuracy": accuracy
    }

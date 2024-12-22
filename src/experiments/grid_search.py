import pickle
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed
from keras import Sequential, Input
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import Adam, SGD
from matplotlib import pyplot as plt
from scikeras.wrappers import KerasClassifier
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
from src.metrics.true_ece import true_ece
from src.qualitative_analysis import util
from _search_modified import GridSearchWithEstimatorOutput


def svm_info():
    svm_model = SVC()  # Enable probability estimation

    param_grid = [
        # Solver with linear kernel - no need for degree or gamma
        {
            'probability': [True],
            'kernel': ['linear'],
            'C': [0.01, 0.1, 10],
            'class_weight': [None, 'balanced'],
            'shrinking': [True, False]
        },
        # Solver with polynomial kernel - requires degree and gamma
        {
            'probability': [True],
            'kernel': ['poly'],
            'C': [0.01, 0.1, 10],
            'degree': [2, 3],  # Polynomial degree
            'gamma': ['scale', 'auto'],  # 'scale' and 'auto' are common gamma values
            'coef0': [0, 1],  # Independent term for the polynomial kernel
            'class_weight': [None, 'balanced']
        },
        # Solver with rbf kernel - requires gamma, no degree or coef0
        {
            'probability': [True],
            'kernel': ['rbf'],
            'C': [0.01, 0.1, 10],
            'gamma': ['scale', 'auto'],
            'class_weight': [None, 'balanced'],
            'shrinking': [True, False]
        },
        # Solver with sigmoid kernel - requires gamma and coef0
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


def create_neural_network(optimizer='adam', activation='relu', neurons=12, layers=1, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()

    model.add(Input(shape=(2,)))
    # Add input layer
    model.add(Dense(neurons, activation=activation))  # Input layer with `neurons` number of neurons

    # Add hidden layers
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation=activation))  # Adding additional hidden layers

    # Dropout layer for regularization
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification)

    # Optimizer with learning rate if specified
    if optimizer == 'adam':
        optimizer_instance = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer_instance = SGD(learning_rate=learning_rate)
    else:
        optimizer_instance = optimizer

    # Compile the model with the selected optimizer
    model.compile(optimizer=optimizer_instance, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def neural_network_info():
    model = KerasClassifier(model=create_neural_network)

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

def process_model(accuracy, estimator, X_test, y_test, true_probabilities):

    print(f"   Predicting with model: {estimator}")
    print(X_test.shape, " : ", y_test.shape, " : ", true_probabilities.shape)
    predictions = estimator.predict_proba(X_test)

    # Evaluate metrics
    true_ece_score = true_ece(predictions, true_probabilities)
    ece_score = ece(predictions, y_test, 15)
    balance_score_score = np.abs(balance_score(predictions, y_test))
    fce_score = fce(predictions, y_test, 15)
    ksce_score = ksce(predictions, y_test)
    tce_score = tce(predictions, y_test, n_bin=15) / 100.0
    ace_score = ace(predictions, y_test, 15)

    # Store metric values
    return {
        'accuracy': accuracy,
        'true_ece': true_ece_score,
        'ece': ece_score,
        'balance_score': balance_score_score,
        'fce': fce_score,
        'ksce': ksce_score,
        'tce': tce_score,
        'ace': ace_score
    }

def sort_by_key_index(metric_values: dict, keyIndex: int, reverse: bool):
    combined = list(zip(
        metric_values['accuracy'],
        metric_values['true_ece'],
        metric_values['ece'],
        metric_values['balance_score'],
        metric_values['fce'],
        metric_values['ksce'],
        metric_values['tce'],
        metric_values['ace']
    ))
    # Sort combined tuples by element
    combined_sorted = sorted(combined, key=lambda x: x[keyIndex], reverse=reverse)

    # Unpack sorted tuples back into metric_values
    metric_values_sorted = {
        'accuracy': [item[0] for item in combined_sorted],
        'true_ece': [item[1] for item in combined_sorted],
        'ece': [item[2] for item in combined_sorted],
        'balance_score': [item[3] for item in combined_sorted],
        'fce': [item[4] for item in combined_sorted],
        'ksce': [item[5] for item in combined_sorted],
        'tce': [item[6] for item in combined_sorted],
        'ace': [item[7] for item in combined_sorted]
    }

    return metric_values_sorted


def plot_absolute_metrics(model_name, sorted_by, num_estimators, metric_values_sorted, datetime_now, sample_size, num_folds):
    # Plotting Absolute Metrics #
    plt.figure(figsize=(18, 6), dpi=150)
    plt.title("Grid Search " + model_name + " - Estimators: " + str(num_estimators) + ", Folds: " + str(num_folds) + ", Sample Size: " + str(sample_size), fontsize=14, fontweight='bold')
    plt.xlabel(model_name + 's' + " (Sorted by: " + sorted_by + ")", fontsize=12)
    plt.ylabel("Metrics", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    x_values = range(1, num_estimators + 1)
    plt.plot(x_values, metric_values_sorted['true_ece'], label="True ECE")
    plt.plot(x_values, metric_values_sorted['ece'], label="ECE 15")
    plt.plot(x_values, metric_values_sorted['balance_score'], label="Balance Score")
    plt.plot(x_values, metric_values_sorted['fce'], label="FCE 15")
    plt.plot(x_values, metric_values_sorted['ksce'], label="KSCE")
    plt.plot(x_values, metric_values_sorted['tce'], label="TCE 15")
    plt.plot(x_values, metric_values_sorted['ace'], label="ACE 15")
    plt.plot(x_values, metric_values_sorted['accuracy'], label="Accuracy")

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Position outside the top-right corner
    plt.tight_layout(pad=1.12)
    filename = f"{model_name}__Samples_{sample_size}__Estimators_{x_values[-1]}__Folds_{num_folds}__AbsoluteValues__SortedBy_{sorted_by}__{datetime_now.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig("./plots/grid_search/" + filename)
    plt.show()

def plot_relative_metrics(model_name, sorted_by, num_estimators, metric_values_sorted, datetime_now, sample_size, num_folds):
    # Plotting Relative Metrics #
    plt.figure(figsize=(18, 6))
    plt.title("Grid Search " + model_name + " (Relative Values) " + "- Estimators: " + str(num_estimators) + ", Folds: " + str(num_folds) + ", Sample Size: " + str(sample_size), fontsize=14, fontweight='bold')
    plt.xlabel(model_name + 's' + " (Sorted by: " + sorted_by + ")", fontsize=12)
    plt.ylabel("Metrics (Relative to True ECE)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    x_values = range(1, num_estimators + 1)
    true_ece_array = np.array(metric_values_sorted['true_ece'])
    plt.plot(x_values, metric_values_sorted['true_ece'] - true_ece_array, label="True ECE")
    plt.plot(x_values, metric_values_sorted['ece'] - true_ece_array, label="ECE 15")
    plt.plot(x_values, metric_values_sorted['balance_score'] - true_ece_array, label="Balance Score")
    plt.plot(x_values, metric_values_sorted['fce'] - true_ece_array, label="FCE 15")
    plt.plot(x_values, metric_values_sorted['ksce'] - true_ece_array, label="KSCE")
    plt.plot(x_values, metric_values_sorted['tce'] - true_ece_array, label="TCE 15")
    plt.plot(x_values, metric_values_sorted['ace'] - true_ece_array, label="ACE 15")
    plt.plot(x_values, metric_values_sorted['accuracy'], label="Accuracy")

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Position outside the top-right corner
    plt.tight_layout(pad=1.12)
    filename = f"{model_name}__Samples_{sample_size}__Estimators_{x_values[-1]}__Folds_{num_folds}__RelativeValues__SortedBy_{sorted_by}__{datetime_now.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig("./plots/grid_search/" + filename)
    plt.show()

model_infos = {
    #"SVM": svm_info,
    #"Neural Network": neural_network_info,
    "Logistic Regression": logistic_regression_info,
    "Random Forest": random_forest_info
}

sample_size = 10000
num_folds = 5

def main():
    data_generation = util.gummy_worm_dataset()
    samples, labels = data_generation.generate_data(n_examples=sample_size)

    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=.2)
    true_probabilities = np.array(
        [[data_generation.cond_prob(x, k=0), data_generation.cond_prob(x, k=1)] for x in X_test])

    for model_name, model_info in model_infos.items():
        print("Model: ", model_name)

        model, parameter_grid = model_info()

        print(" Performing GridSearch")
        grid_search = GridSearchWithEstimatorOutput(estimator=model, param_grid=parameter_grid, scoring='accuracy', cv=num_folds, n_jobs=-2, verbose=3)
        grid_search.fit(X_train, y_train)

        accuracies = grid_search.cv_results_["mean_test_score"]
        estimators = grid_search.cv_results_["estimator"]
        print("Estimators: ", estimators)

        metric_values = {
            "accuracy": [],
            "true_ece": [],
            "ece": [],
            "balance_score": [],
            "fce": [],
            "ksce": [],
            "tce": [],
            "ace": []
        }

        print("Length Accuracies and Estimators:", len(estimators))

        results = Parallel(n_jobs=-2, verbose=10)(  # n_jobs=-1 uses all available CPUs
            delayed(process_model)(accuracies[i], estimators[i], X_test.copy(), y_test.copy(), true_probabilities.copy())
            for i in range(len(estimators))
        )

        # Persisting Results #
        print("DEBUG: Persisting Results...")
        datetime_now = datetime.now()
        num_estimators = len(estimators)
        pickle_filename = f"{model_name}__Samples_{sample_size}__Estimators_{num_estimators}__Folds_{num_folds}__AbsoluteValues__{datetime_now.strftime('%Y%m%d_%H%M%S')}"
        with open('./data/grid_search/' + pickle_filename + '.pkl', 'wb') as file:
            pickle.dump(results, file)

        print("Results: ", results)
        print("Length Results:", len(results))
        for result in results:
            print("Result: ", result)
            # Update the shared metric_values dictionary
            metric_values['accuracy'].append(result['accuracy'])
            metric_values['true_ece'].append(result['true_ece'])
            metric_values['ece'].append(result['ece'])
            metric_values['balance_score'].append(result['balance_score'])
            metric_values['fce'].append(result['fce'])
            metric_values['ksce'].append(result['ksce'])
            metric_values['tce'].append(result['tce'])
            metric_values['ace'].append(result['ace'])

        print("Metric Values: ", metric_values)
        print("Length Metric Values: ", len(metric_values))

        print(" Plotting...")

        indices_and_sort_order = [(0, False), (1, True)]

        for index, sort_order in indices_and_sort_order:
            metric_values_sorted = sort_by_key_index(metric_values, index, reverse=sort_order)
            sorted_by = list(metric_values.keys())[index]
            plot_absolute_metrics(model_name, sorted_by, num_estimators, metric_values_sorted, datetime_now, sample_size, num_folds)
            plot_relative_metrics(model_name, sorted_by, num_estimators, metric_values_sorted, datetime_now, sample_size, num_folds)





if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    main()
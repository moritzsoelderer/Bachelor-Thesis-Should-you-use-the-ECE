import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from metrics.ace import ace
from metrics.balance_score import balance_score
from metrics.ece import ece
from metrics.fce import fce
from metrics.ksce import ksce
from metrics.tce import tce
from metrics.true_ece import true_ece
from qualitative_analysis import util

model = SVC(probability=True)
parameter_grid = {
    'C': [0.1, 1, 100],                   # Regularization parameter
    'kernel': ['linear', 'rbf', 'sigmoid'],  # Kernels to try
    'gamma': ['scale', 'auto'],                # Gamma values to try
}

data_generation = util.gummy_worm_dataset()
samples, labels = data_generation.generate_data(n_examples=100)

true_probabilities = np.array([[data_generation.cond_prob(x, k=0), data_generation.cond_prob(x, k=1)] for x in samples])

grid_search = GridSearchCV(estimator=model, param_grid=parameter_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(samples, labels)

accuracies = grid_search.cv_results_["mean_test_score"]
parameters = grid_search.cv_results_["params"]

sorted_models = sorted(zip(accuracies, parameters), key= lambda x: x[0], reverse=True)

metric_values = {
    "true_ece": [],
    "ece": [],
    "balance_score": [],
    "fce": [],
    "ksce": [],
    "tce": [],
    "ace": []
}

for score, params in sorted_models:
    title = "Accuracy: " + str(score) + ", Parameters: " + str(params)

    svc = SVC(kernel=params['kernel'], gamma=params['gamma'], C=params['C'], probability=True)
    svc.fit(samples, labels)
    predictions = svc.predict_proba(samples)
    print(predictions)

    # util.plot_true_prob_reliability_diagram(true_probabilities, true_labels=labels, pred_prob=predictions, title=title)

    print("Evaluating Metrics...")
    true_ece_score = true_ece(predictions, true_probabilities)
    ece_score = ece(predictions, labels, 15)
    balance_score_score = np.abs(balance_score(predictions, labels))
    fce_score = fce(predictions, labels, 15)
    ksce_score = ksce(predictions, labels)
    tce_score = tce(predictions, labels, n_bin=15) / 100.0
    ace_score = ace(predictions, labels, 15)

    # Store Metric Values #
    metric_values["ece"].append(ece_score)
    metric_values["fce"].append(fce_score)
    metric_values["ksce"].append(ksce_score)
    metric_values["tce"].append(tce_score)
    metric_values["ace"].append(ace_score)
    metric_values["true_ece"].append(true_ece_score)
    metric_values["balance_score"].append(balance_score_score)

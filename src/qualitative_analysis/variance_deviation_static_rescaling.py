import random

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from src.metrics.ace import ace
from src.metrics.balance_score import balance_score
from src.metrics.ece import ece
from src.metrics.fce import fce
from src.metrics.ksce import ksce
from src.metrics.tce import tce
from src.metrics.true_ece import true_ece
from src.qualitative_analysis import util

# declare independent variables (configure for experiment adjustements)
variance = 4
max_negative_variance_deviation = -3
max_positive_variance_deviation = 4
deviation_steps = 0.1

# declare dependent variables (values infered with independent variables)
scaling_constant = variance * np.sqrt(2 * np.pi)/4  # inverse of max. value of gaussian
p_true_dist = stats.norm(loc=0, scale=variance)
X = p_true_dist.rvs(size=40000)
pdf_values = p_true_dist.pdf(X) * scaling_constant
p_true = np.column_stack((1 - pdf_values, pdf_values))
y_true = np.array(list(map(lambda x : 1 if random.random() < x[1] else 0, p_true)))

p_true_ece = ece(p_true, y_true, 15)
p_true_balance_score = balance_score(p_true, y_true)
variance_deviations = np.arange(max_negative_variance_deviation, max_positive_variance_deviation, deviation_steps)


# plot X, conditional probabilities and labels (for testing purposes, can later be deleted)
plt.hist(X)
plt.show(block=False)
plt.scatter(X, p_true[:, 0])
plt.show(block=False)
plt.scatter(X, p_true[:, 1])
plt.show(block=False)
plt.scatter(X, y_true)
plt.show(block=False)

# plotting some of the conditional probabilities resulting from variance deviation of the underlying distribution (can be deleted later on)
for deviation in np.arange(max_negative_variance_deviation, max_positive_variance_deviation, 1):
    #scaling_constant = (variance + deviation) * np.sqrt(2 * np.pi)
    predicted_prob_dist = stats.norm(loc=0, scale=variance + deviation)
    # Get the predicted probabilities for all X at once
    pdf_values = predicted_prob_dist.pdf(X) * scaling_constant
    # Create the result by stacking the arrays for 1 - p and p
    p_pred = np.column_stack((1 - pdf_values, pdf_values))
    plt.scatter(X, p_pred[:, 1], label="variance: " + str(variance + deviation), s=4)
plt.legend()
plt.show(block=False)

# plotting some of the conditional probabilities resulting from variance deviation of the underlying distribution (can be deleted later on)
for deviation in np.arange(max_negative_variance_deviation, max_positive_variance_deviation, 1):
    predicted_prob_dist = stats.norm(loc=1, scale=variance + deviation)
    pdf_values = predicted_prob_dist.pdf(X) * scaling_constant
    p_pred = np.column_stack((1 - pdf_values, pdf_values))

    # plotting predicted probabilities against true probabilities
    colors = ['#111111', '#FF5733']  # Define colors for the classes
    class_colors = [colors[label] for label in y_true]
    plt.scatter(p_pred[:, 1],
                p_true[:, 1], c=class_colors, s=0.9)

    unique_labels = np.unique(y_true)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label='Class ' + str(label),
                          markerfacecolor=colors[label], markersize=10) for label in unique_labels]

    # Plotting the scatter plot
    plt.xlabel("Predicted Probabilities")
    plt.ylabel("True Probabilities")
    plt.title("Reliability Diagram - Variance: " + str(variance + deviation))
    plt.legend(handles=handles, title="True Labels")
    plt.show(block=False)


true_ece_vals = np.array([], dtype=np.float64)
ece_vals = np.array([], dtype=np.float64)
balance_score_vals = np.array([], dtype=np.float64)
fce_vals = np.array([], dtype=np.float64)
tce_vals = np.array([], dtype=np.float64)
ksce_vals = np.array([], dtype=np.float64)
ace_vals = np.array([], dtype=np.float64)

# deviating variances of p_trueability distribution and calculating ece values (note that each conditional
# probability plot is a proper upscaled version of the underlying distribution plot with updated variance)
# must later be extended with other values
print("ECE values")
print("ece true prob", p_true_ece)
print("balance score true prob", p_true_balance_score)
for variance_deviation in variance_deviations:
    #scaling_constant = (variance + variance_deviation) * np.sqrt(2 * np.pi)  # inverse of max. value of gaussian
    print("variance_deviation", variance_deviation)
    predicted_prob_dist = stats.norm(loc=0, scale=variance + variance_deviation)
    # Get the predicted probabilities for all X at once
    pdf_values = predicted_prob_dist.pdf(X) * scaling_constant
    # Create the result by stacking the arrays for 1 - p and p
    p_pred = np.column_stack((1 - pdf_values, pdf_values))

    # calculating true ece vals

    true_ece_val = true_ece(p_pred, p_true)
    true_ece_vals = np.append(true_ece_vals, [true_ece_val])

    # calculating ece vals
    ece_val = ece(p_pred, y_true, 15)
    ece_vals = np.append(ece_vals, [ece_val])

    # calculating balance score vals
    balance_score_val = balance_score(p_pred, y_true)
    balance_score_vals = np.append(balance_score_vals, [balance_score_val])

    # calculating fce vals
    fce_val = fce(p_pred, y_true, 15)
    fce_vals = np.append(fce_vals, [fce_val])

    # calculating tce vals
    tce_val = tce(p_pred, y_true, 0.05, "uniform", 15, 15, 15) * 0.01
    tce_vals = np.append(tce_vals, [tce_val])

    # calculating ksce vals
    ksce_val = ksce(p_pred, y_true)
    ksce_vals = np.append(ksce_vals, [ksce_val])

    # calculating ace vals
    ace_val = ace(p_pred, y_true, 15)
    ace_vals = np.append(ace_vals, [ace_val])

util.plot_metrics(variance, variance + variance_deviations, p_true, y_true, true_ece_vals, ece_vals,
                  balance_score_vals,
                  fce_vals, tce_vals, ksce_vals, ace_vals, "Metric Behaviour on Variance Deviation", "Variance")

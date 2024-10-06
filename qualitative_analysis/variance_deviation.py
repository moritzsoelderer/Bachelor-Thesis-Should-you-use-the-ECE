import random

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from metrics.ace import ace
from metrics.balance_score import balance_score
from metrics.ece import ece
from metrics.fce import fce
from metrics.ksce import ksce
from metrics.tce import tce
from metrics.true_ece import true_ece
from qualitative_analysis.mean_deviation import plot_metrics, plot_pred_prob_dists, plot_true_prob_reliability_diagram


def main():
    # declare independent variables (configure for experiment adjustements)
    variance = 4
    max_negative_variance_deviation = -3
    max_positive_variance_deviation = 4
    deviation_steps = 0.1

    # declare dependent variables (values infered with independent variables)
    scaling_constant = variance * np.sqrt(2 * np.pi)  # inverse of max. value of gaussian
    true_prob_dist = stats.norm(loc=0, scale=variance)
    samples = true_prob_dist.rvs(size=40000)
    true_prob = np.array(list(
        map(lambda x: [1 - true_prob_dist.pdf(x) * scaling_constant, true_prob_dist.pdf(x) * scaling_constant],
            samples)))
    true_labels = np.array(list(map(lambda x: 1 if random.random() < x[1] else 0, true_prob)))

    true_prob_ece = ece(true_prob, true_labels, 15)
    true_prob_balance_score = balance_score(true_prob, true_labels)
    variance_deviations = np.arange(max_negative_variance_deviation, max_positive_variance_deviation, deviation_steps)

    # plotting some of the distributions
    plot_pred_prob_dists(np.arange(variance + max_negative_variance_deviation,
                                   variance + max_positive_variance_deviation, 1), samples,
                         "variance")
    # plotting some of the true probability reliability diagrams
    plot_true_prob_reliability_diagram(np.arange(variance + max_negative_variance_deviation,
                                                 variance + max_positive_variance_deviation, 1), samples, true_prob,
                                       true_labels, "variance")

    true_ece_vals = np.array([], dtype=np.float64)
    ece_vals = np.array([], dtype=np.float64)
    balance_score_vals = np.array([], dtype=np.float64)
    fce_vals = np.array([], dtype=np.float64)
    tce_vals = np.array([], dtype=np.float64)
    ksce_vals = np.array([], dtype=np.float64)
    ace_vals = np.array([], dtype=np.float64)

    # deviating variances of true_probability distribution and calculating ece values (note that each conditional
    # probability plot is a proper upscaled version of the underlying distribution plot with updated variance)
    # must later be extended with other values
    print("ECE values")
    print("ece true prob", true_prob_ece)
    print("balance score true prob", true_prob_balance_score)
    for variance_deviation in variance_deviations:
        print("variance_deviation: ", variance_deviation)
        scaling_constant = (variance + variance_deviation) * np.sqrt(2 * np.pi)  # inverse of max. value of gaussian
        predicted_prob_dist = stats.norm(loc=0, scale=variance + variance_deviation)
        # Get the predicted probabilities for all samples at once
        pdf_values = predicted_prob_dist.pdf(samples) * scaling_constant
        # Create the result by stacking the arrays for 1 - p and p
        pred_prob = np.column_stack((1 - pdf_values, pdf_values))

        # calculating true ece vals
        true_ece_val = true_ece(pred_prob, true_prob)
        true_ece_vals = np.append(true_ece_vals, [true_ece_val])

        # calculating ece vals
        ece_val = ece(pred_prob, true_labels, 15)
        ece_vals = np.append(ece_vals, [ece_val])

        # calculating balance score vals
        balance_score_val = balance_score(pred_prob, true_labels)
        balance_score_vals = np.append(balance_score_vals, [balance_score_val])

        # calculating fce vals
        fce_val = fce(pred_prob, true_labels, 15)
        fce_vals = np.append(fce_vals, [fce_val])

        # calculating tce vals
        tce_val = tce(pred_prob, true_labels, 0.05, "uniform", 15, 15, 15) * 0.01
        tce_vals = np.append(tce_vals, [tce_val])

        # calculating ksce vals
        ksce_val = ksce(pred_prob, true_labels)
        ksce_vals = np.append(ksce_vals, [ksce_val])

        # calculating ace vals
        ace_val = ace(pred_prob, true_labels, 15)
        ace_vals = np.append(ace_vals, [ace_val])

    plot_metrics(variance, variance_deviations, true_prob, true_labels, true_ece_vals, ece_vals, balance_score_vals,
                 fce_vals, tce_vals, ksce_vals, ace_vals, "Metric Behaviour on Variance Deviation")


if __name__ == "__main__":
    main()

import random

import numpy as np
import scipy.stats as stats

from src.metrics.ace import ace
from src.metrics.balance_score import balance_score
from src.metrics.ece import ece
from src.metrics.fce import fce
from src.metrics.ksce import ksce
from src.metrics.tce import tce
from src.metrics.true_ece import true_ece
from src.qualitative_analysis import util


def calculate_probabilities(X, step):
    # noise = stats.norm(loc=0, scale=0.05).rvs(size=X.shape[0])
    prob_dist = stats.norm(loc=0, scale=step)
    scaling_constant = step * np.sqrt(2 * np.pi)
    # pdf_values = prob_dist.pdf(X) * scaling_constant + noise
    pdf_values = prob_dist.pdf(X) * scaling_constant
    # pdf_values = np.clip(pdf_values, 0, 1)
    prob = np.column_stack((1 - pdf_values, pdf_values))
    return prob

def main():
    # declare independent variables (configure for experiment adjustements)
    variance = 4
    max_negative_variance_deviation = -3
    max_positive_variance_deviation = 4
    deviation_steps = .1

    # declare dependent variables (values infered with independent variables)
    p_true_dist = stats.norm(loc=0, scale=variance)
    X = p_true_dist.rvs(size=40000)
    p_true = calculate_probabilities(X, variance)
    y_true = np.array(list(map(lambda x: 1 if random.random() < x[1] else 0, p_true)))

    p_true_ece = ece(p_true, y_true, 15)
    p_true_balance_score = balance_score(p_true, y_true)
    variance_deviations = np.arange(max_negative_variance_deviation, max_positive_variance_deviation, deviation_steps)

    # plotting some of the distributions
    util.plot_p_pred_dists(np.arange(variance + max_negative_variance_deviation,
                                        variance + max_positive_variance_deviation, 1), X,
                              calculate_probabilities, "Variance")
    # plotting some of the true probability reliability diagrams
    util.plot_p_true_reliability_diagrams(np.arange(variance + max_negative_variance_deviation,
                                                       variance + max_positive_variance_deviation, 1), X, p_true,
                                             y_true, calculate_probabilities, "Variance")

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
        print("variance_deviation: ", variance_deviation)
        p_pred = calculate_probabilities(X, variance + variance_deviation)

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

    util.plot_metrics(variance, variance + variance_deviations, p_true, y_true, true_ece_vals, ece_vals, balance_score_vals,
                      fce_vals, tce_vals, ksce_vals, ace_vals, "Metric Behaviour on Variance Deviation", "Variance")


if __name__ == "__main__":
    main()

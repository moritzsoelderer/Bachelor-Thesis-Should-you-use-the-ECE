import random

import numpy as np
import scipy.stats as stats

from src.qualitative_analysis import util


def calculate_probabilities(X, step):
    scaling_constant = np.sqrt(2 * np.pi)  # inverse of max. value of gaussian with variance=1
    prob_dist = stats.norm(loc=step, scale=1)
    pdf_values = prob_dist.pdf(X) * scaling_constant
    prob = np.column_stack((1 - pdf_values, pdf_values))
    return prob

def main():
    mean = 0
    mean_deviations = np.arange(-3, 4, 1)
    p_true_dist = stats.norm(loc=mean, scale=1)
    X = p_true_dist.rvs(size=40000)

    p_true = calculate_probabilities(X, mean)
    y_true = np.array(list(map(lambda x: 1 if random.random() < x[1] else 0, p_true)))

    util.plot_p_pred_dists(np.arange(mean - 4, mean + 4, 2), X, calculate_probabilities, "mean")
    util.plot_p_true_reliability_diagrams(np.arange(mean - 4, mean + 4, 1), X, p_true, y_true, calculate_probabilities, "Mean")

    true_ece_vals, ece_vals, balance_score_vals, fce_vals, tce_vals, ksce_vals, ace_vals = util.calculate_metrics(
        mean + mean_deviations, p_true,
        y_true, X, calculate_probabilities,
        log="Mean Deviation"
    )

    util.plot_metrics(mean, mean_deviations, p_true, y_true, true_ece_vals, ece_vals, balance_score_vals, fce_vals,
                      tce_vals, ksce_vals, ace_vals, "Metric Behaviour on Mean Deviation", "Mean")


if __name__ == "__main__":
    main()

import random

import numpy as np
import scipy.stats as stats

from qualitative_analysis import util


def calculate_probabilities(samples, step):
    scaling_constant = np.sqrt(2 * np.pi)  # inverse of max. value of gaussian with variance=1
    prob_dist = stats.norm(loc=step, scale=1)
    pdf_values = prob_dist.pdf(samples) * scaling_constant
    prob = np.column_stack((1 - pdf_values, pdf_values))
    return prob

def main():
    mean = 0
    mean_deviations = np.arange(-3, 4, 1)
    true_prob_dist = stats.norm(loc=mean, scale=1)
    samples = true_prob_dist.rvs(size=40000)

    true_prob = calculate_probabilities(samples, mean)
    true_labels = np.array(list(map(lambda x: 1 if random.random() < x[1] else 0, true_prob)))

    util.plot_pred_prob_dists(np.arange(mean - 4, mean + 4, 2), samples, calculate_probabilities, "mean")
    util.plot_true_prob_reliability_diagrams(np.arange(mean - 4, mean + 4, 1), samples, true_prob, true_labels, calculate_probabilities, "Mean")

    true_ece_vals, ece_vals, balance_score_vals, fce_vals, tce_vals, ksce_vals, ace_vals = util.calculate_metrics(
        mean + mean_deviations, true_prob,
        true_labels, samples, calculate_probabilities,
        log="Mean Deviation"
    )

    util.plot_metrics(mean, mean_deviations, true_prob, true_labels, true_ece_vals, ece_vals, balance_score_vals, fce_vals,
                 tce_vals, ksce_vals, ace_vals, "Metric Behaviour on Mean Deviation", "Mean")


if __name__ == "__main__":
    main()

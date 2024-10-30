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
from qualitative_analysis import util


def calculate_probabilities(samples, step):
    predicted_probs = 0.5 * (np.sin(samples + step) + 1)
    prob = np.column_stack((1 - predicted_probs, predicted_probs))
    return prob

def main():
    initial_value = 0
    sine_deviations = np.arange(-np.pi, np.pi, .1)
    samples = stats.uniform().rvs(size=40000) * (2 * np.pi) - 1

    pdf_values = 0.5 * (np.sin(samples) + 1)
    true_prob = np.column_stack((1 - pdf_values, pdf_values))
    true_labels = np.array(list(map(lambda x: 1 if random.random() < x[1] else 0, true_prob)))

    # plotting some of the conditional probabilities resulting from mean deviation of the underlying distribution (can be deleted later on)
    util.plot_pred_prob_dists(np.arange(initial_value - np.pi, initial_value + np.pi, 2), samples, calculate_probabilities, "Sine Shift")
    util.plot_true_prob_reliability_diagrams(np.arange(initial_value - np.pi, initial_value + np.pi + 1, np.pi / 3), samples, true_prob, true_labels, calculate_probabilities, "Sine Shift")

    true_ece_vals = np.array([], dtype=np.float64)
    ece_vals = np.array([], dtype=np.float64)
    balance_score_vals = np.array([], dtype=np.float64)
    fce_vals = np.array([], dtype=np.float64)
    tce_vals = np.array([], dtype=np.float64)
    ksce_vals = np.array([], dtype=np.float64)
    ace_vals = np.array([], dtype=np.float64)

    for sine_deviation in sine_deviations:
        print("sine shift", sine_deviation)
        pred_prob = calculate_probabilities(samples, sine_deviation)

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

    util.plot_metrics(initial_value, sine_deviations, true_prob, true_labels, true_ece_vals, ece_vals, balance_score_vals, fce_vals,
                 tce_vals, ksce_vals, ace_vals, "Metric Behaviour on Sine Deviation (Shift)", "Sine")


if __name__ == "__main__":
    main()

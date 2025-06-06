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
    predicted_probs = 0.5 * (np.sin(X + step) + 1)
    prob = np.column_stack((1 - predicted_probs, predicted_probs))
    return prob

def main():
    initial_value = 0
    sine_deviations = np.arange(-np.pi, np.pi, .1)
    X = stats.uniform().rvs(size=40000) * (2 * np.pi) - 1

    pdf_values = 0.5 * (np.sin(X) + 1)
    p_true = np.column_stack((1 - pdf_values, pdf_values))
    y_true = np.array(list(map(lambda x: 1 if random.random() < x[1] else 0, p_true)))

    # plotting some of the conditional probabilities resulting from mean deviation of the underlying distribution (can be deleted later on)
    util.plot_p_pred_dists(np.arange(initial_value - np.pi, initial_value + np.pi, 2), X, calculate_probabilities, "Sine Shift")
    util.plot_p_true_reliability_diagrams(np.arange(initial_value - np.pi, initial_value + np.pi + 1, np.pi / 3), X, p_true, y_true, calculate_probabilities, "Sine Shift")

    true_ece_vals = np.array([], dtype=np.float64)
    ece_vals = np.array([], dtype=np.float64)
    balance_score_vals = np.array([], dtype=np.float64)
    fce_vals = np.array([], dtype=np.float64)
    tce_vals = np.array([], dtype=np.float64)
    ksce_vals = np.array([], dtype=np.float64)
    ace_vals = np.array([], dtype=np.float64)

    for sine_deviation in sine_deviations:
        print("sine shift", sine_deviation)
        p_pred = calculate_probabilities(X, sine_deviation)

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

    util.plot_metrics(initial_value, sine_deviations, p_true, y_true, true_ece_vals, ece_vals, balance_score_vals, fce_vals,
                      tce_vals, ksce_vals, ace_vals, "Metric Behaviour on Sine Deviation (Shift)", "Sine")


if __name__ == "__main__":
    main()

import numpy as np

from metrics.balance_score import balance_score
from metrics.ece import ece
from metrics.fce import fce
from metrics.ksce import ksce
from metrics.tce import tce
from metrics.ace import ace
from utilities.utils import *


def true_ece(scores, true_prob):
    check_scores(scores)
    check_scores(true_prob)
    check_shapes(scores, true_prob)

    scores = np.max(scores, axis=-1)
    true_prob = np.max(true_prob, axis=-1)

    return sum(abs(np.array(scores) - np.array(true_prob))) / len(true_prob)


def calibration_error_summary(scores, labels, n_bins: np.ndarray, round_to=4):
    check_metric_params(scores, labels)

    ece_vals = np.array([])
    fce_vals = np.array([])
    tce_vals = np.array([])
    ace_vals = np.array([])

    for n_bin in n_bins:
        n_bin = check_bins(n_bin)

        ece_vals = np.append(ece_vals, round(ece(scores, labels, n_bin), round_to))
        fce_vals = np.append(fce_vals, round(fce(scores, labels, n_bin), round_to))
        tce_vals = np.append(tce_vals, round(tce(scores, labels, n_bin=n_bin, strategy="uniform"), round_to))
        ace_vals = np.append(ace_vals, round(ace(scores, labels, n_ranges=n_bin), round_to))

    ksce_val = round(ksce(scores, labels), round_to)
    balance_score_val = round(balance_score(scores, labels), round_to)

    return np.array([ece_vals, fce_vals, tce_vals, ace_vals]),  balance_score_val, ksce_val


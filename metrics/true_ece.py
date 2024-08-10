from itertools import groupby

import pandas as pd

from metrics.ace import ace
from metrics.balance_score import balance_score
from metrics.ece import ece
from metrics.fce import fce
from metrics.ksce import ksce
from metrics.tce import tce
from utilities.utils import *


def true_ece(scores, true_prob):
    check_scores(scores)
    check_scores(true_prob)
    check_shapes(scores, true_prob)

    scores = np.array(scores[:, 1])
    true_prob = np.array(true_prob[:, 1])

    scores_prob_zip = np.array(sorted(zip(scores, true_prob), key=lambda x: x[1]))
    grouped_scores_prob_zip = [list(group) for _, group in groupby(scores_prob_zip, key=lambda x: x[0])]
    true_ece_vals_per_pred_prob = [abs(sum([x[1] for x in group]) / len(group) - group[0][0]) for group in grouped_scores_prob_zip]
    return sum(true_ece_vals_per_pred_prob)/len(grouped_scores_prob_zip)


#TEST

test_scores = np.array([[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]])
test_true_prob = np.array([[0.2, 0.8], [0.1, 0.9], [0.3, 0.7]])

print(round(true_ece(test_scores, test_true_prob), 3))

def calibration_error_summary(scores, labels, n_bins: np.ndarray, round_to=4):
    check_metric_params(scores, labels)

    ece_vals = np.array([])
    fce_vals = np.array([])
    tce_vals_uniform = np.array([])
    tce_vals_pavabc = np.array([])
    ace_vals = np.array([])

    for n_bin in n_bins:
        n_bin = check_bins(n_bin)

        ece_vals = np.append(ece_vals, round(ece(scores, labels, n_bin), round_to))
        fce_vals = np.append(fce_vals, round(fce(scores, labels, n_bin), round_to))
        tce_vals_uniform = np.append(tce_vals_uniform, round(tce(scores, labels, n_bin=n_bin, strategy="uniform"), round_to))
        tce_vals_pavabc = np.append(tce_vals_pavabc, round(tce(scores, labels, n_bin=n_bin, strategy="pavabc"), round_to))
        ace_vals = np.append(ace_vals, round(ace(scores, labels, n_ranges=n_bin), round_to))

    ksce_val = round(ksce(scores, labels), round_to)
    balance_score_val = round(balance_score(scores, labels), round_to)

    return np.array([ece_vals, fce_vals, tce_vals_uniform, tce_vals_pavabc, ace_vals]),  balance_score_val, ksce_val


def print_calibration_error_summary_table(scores, labels, true_prob, n_bins: np.ndarray, round_to=4):
    true_ece_val = true_ece(scores, true_prob)
    binned_metrics, balance_score_val, ksce_val = calibration_error_summary(scores, labels, n_bins, round_to=round_to)

    not_binned_metrics_data = {
        "true ece": [true_ece_val],
        "balance score": [balance_score_val],
        "balance score ABS": [abs(balance_score_val)],
        "ksce": [ksce_val]
    }

    binned_metrics_data = {
        "number of bins": n_bins,
        "ece": binned_metrics[0],
        "fce": binned_metrics[1],
        "tce (uniform, normalized)": binned_metrics[2] / 100,
        "tce (pava-bc, normalized)": binned_metrics[3] / 100,
        "ace": binned_metrics[4],
    }

    not_binned_df = pd.DataFrame(not_binned_metrics_data)
    binned_df = pd.DataFrame(binned_metrics_data)
    return not_binned_df, binned_df


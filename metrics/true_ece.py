import pandas as pd

from metrics.ace import ace
from metrics.balance_score import balance_score
from metrics.ece import ece
from metrics.fce import fce
from metrics.ksce import ksce
from metrics.tce import tce
from utilities.utils import *


def true_ece(scores, true_prob):
    # Check input validity (assuming these functions are already defined)
    check_scores(scores)
    check_scores(true_prob)
    check_shapes(scores, true_prob)

    # Convert inputs to numpy arrays and extract the relevant columns
    scores = np.array(scores[:, 1])
    true_prob = np.array(true_prob[:, 1])

    # Sort scores and true_prob by scores
    sorted_indices = np.argsort(scores)
    scores = scores[sorted_indices]
    true_prob = true_prob[sorted_indices]

    # Find unique scores and their corresponding indices
    unique_scores, inverse_indices = np.unique(scores, return_inverse=True)

    # Compute the mean true probability for each unique score
    sum_true_prob = np.bincount(inverse_indices, weights=true_prob)
    count_per_score = np.bincount(inverse_indices)
    mean_true_prob = sum_true_prob / count_per_score

    # Calculate the true ECE values per predicted probability
    true_ece_vals_per_pred_prob = np.abs(mean_true_prob - unique_scores)

    # Compute the final true ECE value
    return np.mean(true_ece_vals_per_pred_prob)

#TEST
test_scores = np.array([[0.2, 0.8], [0.6, 0.4], [0.2, 0.8]])
test_true_prob = np.array([[0.2, 0.8], [0.5, 0.5], [1, 0]])

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

        ece_vals = np.append(ece_vals, np.round(ece(scores, labels, n_bin), round_to))
        fce_vals = np.append(fce_vals, np.round(fce(scores, labels, n_bin), round_to))
        tce_vals_uniform = np.append(tce_vals_uniform, np.round(tce(scores, labels, n_bin=n_bin, strategy="uniform"), round_to))
        tce_vals_pavabc = np.append(tce_vals_pavabc, np.round(tce(scores, labels, n_bin=n_bin, strategy="pavabc"), round_to))
        ace_vals = np.append(ace_vals, np.round(ace(scores, labels, n_ranges=n_bin), round_to))

    ksce_val = np.round(ksce(scores, labels), round_to)
    balance_score_val = np.round(balance_score(scores, labels), round_to)

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


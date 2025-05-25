import pandas as pd

from src.metrics.ace import ace
from src.metrics.balance_score import balance_score
from src.metrics.ece import ece
from src.metrics.fce import fce
from src.metrics.ksce import ksce
from src.metrics.tce import tce
from src.utilities.utils import *


def true_ece(y_pred, p_true):
    check_y_pred(y_pred)
    check_y_pred(p_true)
    check_shapes(y_pred, p_true)

    # Convert inputs to numpy arrays and extract the relevant columns
    scores = np.array(y_pred[:, 1])
    p_true = np.array(p_true[:, 1])

    # Sort scores and p_true by scores
    sorted_indices = np.argsort(scores)
    scores = scores[sorted_indices]
    p_true = p_true[sorted_indices]

    # Find unique scores and their corresponding indices
    unique_scores, inverse_indices = np.unique(scores, return_inverse=True)

    # Compute the mean true probability for each unique score
    sum_p_true = np.bincount(inverse_indices, weights=p_true)
    count_per_score = np.bincount(inverse_indices)
    mean_p_true = sum_p_true / count_per_score

    # Calculate the true ECE values per predicted probability
    true_ece_vals_per_p_pred = np.abs(mean_p_true - unique_scores)

    # Compute the final true ECE value
    return np.mean(true_ece_vals_per_p_pred)


def true_ece_binned(p_pred, p_true, bin_boundaries, return_metadata=False):
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    p_pred = np.max(p_pred, axis=-1)
    p_true = np.max(p_true, axis=-1)

    calibration_error = 0.0
    bin_counts = []
    bin_true_prob = []
    bin_confidences = []
    bin_probs = []
    bin_errors = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        samples_in_bin = (p_pred > bin_lower) & (p_pred <= bin_upper)
        relative_samples_in_bin = samples_in_bin.mean()

        bin_counts.append(np.sum(samples_in_bin))
        bin_probs.append(relative_samples_in_bin)

        if relative_samples_in_bin > 0:
            p_true_in_bin = p_true[samples_in_bin].mean()
            avg_confidence_in_bin = p_pred[samples_in_bin].mean()

            bin_error = np.abs(avg_confidence_in_bin - p_true_in_bin) * relative_samples_in_bin
            calibration_error += bin_error

            bin_true_prob.append(p_true_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_errors.append(bin_error)
        else:
            bin_true_prob.append(0)
            bin_confidences.append(0)
            bin_errors.append(0)

    if return_metadata:
        return calibration_error, bin_counts, bin_errors, bin_true_prob, bin_confidences, bin_probs

    return calibration_error, bin_counts


def calibration_error_summary(scores, y_true, n_bins: np.ndarray, round_to=4):
    check_metric_params(scores, y_true)

    ece_vals = np.array([])
    fce_vals = np.array([])
    tce_vals_uniform = np.array([])
    tce_vals_pavabc = np.array([])
    ace_vals = np.array([])

    for n_bin in n_bins:
        n_bin = check_bins(n_bin)

        ece_vals = np.append(ece_vals, np.round(ece(scores, y_true, n_bin), round_to))
        fce_vals = np.append(fce_vals, np.round(fce(scores, y_true, n_bin), round_to))
        tce_vals_uniform = np.append(tce_vals_uniform, np.round(tce(scores, y_true, n_bin=n_bin, strategy="uniform"), round_to))
        tce_vals_pavabc = np.append(tce_vals_pavabc, np.round(tce(scores, y_true, n_bin=n_bin, strategy="pavabc"), round_to))
        ace_vals = np.append(ace_vals, np.round(ace(scores, y_true, n_ranges=n_bin), round_to))

    ksce_val = np.round(ksce(scores, y_true), round_to)
    balance_score_val = np.round(balance_score(scores, y_true), round_to)

    return np.array([ece_vals, fce_vals, tce_vals_uniform, tce_vals_pavabc, ace_vals]),  balance_score_val, ksce_val


def print_calibration_error_summary_table(scores, labels, p_true, n_bins: np.ndarray, round_to=4):
    true_ece_val = true_ece(scores, p_true)
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


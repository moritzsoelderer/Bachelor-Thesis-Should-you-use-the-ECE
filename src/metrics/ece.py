from src.utilities.utils import *


def ece(p_pred: np.ndarray, y_true: np.ndarray, n_bins: int = None) -> np.float32:
    # p_pred should be an n-D array consisting of predicted probabilities for every class
    # y_true should be a 1-D array consisting of the true class labels
    # n_bins should be an Integer

    n_bins = check_binned_metric_params(p_pred, y_true, n_bins)

    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    return calibration_error(p_pred, y_true, bin_boundaries)


def calibration_error(p_pred: np.ndarray, y_true: np.ndarray, bin_boundaries: np.ndarray):
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(p_pred, axis=1)
    predicted_label = np.argmax(p_pred, axis=1)
    accuracies = predicted_label == y_true

    calibration_error = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return calibration_error

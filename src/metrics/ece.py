from src.utilities.utils import *


def ece(pred_prob: np.ndarray, true_labels: np.ndarray, n_bins: int = None) -> np.float32:
    # pred_prob should be an n-D array consisting of predicted probabilities for every class
    # true_labels should be a 1-D array consisting of the true class labels
    # n_bins should be an Integer

    n_bins = check_binned_metric_params(pred_prob, true_labels, n_bins)

    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    return calibration_error(pred_prob, true_labels, bin_boundaries)


def calibration_error(pred_prob: np.ndarray, true_labels: np.ndarray, bin_boundaries: np.ndarray):
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = pred_prob[:, 1]
    predicted_label = (confidences > 0.5).astype(int)
    accuracies = predicted_label == true_labels

    calibration_error = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        samples_in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        relative_samples_in_bin = samples_in_bin.mean()

        if relative_samples_in_bin > 0:
            accuracy_in_bin = accuracies[samples_in_bin].mean()
            avg_confidence_in_bin = confidences[samples_in_bin].mean()

            calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * relative_samples_in_bin

    return calibration_error
    
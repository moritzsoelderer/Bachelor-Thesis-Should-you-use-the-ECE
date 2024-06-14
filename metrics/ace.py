import numpy as np

from metrics.ece import calibration_error


def ace(pred_prob: np.ndarray, true_labels: np.ndarray, n_ranges: np.int64):

    bin_boundaries = adaptive_bin_boundaries(pred_prob, n_ranges)

    return calibration_error(pred_prob, true_labels, bin_boundaries)


def adaptive_bin_boundaries(pred_prob: np.ndarray, n_ranges: np.int64) -> np.ndarray:
    confidences = np.max(pred_prob, axis=1)
    split_arrays = np.array_split(sorted(confidences), n_ranges)
    bin_boundaries = [0.0]
    bin_boundaries = (bin_boundaries +
                      [round((split_arrays[i][-1] + split_arrays[i+1][0])/2.0, 2)
                       for i in range(len(split_arrays) - 1)]
                      )
    bin_boundaries = bin_boundaries + [1.0]
    return np.array(bin_boundaries)


if __name__ == '__main__':
    pred_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 0.7, 0.6])
    true_labels = np.array([0, 1, 0, 1, 1, 0, 0, 0])

    ace(pred_prob, true_labels)

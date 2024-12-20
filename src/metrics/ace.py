import numpy as np

from src.metrics.ece import calibration_error


def ace(pred_prob: np.ndarray, true_labels: np.ndarray, n_ranges: int) -> float:
    # Calculate adaptive bin boundaries in a performant way
    bin_boundaries = adaptive_bin_boundaries(pred_prob, n_ranges)
    return calibration_error(pred_prob, true_labels, bin_boundaries)


def adaptive_bin_boundaries(pred_prob: np.ndarray, n_ranges: int) -> np.ndarray:
    # Calculate the maximum probabilities for each prediction
    confidences = np.max(pred_prob, axis=1)

    # Calculate the bin edges using quantiles instead of sorting
    bin_edges = np.quantile(confidences, np.linspace(0, 1, n_ranges + 1))

    # Create the bin boundaries: first and last are 0.0 and 1.0
    bin_boundaries = np.concatenate(([0.0], 0.5 * (bin_edges[:-1] + bin_edges[1:]), [1.0]))

    return bin_boundaries




if __name__ == '__main__':
    pred_prob = np.array([[0.1, 0.9],
         [0.2, 0.8],
         [0.3, 0.7],
         [0.4, 0.6],
         [0.5, 0.5],
         [1., 0.],
         [0.7, 0.3],
         [0.6, 0.4]])
    true_labels = np.array([0, 1, 0, 1, 1, 0, 0, 0])

    ace_val = ace(pred_prob, true_labels, 8)
    print(ace_val)

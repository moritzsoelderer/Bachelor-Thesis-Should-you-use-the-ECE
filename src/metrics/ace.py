import numpy as np

from src.metrics.ece import calibration_error


def ace(p_pred: np.ndarray, y_true: np.ndarray, n_ranges: int) -> float:
    bin_boundaries = adaptive_bin_boundaries(p_pred, n_ranges)
    return calibration_error(p_pred, y_true, bin_boundaries)


def adaptive_bin_boundaries(p_pred: np.ndarray, n_ranges: int) -> np.ndarray:
    confidences = np.max(p_pred, axis=1)

    bin_edges = np.quantile(confidences, np.linspace(0, 1, n_ranges + 1))
    bin_boundaries = np.concatenate(([0.0], 0.5 * (bin_edges[:-1] + bin_edges[1:]), [1.0]))

    return bin_boundaries

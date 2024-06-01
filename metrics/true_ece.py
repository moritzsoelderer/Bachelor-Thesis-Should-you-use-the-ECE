import numpy as np

from metrics.ece import expected_calibration_error

def true_ece(scores, true_prob):
    if not isinstance(scores, np.ndarray):
        raise ValueError("scores should be a numpy array")
    if not isinstance(true_prob, np.ndarray):
        raise ValueError("true_prob should be a numpy array")
    if np.shape(scores) != np.shape(true_prob):
        raise ValueError("shape of scores and true_prob do not match: ", np.shape(scores), np.shape(true_prob))
    if not all(0 <= score <= 1 for score in scores):
        raise ValueError("all scores must be in range [0,1]")
    if not all(0 == prob or prob == 1 for prob in true_prob):
        raise ValueError("all labels must be either 0 or 1")

    return sum(abs(np.array(true_prob) - np.array(true_prob))) / len(true_prob)

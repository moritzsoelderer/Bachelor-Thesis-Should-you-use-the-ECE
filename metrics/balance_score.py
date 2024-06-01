import numpy as np
from numpy import floating
from numpy._typing import _32Bit


def balance_score(scores: np.ndarray, y_true: np.ndarray) -> float:

    # scores should be a 1d numpy array containing probabilities for the positive class (y = 1)
    # y_true should be a 1d numpy array containing the true labels (0 or 1) for each sample

    if not isinstance(scores, np.ndarray):
        raise ValueError("scores should be a numpy array")
    if not isinstance(y_true, np.ndarray):
        raise ValueError("y_true should be a numpy array")
    if np.shape(scores) != np.shape(y_true):
        raise ValueError("shape of scores and true labels do not match: ", np.shape(scores),  np.shape(y_true))
    if not all(0 <= score <= 1 for score in scores):
        raise ValueError("all scores must be in range [0,1]")
    if not all(0 == label or label == 1 for label in y_true):
        raise ValueError("all labels must be either 0 or 1")

    # scoring function
    def scoring_function(p: np.float32, y: np.int64) -> float:
        # p is the probability for the positive class of one sample
        # y is the true label of that sample

        if y == 1:
            if p < .5:
                return -1.0 + p
            else:
                return 1.0 - p
        if y == 0:
            if p < .5:
                return p
            else:
                return -p

    n_samples = len(scores)

    return (1/n_samples) * sum([scoring_function(scores[i], y_true[i]) for i in range(n_samples)])


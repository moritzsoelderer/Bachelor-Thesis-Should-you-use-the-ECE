import numpy as np
from utilities.utils import *


def balance_score(scores: np.ndarray[np.float32], y_true: np.ndarray[np.int64]) -> float:

    # scores should be a 1d numpy array containing probabilities for the positive class (y = 1)
    # y_true should be a 1d numpy array containing the true labels (0 or 1) for each sample
    check_metric_params(scores, y_true)

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


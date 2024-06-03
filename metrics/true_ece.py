import numpy as np

from metrics.ece import expected_calibration_error
from utilities.utils import *


def true_ece(scores, true_prob):
    check_scores(scores)
    check_scores(true_prob)
    check_shapes(scores, true_prob)

    return sum(abs(np.array(true_prob) - np.array(true_prob))) / len(true_prob)

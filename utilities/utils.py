import numpy as np


def check_scores(scores) -> None:
    if scores is None:
        raise ValueError("scores must not be None")
    if not isinstance(scores, np.ndarray):
        raise ValueError("scores should be a numpy array")
    if not all(0 <= s <= 1 for score in scores for s in score):
        raise ValueError("all scores must be in range [0,1]")


def check_labels(labels) -> None:
    if labels is None:
        raise ValueError("labels must not be None")
    if not isinstance(labels, np.ndarray):
        raise ValueError("labels should be a numpy array")
    if not all(0 == label or label == 1 for label in labels):
        raise ValueError("all labels must be either 0 or 1")


def check_bins(n_bins) -> np.uint64:
    if n_bins is None:
        raise ValueError("n_bins must not be None")
    if n_bins <= 0:
        raise RuntimeWarning("n_bins is negative - using absolute value...")
    return abs(n_bins)


def check_shapes(array1: np.ndarray, array2: np.ndarray) -> None:
    if len(array1) != len(array2):
        raise ValueError("array shape mismatch: ", np.shape(array1), np.shape(array2))


def check_metric_params(scores: np.ndarray[np.float32], labels: np.ndarray[np.int64]) -> None:
    check_scores(scores)
    check_labels(labels)
    check_shapes(scores, labels)


def check_binned_metric_params(scores: np.ndarray[np.float32], labels: np.ndarray[np.int64], n_bins) -> np.uint64:
    check_metric_params(scores, labels)
    bins = check_bins(n_bins)
    if n_bins > len(scores):
        raise Warning("n_bins is larger than the number of samples, which results into empty bins")
    return bins


def check_binned_metric_params_probs(scores: np.ndarray[np.float32], true_prob: np.ndarray[np.float32],
                                     n_bins) -> np.uint64:
    check_scores(scores)
    check_scores(true_prob)
    check_shapes(scores, true_prob)
    return check_bins(n_bins)
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt, pyplot


def check_y_pred(y_pred) -> None:
    if y_pred is None:
        raise ValueError("y_pred must not be None")
    if not isinstance(y_pred, np.ndarray):
        raise ValueError("y_pred should be a numpy array")
    if not all(0 <= s <= 1 for score in y_pred for s in score):
        raise ValueError("all y_pred must be in range [0,1]")


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


def check_metric_params(y_pred: np.ndarray[np.float32], labels: np.ndarray[np.int64]) -> None:
    check_y_pred(y_pred)
    check_labels(labels)
    check_shapes(y_pred, labels)


def check_binned_metric_params(y_pred: np.ndarray[np.float32], labels: np.ndarray[np.int64], n_bins) -> np.uint64:
    check_metric_params(y_pred, labels)
    bins = check_bins(n_bins)
    if n_bins > len(y_pred):
        raise Warning("n_bins is larger than the number of X, which results into empty bins")
    return bins


def check_binned_metric_params_probs(y_pred: np.ndarray[np.float32], p_true: np.ndarray[np.float32],
                                     n_bins) -> np.uint64:
    check_y_pred(y_pred)
    check_y_pred(p_true)
    check_shapes(y_pred, p_true)
    return check_bins(n_bins)


def sample_uniformly_within_bounds(locs: np.ndarray, scales: np.ndarray, size: int, seed=1) -> np.ndarray:
    locs = np.array(locs)
    scales = np.array(scales)

    if locs.shape != scales.shape:
        raise ValueError("There must be just as much locations as scales")

    return np.array([tf.random.uniform(shape=(size,), minval=loc, maxval=scales[index], seed=seed) for index, loc in enumerate(locs)]).T


def plot_samples_probability_mask(
        X: np.ndarray, probs: np.ndarray,
        colorbar_label: str, title: str, xlabel: str = 'feature 0', ylabel: str = 'feature 1',
        show: bool = True, save_path: str = None
) -> pyplot.figure:
    plt.scatter(X[:, 0], X[:, 1], c=probs[:, 1], cmap='coolwarm_r', s=1)

    plt.colorbar(label=colorbar_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path)
    if show is True:
        plt.show(block=False)

    return plt


import numpy as np
from sklearn.metrics import accuracy_score

from src.metrics.ece import ece


def approximate_optimal_ece(
        estimator,
        predict_proba_fun,
        test_sample: np.ndarray,
        y_true_test: np.ndarray,
        plateau_threshold: float,
        plateau_steps: int,
        min_sample_size: int,
        steps: int,
        n_bins: int,
):
    # init variables
    is_plateau_count = 0
    first_plateau_sample_size = None
    delta = np.finfo(np.float64).max
    current_sample_size = min_sample_size
    test_sample_size = test_sample.shape[0]

    iterations = 0
    current_sample = test_sample[:current_sample_size]
    current_labels = y_true_test[:current_sample_size]
    p_pred = predict_proba_fun(estimator, current_sample)
    pred_labels = (np.array([p[1] for p in p_pred]) >= 0.5).astype(int)
    first_plateau_ece_value = ece(p_pred, current_labels, n_bins=n_bins)
    min_sample_size_ece = first_plateau_ece_value

    ece_values = np.array([first_plateau_ece_value])
    sample_sizes = np.array([current_sample_size])
    accuracies = np.array([accuracy_score(current_labels, pred_labels)])
    while is_plateau_count < plateau_steps:
        current_sample_size += steps

        if current_sample_size > test_sample_size:
            raise Exception(f"Did not converge after {iterations} iterations, min. delta = {delta}")

        iterations += 1
        current_sample = test_sample[:current_sample_size]
        current_labels = y_true_test[:current_sample_size]
        p_pred = predict_proba_fun(estimator, current_sample)
        pred_labels = (np.array([p[1] for p in p_pred]) >= 0.5).astype(int)

        ece_value = ece(p_pred, current_labels, n_bins=n_bins)
        ece_values = np.append(ece_values, ece_value)
        sample_sizes = np.append(sample_sizes, current_sample_size)
        accuracy = accuracy_score(current_labels, pred_labels)
        accuracies = np.append(accuracies, accuracy)

        delta = np.abs(first_plateau_ece_value - ece_value)

        if delta <= plateau_threshold:
            if first_plateau_sample_size is None:
                first_plateau_sample_size = current_sample_size
            is_plateau_count += 1
        else:
            first_plateau_ece_value = ece_value
            first_plateau_sample_size = None
            is_plateau_count = 0

    # gradient
    dx = np.abs(min_sample_size - first_plateau_sample_size)
    dy = np.abs(min_sample_size_ece - first_plateau_ece_value)

    # factors
    ece_factor = np.sqrt(1 - (dy / min_sample_size_ece - 1) ** 2)  # modified third quadrant of unit circle
    sample_size_factor = 1 / (1 + np.exp(-(16 * first_plateau_sample_size / 10000 - 8)))  # modified sigmoid
    accuracy = np.mean(accuracies)
    accuracy_factor = 0.9 * accuracy if accuracy >= 0.9 else accuracy ** 16

    # optimal sample size
    optimal_sample_size = min_sample_size + int(ece_factor * dx * sample_size_factor * accuracy_factor)

    # optimal ece value
    optimal_sample = test_sample[:optimal_sample_size]
    optimal_labels = y_true_test[:optimal_sample_size]
    p_pred = predict_proba_fun(estimator, optimal_sample)
    optimal_ece_value = ece(p_pred, optimal_labels, n_bins=n_bins)

    return optimal_ece_value, optimal_sample_size, iterations, ece_values, sample_sizes, first_plateau_sample_size, first_plateau_ece_value,


def find_first_plateau(sample_sizes, ece_values, plateau_threshold, plateau_steps):
    is_plateau_count = 0
    first_plateau_sample_size = None
    delta = np.finfo(np.float64).max
    max_index = len(sample_sizes) - 1
    index = 1
    first_plateau_ece_value = ece_values[0]
    while is_plateau_count < plateau_steps:
        if index > max_index:
            raise Exception(f"Did not converge after {index - 1} iterations, min. delta = {delta}")

        current_sample_size = sample_sizes[index]
        ece_value = ece_values[index]
        delta = np.abs(first_plateau_ece_value - ece_value)

        if delta <= plateau_threshold:
            if first_plateau_sample_size is None:
                first_plateau_sample_size = current_sample_size
            is_plateau_count += 1
        else:
            first_plateau_ece_value = ece_value
            first_plateau_sample_size = None
            is_plateau_count = 0

        index += 1

    return first_plateau_sample_size, first_plateau_ece_value, index


def approximate_optimal_ece_variable_threshold(
        min_plateau_threshold: float,
        max_plateau_threshold: float,
        plateau_steps: int,
        sample_sizes: np.ndarray,
        ece_values: np.ndarray,
        accuracies: float
):
    optimal_result = approximate_optimal_ece_sample_size(
        max_plateau_threshold, plateau_steps, sample_sizes, ece_values, accuracies
    )

    current_threshold = max_plateau_threshold - 0.0001
    while current_threshold >= min_plateau_threshold:
        try:
            optimal_result = approximate_optimal_ece_sample_size(
                current_threshold, plateau_steps, sample_sizes, ece_values, accuracies
            )
        except Exception:
            break
        current_threshold -= 0.0001
    return optimal_result


def approximate_optimal_ece_sample_size(
        plateau_threshold: float,
        plateau_steps: int,
        sample_sizes: np.ndarray,
        ece_values: np.ndarray,
        accuracies: float
):
    min_sample_size = sample_sizes[0]
    min_sample_size_ece = ece_values[0]
    first_plateau_sample_size, first_plateau_ece_value, index = find_first_plateau(
        sample_sizes, ece_values, plateau_threshold, plateau_steps
    )

    # gradient
    dx = np.abs(min_sample_size - first_plateau_sample_size)
    dy = np.abs(min_sample_size_ece - first_plateau_ece_value)
    mean_accuracy = np.mean(accuracies)

    # factors
    ece_factor = np.sqrt(1 - np.abs(dy/min_sample_size_ece - 1) ** 1.75)  # modified third quadrant of unit circle
    sample_size_factor = 1 / (1 + np.exp(-(16 * first_plateau_sample_size/10000 - 8)))  # modified sigmoid

    if mean_accuracy >= 0.85:
        accuracy_factor = mean_accuracy
    elif mean_accuracy >= 0.75:
        accuracy_factor = mean_accuracy ** 2
    else:
        accuracy_factor = mean_accuracy ** 4

    optimal_sample_size = min_sample_size + int(ece_factor * dx * sample_size_factor * accuracy_factor)

    return optimal_sample_size, index - 1, first_plateau_sample_size, first_plateau_ece_value
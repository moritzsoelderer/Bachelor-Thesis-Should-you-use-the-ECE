import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import accuracy_score


def accuracy_rejection(y_true: np.ndarray, p_pred: np.ndarray, steps: int = None, strategy=None) -> [np.ndarray]:
    assert y_true.shape[0] == p_pred.shape[0], "true labels and predicted probabilities must have the same length"
    assert steps is not None, "steps must not be None"

    positive_p_pred = p_pred[:, 1]

    if strategy == "entropy":
        entropies = entropy(p_pred, axis=-1)
        indices = np.argsort(-entropies)
    elif strategy == "predictions":
        min_distances = np.minimum(positive_p_pred, 1-positive_p_pred)
        indices = np.argsort(-min_distances)
    else:
        raise ValueError(f"Unknown strategy {strategy} - must be one of ['entropy', 'predictions']")

    y_true_sorted = y_true[indices]
    y_pred_1_sorted = (positive_p_pred[indices] >= 0.5).astype(np.int64)

    rejection_steps = np.linspace(0, np.shape(p_pred)[0], steps, endpoint=False, dtype=np.int64)
    rejection_rates = np.linspace(0, 1, steps)

    rejection_accuracies = np.array(
        [accuracy_score(y_true_sorted[r:], y_pred_1_sorted[r:]) for r in rejection_steps]
    )

    return rejection_accuracies, rejection_rates


def plot_accuracy_rejection(
        rejection_accuracies: np.ndarray,
        rejection_rates: np.ndarray,
        title: str = "Accuracy Rejection Curve (ARC)"
) -> plt.plot:
    plot = plt.plot(rejection_rates, rejection_accuracies, color='red')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel("Rejection Rates")
    plt.ylabel("Accuracy")
    plt.tight_layout(pad=1.12)
    plt.show(block=False)

    return plot

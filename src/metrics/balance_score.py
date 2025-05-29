from src.utilities.utils import *


def balance_score(y_pred: np.ndarray, y_true: np.ndarray):
    check_metric_params(y_pred, y_true)
    p = y_pred[:, 1].astype(np.float32)
    y = y_true.astype(np.int64)

    balance_scores = np.zeros_like(p)

    balance_scores[y == 1] = np.where(p[y == 1] < 0.5, -1.0 + p[y == 1], 1.0 - p[y == 1])
    balance_scores[y == 0] = np.where(p[y == 0] < 0.5, p[y == 0], -p[y == 0])

    return np.mean(balance_scores)


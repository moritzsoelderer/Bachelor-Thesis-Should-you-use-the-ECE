from src.utilities.utils import *


def balance_score(scores: np.ndarray, y_true: np.ndarray):
    # Validate input
    check_metric_params(scores, y_true)

    # Ensure scores is a 2D array and extract probabilities for the positive class
    scores = scores[:, 1].astype(np.float32)

    # Calculate scoring function using vectorized operations
    p = scores
    y = y_true.astype(np.int64)

    # Initialize an array for the scores
    balance_scores = np.zeros_like(p)

    # Positive class scoring
    balance_scores[y == 1] = np.where(p[y == 1] < 0.5, -1.0 + p[y == 1], 1.0 - p[y == 1])

    # Negative class scoring
    balance_scores[y == 0] = np.where(p[y == 0] < 0.5, p[y == 0], -p[y == 0])

    # Calculate the average score
    return np.mean(balance_scores)


if __name__ == '__main__':
    pred_prob = np.array([[0.1, 0.9],
         [0.2, 0.8],
         [0.3, 0.7],
         [0.4, 0.6],
         [0.5, 0.5],
         [1., 0.],
         [0.7, 0.3],
         [0.6, 0.4]])
    true_labels = np.array([0, 1, 0, 1, 1, 0, 0, 0])

    balance_score_val = balance_score(pred_prob, true_labels)
    print(balance_score_val)

    #0.02500000223517418


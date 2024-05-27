import numpy as np

import metrics.ece

y_true = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 0], dtype=np.int64)
y_pred = np.array(
    [[1, 0], [0, 1], [0, 1], [.4, .6], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [1, 0]],
    dtype=np.float32)
y_pred_all_wrong = np.array(
    [[0, 1], [1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]],
    dtype=np.float32)

print(y_true)
print(y_pred)

n_bins = 10

medium_ece_val = metrics.ece.expected_calibration_error(pred_prob=y_pred, true_labels=y_true, n_bins=n_bins)
print("medium_ece_val: ", medium_ece_val)
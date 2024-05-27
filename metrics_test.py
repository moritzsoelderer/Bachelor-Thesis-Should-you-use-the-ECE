import numpy as np

from metrics.ece import expected_calibration_error
from metrics.fce import fuzzy_calibration_error

n_samples = 10000

y_true = np.array(np.random.randint(0, 2, size=n_samples),  dtype=np.int64)
print(len(y_true))

y_pred = np.array(
    [[p, 1.0 - p] for label in y_true for p in [round(np.random.rand(), 3)]], dtype=np.float32
)
y_pred_all_correct = np.array(
    [[not label, label] for label in y_true], dtype=np.float32
)
y_pred_all_wrong = np.array(
    [[label, not label] for label in y_true], dtype=np.float32
)


print(len(y_true))
print(len(y_pred))
print(len(y_pred_all_wrong))

n_bins = 10

bins = [10, 50, 100, 1000]
for bin in bins:
    fce_vals = fuzzy_calibration_error(y_true, y_pred, n_bins=bin)
    print("fce_vals: ", fce_vals)

    medium_ece_val = expected_calibration_error(pred_prob=y_pred, true_labels=y_true, n_bins=bin)
    print("medium_ece_val: ", medium_ece_val)

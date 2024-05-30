import numpy as np

from metrics.ece import expected_calibration_error
from metrics.fce import fuzzy_calibration_error
from metrics.tce import tce
from metrics.tce import tce_ttest
from metrics.ksce import ksce

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

# y_pred = y_pred_all_wrong
y_pred = y_pred[:, 1]

print(y_pred)

ksce_val = ksce(scores=y_pred, labels=y_true)
print("ksce_val: ", ksce_val)

exit()

tce_val = tce(preds=y_pred, labels=y_true, n_min=n_bins, n_max=n_bins, n_bin=n_bins, strategy="uniform")
print("tce_val: ", tce_val)

tce_ttest_val = tce_ttest(preds=y_pred, labels=y_true, n_min=n_bins, n_max=n_bins, n_bin=n_bins, strategy="uniform")
print("tce_ttest_val: ", tce_ttest_val)

bins = [10, 50, 100, 1000]
for bin in bins:
    fce_vals = fuzzy_calibration_error(y_true, y_pred, n_bins=bin)
    print("fce_vals: ", fce_vals)

    medium_ece_val = expected_calibration_error(pred_prob=y_pred, true_labels=y_true, n_bins=bin)
    print("medium_ece_val: ", medium_ece_val)

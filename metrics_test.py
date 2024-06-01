import numpy as np

from metrics.ece import expected_calibration_error
from metrics.fce import fuzzy_calibration_error
from metrics.ksce import ksce
from metrics.tce import tce
from metrics.tce import tce_ttest
from metrics.balance_score import balance_score

n_samples = 10000

y_true = np.array(np.random.randint(0, 2, size=n_samples),  dtype=np.int64)

y_pred = np.array(
    [[p, 1.0 - p] for label in y_true for p in [round(np.random.rand(), 3)]], dtype=np.float32
)
y_pred_all_correct = np.array(
    [[not label, label] for label in y_true], dtype=np.float32
)
y_pred_all_wrong = np.array(
    [[label, not label] for label in y_true], dtype=np.float32
)

y_pred = y_pred_all_wrong

# for tce, tce_ttest and ksce, rest supports 2 dims
y_pred_positive_class = np.array([elem[1] for elem in y_pred])

ksce_val = ksce(y_true, y_pred_positive_class)
print("ksce_val: ", ksce_val)

bins = [10]
for bin in bins:
    print("--------------------------------------------------------------------")
    print("BINS: ", bin)

    ece_val = expected_calibration_error(pred_prob=y_pred, true_labels=y_true, n_bins=bin)
    print("ece_val: ", ece_val)

    balance_score_val = balance_score(scores=y_pred_positive_class, y_true=y_true)
    print("balance_score_val: ", balance_score_val)

    _, fce_val = fuzzy_calibration_error(y_true, y_pred, n_bins=bin)
    print("fce_val ", fce_val)

    tce_val = tce(preds=y_pred_positive_class, labels=y_true, n_min=bin, n_max=bin, n_bin=bin, strategy="uniform")
    print("tce_val: ", tce_val)

    tce_ttest_val = tce_ttest(preds=y_pred_positive_class, labels=y_true, n_min=bin, n_max=bin, n_bin=bin, strategy="uniform")
    print("tce_ttest_val: ", tce_ttest_val)

import numpy as np
from matplotlib import pyplot as plt

from src.metrics.ace import ace
from src.metrics.ece import ece
from src.metrics.fce import fce
from src.metrics.ksce import ksce
from src.metrics.tce import tce

n_samples = 100

y_true = np.array([0] * n_samples)

y_pred = np.array(
    [[p, 1.0 - p] for label in y_true for p in [round(np.random.rand(), 3)]], dtype=np.float32
)

y_pred_all_correct = np.array(
    [[not label, label] for label in y_true], dtype=np.float32
)
y_pred_all_wrong = np.array(
    [[label, not label] for label in y_true], dtype=np.float32
)


n_ece_bins = 10
n_ace_ranges = 10
n_tce_bins = 10
n_fce_bins = 10

percentages = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ece_vals = []
ace_vals = []
ksce_vals = []
tce_vals = []
fce_vals = []

for percentage in percentages:
    deviation = percentage * y_pred_all_correct[:, 0]
    y_pred_percentage_deviated = np.column_stack([y_pred_all_correct[:, 0] - deviation, y_pred_all_correct[:, 1] + deviation])

    ece_vals = np.append(ece_vals, [ece(y_pred_percentage_deviated, y_true, n_ece_bins)])
    ace_vals = np.append(ace_vals, [ace(y_pred_percentage_deviated, y_true, n_ace_ranges)])
    ksce_vals = np.append(ksce_vals, [ksce(y_pred_percentage_deviated, y_true)])
    tce_vals = np.append(tce_vals, [tce(y_pred_percentage_deviated, y_true, 0.05, "uniform", n_tce_bins, n_tce_bins, n_tce_bins)])
    fce_vals = np.append(fce_vals, [fce(y_pred_percentage_deviated, y_true, n_fce_bins)])

plt.scatter(percentages, ece_vals)
plt.xlabel("percentage of predictions deviation")
plt.ylabel("ECE " + str(n_ece_bins))
plt.show(block=False)
print("ECE values: ", ece_vals)

plt.scatter(percentages, ace_vals)
plt.xlabel("percentage of predictions deviation")
plt.ylabel("ACE " + str(n_ace_ranges))
plt.show(block=False)
print("ACE values: ", ace_vals)

plt.scatter(percentages, ksce_vals)
plt.xlabel("percentage of predictions deviation")
plt.ylabel("KSCE")
plt.show(block=False)
print("KSCE values: ", ksce_vals)

plt.scatter(percentages, tce_vals)
plt.xlabel("percentage of predictions deviation")
plt.ylabel("TCE " + str(n_tce_bins))
plt.show(block=False)
print("TCE values: ", tce_vals)

plt.scatter(percentages, fce_vals)
plt.xlabel("percentage of predictions deviation")
plt.ylabel("FCE " + str(n_fce_bins))
plt.show(block=False)
print("FCE values: ", fce_vals)



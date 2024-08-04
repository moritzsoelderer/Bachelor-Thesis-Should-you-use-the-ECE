import numpy as np
from matplotlib import pyplot as plt

from metrics.ace import ace
from metrics.ece import ece
from metrics.fce import fce
from metrics.ksce import ksce
from metrics.tce import tce
from metrics.tce import tce_ttest
from metrics.balance_score import balance_score

n_samples = 100

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

y_pred_75_percent_correct = np.append(
    y_pred_all_correct[:int(n_samples*0.75)] , y_pred_all_wrong[int(n_samples*0.75):], axis=0
)

y_pred_50_percent_correct = np.append(
    y_pred_all_correct[:int(n_samples*0.5)], y_pred_all_wrong[int(n_samples*0.5):], axis=0
)

y_pred_25_percent_correct = np.append(
    y_pred_all_correct[:int(n_samples*0.25)] , y_pred_all_wrong[int(n_samples*0.25):], axis=0
)

n_precentages = 100
percentages = [i/float(n_precentages) for i in range(n_precentages)]
ece_vals = []
ace_vals = []
ksce_vals = []
tce_vals = []

y_true_ksce = np.array(list(filter(lambda a: a == 1, y_true)))
y_pred_ksce_all_correct = np.array([y_pred_all_correct[i] for i in range(len(y_pred_all_correct)) if y_true[i] == 1])
y_pred_ksce_all_wrong = np.array([y_pred_all_wrong[i] for i in range(len(y_pred_all_wrong)) if y_true[i] == 1])

y_pred_ksce_all_wrong = y_pred_all_wrong
y_pred_ksce_all_correct = y_pred_all_correct
y_true_ksce = y_true

n_samples_ksce = len(y_pred_ksce_all_correct) + len(y_pred_ksce_all_wrong)
n_samples_ksce = n_samples

for percentage in percentages:
    y_pred_percentage_correct = np.append(
        y_pred_ksce_all_correct[:int(n_samples_ksce*(1-percentage))], y_pred_ksce_all_wrong[int(n_samples_ksce*(1-percentage)):], axis=0
    )

    ece_vals = np.append(ece_vals, [ece(y_pred_percentage_correct, y_true_ksce, 10)])
    ace_vals = np.append(ace_vals, [ace(y_pred_percentage_correct, y_true_ksce, 10)])
    ksce_vals = np.append(ksce_vals, [ksce(y_pred_percentage_correct, y_true_ksce)])
    tce_vals = np.append(tce_vals, [tce(y_pred_percentage_correct, y_true_ksce)])


plt.scatter(percentages, ece_vals)
plt.xlabel("percentage of incorrect predictions")
plt.ylabel("ECE")
plt.show()
print("ECE values: ", ece_vals)

plt.scatter(percentages, ace_vals)
plt.xlabel("percentage of incorrect predictions")
plt.ylabel("ACE")
plt.show()
print("ACE values: ", ace_vals)

plt.scatter(percentages, ksce_vals)
plt.xlabel("percentage of incorrect predictions")
plt.ylabel("KSCE")
plt.show()
print("KSCE values: ", ksce_vals)

plt.scatter(percentages, tce_vals)
plt.xlabel("percentage of incorrect predictions")
plt.ylabel("TCE")
plt.show()
print("TCE values: ", tce_vals)



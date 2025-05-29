import pickle

import numpy as np
from matplotlib import pyplot as plt


"""
This script is used to retrieve data regarding ECE from existing experiments
"""

eces = []
filenames = [
"../varying_test_sample_size_dataset_family/data/Gummy Worm Dataset__SVM__Gummy Worm Dataset Family__AbsoluteValues__20250405_032919.pkl",
"../varying_test_sample_size_dataset_family/data/Gummy Worm Dataset__Neural Network__Gummy Worm Dataset Family__AbsoluteValues__20250311_015848.pkl",
"../varying_test_sample_size_dataset_family/data/Gummy Worm Dataset__Logistic Regression__Gummy Worm Dataset Family__AbsoluteValues__20250311_015848.pkl",
"../varying_test_sample_size_dataset_family/data/Gummy Worm Dataset__Random Forest__Gummy Worm Dataset Family__AbsoluteValues__20250311_015848.pkl",
"../varying_test_sample_size_dataset_family/data/Exclamation Mark Dataset__SVM__Exclamation Mark Dataset Family__AbsoluteValues__20250408_001836.pkl",
"../varying_test_sample_size_dataset_family/data/Exclamation Mark Dataset__Neural Network__Exclamation Mark Dataset Family__AbsoluteValues__20250408_234908.pkl",
"../varying_test_sample_size_dataset_family/data/Exclamation Mark Dataset__Logistic Regression__Exclamation Mark Dataset Family__AbsoluteValues__20250408_234908.pkl",
"../varying_test_sample_size_dataset_family/data/Exclamation Mark Dataset__Random Forest__Exclamation Mark Dataset Family__AbsoluteValues__20250408_234908.pkl",
]

max_errors = []
errors_10000 = []
for filename in filenames:
    with (open(filename, 'rb') as file):
        print(filename)
        results = pickle.load(file)

        means = results["Means"]
        ece_means = np.array(means["ECE"])
        true_eces = np.array(means["True ECE Dists (Binned - 15 Bins)"])

        eces.append(ece_means)
        max_errors.append(np.max(np.abs(ece_means - true_eces)))
        errors_10000.append(np.abs(ece_means[-1] - true_eces[-1]))


print(eces)
print(max_errors)
print(errors_10000)

x = ["GW - SVM", "GW - NN", "GW - LR", "GW - RF", "EX - SVM", "EX - NN", "EX - LR", "EX - RF"]

plt.bar(x, max_errors, label="max. Error")
plt.bar(x, errors_10000, label="Error at 10000 samples")
plt.xlabel("Experiments")
plt.ylabel("Error")
plt.title("ECE - max. Error and Error at 10000 samples", fontsize=14, fontweight="bold")
plt.show()

print("Average max. Error:", np.mean(max_errors))
print("Average ECE Error 10000 Samples:", np.mean(errors_10000))
print("Average ECE Error 10000 Samples (Relative to max. Error):", np.mean(errors_10000) / np.mean(max_errors))




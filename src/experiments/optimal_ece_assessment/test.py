import pickle

import numpy as np
from matplotlib import pyplot as plt

from src.metrics.primitive_ece_sample_assessment import find_first_plateau


def find_optimal_ece(ece_values: np.array, true_ece_values: np.array) -> int:
    ece_values = np.array(ece_values)
    true_ece_values = np.array(true_ece_values)

    assert ece_values.shape == true_ece_values.shape

    last_diff = true_ece_values[0] - ece_values[0]
    last_sign = np.sign(last_diff)
    min_diff_index = 0

    index = 1
    while index < len(true_ece_values):
        print(index)
        current_diff = true_ece_values[index] - ece_values[index]
        current_sign = np.sign(current_diff)
        if current_sign != last_sign:
            return index
        elif np.abs(current_diff) < np.abs(last_diff):
            last_diff = current_diff
            min_diff_index = index
        index += 1
    return min_diff_index



file_info = {
    "SVM": ['Gummy.Worm.Dataset__SVM__Gummy.Worm.Dataset.Family__AbsoluteValues__20250311_015848',
            'Gummy.Worm.Dataset__Logistic.Regression__Gummy.Worm.Dataset.Family__AbsoluteValues__20250311_015848',
            'Gummy.Worm.Dataset__Neural.Network__Gummy.Worm.Dataset.Family__AbsoluteValues__20250311_015848',
            'Gummy.Worm.Dataset__Random.Forest__Gummy.Worm.Dataset.Family__AbsoluteValues__20250311_015848']
}

X = []
Y = np.array([])

for model_name, files in file_info.items():
    for filename in files:
        with (open(f'../varying_test_sample_size_dataset_family/data/{filename}.pkl', 'rb') as file):
            results = pickle.load(file)

            # Extract values
            means = results["Means"]
            std_devs = results["Std Devs"]
            subsample_sizes = np.linspace(100, 20000, 200, dtype=np.int64)
            ece_values = means["ECE"]
            true_ece_values = means["True ECE Dists (Binned - 100 Bins)"]

            min_subsample_size = subsample_sizes[0]
            min_subsample_size_ece = ece_values[0]

            print("Finding first plateau")
            first_plateau_sample_size, first_plateau_ece, _ = find_first_plateau(
                subsample_sizes, ece_values, 0.0005, 30
            )

            dy = np.abs(min_subsample_size_ece - first_plateau_ece)

            # features
            relative_ece_falloff = dy/min_subsample_size_ece
            mean_accuracy = np.mean(means["Accuracy"])
            plateau_sample_size = first_plateau_sample_size

            x = np.array([plateau_sample_size, mean_accuracy, relative_ece_falloff])

            print("Finding optimal sample size")
            y = np.array(subsample_sizes[find_optimal_ece(ece_values, true_ece_values)])

            # Add vectors to samples/labels
            X.append(x)
            Y = np.append(Y, y)



print(X)
X = np.vstack(X)
print(X)
print(X.shape)
print(Y)

fig = plt.figure(figsize=(12, 12), dpi=600)
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X[:, 0], X[:, 1], Y, c=X[:, 2], cmap='viridis')
ax.view_init(elev=90, azim=90)
plt.title("Optimal Subsample Sizes for ECE calculation", fontsize=24, fontweight='bold')

plt.colorbar(sc, ax=ax, label='Relative ECE Falloff')

ax.set_xlabel('first plateau sample size', fontsize=12, labelpad=20, fontweight='bold')
ax.set_ylabel('mean accuracy', fontsize=12, labelpad=20, fontweight='bold')
ax.set_zlabel('Optimal Subsample Sizes', fontsize=12, labelpad=20, fontweight='bold')

plt.show()
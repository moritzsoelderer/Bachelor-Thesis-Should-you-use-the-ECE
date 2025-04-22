import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

import seaborn as sns

from src.metrics.primitive_ece_sample_assessment import approximate_optimal_ece_sample_size, \
    approximate_optimal_ece_variable_threshold


def interpolate_indexed_value_array(indexing_array: np.ndarray, value_array: np.ndarray, searched_for):
    index = 0
    while indexing_array[index] < searched_for:
        index += 1

    interpolation_factor = (searched_for - indexing_array[index - 1]) / (indexing_array[index] - indexing_array[index - 1])
    return value_array[index - 1] + interpolation_factor * (value_array[index] - value_array[index - 1])

def find_optimal_ece(ece_values: np.array, true_ece_values: np.array, sample_sizes: np.array):
    ece_values = np.array(ece_values)
    true_ece_values = np.array(true_ece_values)

    assert ece_values.shape == true_ece_values.shape

    last_diff = ece_values[0] - true_ece_values[0]
    closest_index = 0
    last_sign = np.sign(last_diff)

    index = 1
    while index < len(true_ece_values):
        current_diff = ece_values[index] - true_ece_values[index]
        current_sign = np.sign(current_diff)
        if current_sign != last_sign:
            nominator = ece_values[index] - true_ece_values[index - 1]
            denominator = ece_values[index] - true_ece_values[index - 1] + true_ece_values[index] - ece_values[index - 1]
            interpolation_factor = nominator / denominator
            optimal_ece = ece_values[index - 1] + current_sign * (interpolation_factor * (ece_values[index] - ece_values[index - 1]))
            true_ece = optimal_ece
            optimal_sample_size = sample_sizes[index - 1] + interpolation_factor * (sample_sizes[index] - sample_sizes[index - 1])
            return optimal_ece, true_ece, optimal_sample_size
        elif np.abs(current_diff) < np.abs(last_diff):
            last_diff = current_diff
            closest_index = index
        index += 1
    return ece_values[closest_index], true_ece_values[closest_index], sample_sizes[closest_index]



dir = '../varying_test_sample_size_train_test_split_seeds/data/'
svm_filename = dir + "Gummy Worm Dataset__SVM__20_TrainTestSplits__AbsoluteValues__20250318_220436"
nn_filename = dir + 'Gummy Worm Dataset__Neural Network__20_TrainTestSplits__AbsoluteValues__20250320_221632'
lr_filename = dir + 'Gummy Worm Dataset__Logistic Regression__20_TrainTestSplits__AbsoluteValues__20250321_225921'
rf_filename = dir + 'Gummy Worm Dataset__Random Forest__20_TrainTestSplits__AbsoluteValues__20250321_225921'
svm_filename_exlamation_mark = dir + 'Exclamation Mark Dataset__SVM__20_TrainTestSplits__AbsoluteValues__20250405_200029'
nn_filename_exlamation_mark = dir + 'Exclamation Mark Dataset__Neural Network__20_TrainTestSplits__AbsoluteValues__20250406_143851'
lr_filename_exlamation_mark = dir + 'Exclamation Mark Dataset__Logistic Regression__20_TrainTestSplits__AbsoluteValues__20250406_214431'
rf_filename_exlamation_mark = dir + 'Exclamation Mark Dataset__Random Forest__20_TrainTestSplits__AbsoluteValues__20250408_023033'

dir_family = '../varying_test_sample_size_dataset_family/data/'
svm_filename_family = dir_family + 'Gummy Worm Dataset__SVM__Gummy Worm Dataset Family__AbsoluteValues__20250405_032919'
nn_filename_family = dir_family + 'Gummy Worm Dataset__Neural Network__Gummy Worm Dataset Family__AbsoluteValues__20250311_015848'
lr_filename_family = dir_family + 'Gummy Worm Dataset__Logistic Regression__Gummy Worm Dataset Family__AbsoluteValues__20250311_015848'
rf_filename_family = dir_family + 'Gummy Worm Dataset__Random Forest__Gummy Worm Dataset Family__AbsoluteValues__20250311_015848'
svm_filename_family_exclamation_mark = dir_family + 'Exclamation Mark Dataset__SVM__Exclamation Mark Dataset Family__AbsoluteValues__20250408_001836'
nn_filename_family_exclamation_mark = dir_family + 'Exclamation Mark Dataset__Neural Network__Exclamation Mark Dataset Family__AbsoluteValues__20250408_234908'
lr_filename_family_exclamation_mark = dir_family + 'Exclamation Mark Dataset__Logistic Regression__Exclamation Mark Dataset Family__AbsoluteValues__20250408_234908'
rf_filename_family_exclamation_mark = dir_family + 'Exclamation Mark Dataset__Random Forest__Exclamation Mark Dataset Family__AbsoluteValues__20250408_234908'

file_info = {
    'SVM': [svm_filename, svm_filename_exlamation_mark, svm_filename_family, svm_filename_family_exclamation_mark],
    'Neural Network': [nn_filename, nn_filename_exlamation_mark, nn_filename_family, nn_filename_family_exclamation_mark],
    'Logistic Regression': [lr_filename, lr_filename_family, lr_filename_exlamation_mark, lr_filename_family_exclamation_mark],
    'Random Forest': [rf_filename, rf_filename_family, rf_filename_exlamation_mark, rf_filename_family_exclamation_mark]
}

approx_subsample_sizes = np.array([])
optimal_subsample_sizes = np.array([])

X = []
Y = np.array([])
Y_true = np.array([])
Y_hat = np.array([])
Y_tts = np.array([])

min_plateau_threshold = 0.0004
max_plateau_threshold = 0.001
plateau_steps = 30

for model_name, files in file_info.items():
    for filename in files:
        with (open(f'{filename}.pkl', 'rb') as file):
            print(f"File: {filename}")
            results = pickle.load(file)

            # Extract values
            means = results["Means"]
            std_devs = results["Std Devs"]
            subsample_sizes = np.linspace(100, 20000, 200, dtype=np.int64)
            ece_values = means["ECE"]
            true_ece_values = means["True ECE Dists (Binned - 100 Bins)"]
            accuracies = means["Accuracy"]

            min_subsample_size = subsample_sizes[0]
            min_subsample_size_ece = ece_values[0]

            print("Finding first plateau")

            approx_sample_size, iterations, first_plateau_sample_size, first_plateau_ece = (
                approximate_optimal_ece_variable_threshold(
                    min_plateau_threshold, max_plateau_threshold, plateau_steps, subsample_sizes, ece_values, accuracies
                )
            )

            approx_subsample_sizes = np.append(approx_subsample_sizes, approx_sample_size)

            dy = np.abs(min_subsample_size_ece - first_plateau_ece)

            # features
            relative_ece_falloff = dy/min_subsample_size_ece
            mean_accuracy = np.mean(accuracies)
            plateau_sample_size = first_plateau_sample_size

            x = np.array([plateau_sample_size, mean_accuracy, relative_ece_falloff])

            print("Finding optimal sample size")
            optimal_ece, true_ece, optimal_sample_size = find_optimal_ece(ece_values, true_ece_values, subsample_sizes)
            approx_ece = interpolate_indexed_value_array(subsample_sizes, ece_values, approx_sample_size)
            y = np.array([optimal_ece])
            y_true = np.array([true_ece])
            y_hat = np.array([approx_ece])
            y_tts = np.array([ece_values[49]])

            optimal_subsample_sizes = np.append(optimal_subsample_sizes, optimal_sample_size)

            # Add vectors to samples/labels
            X.append(x)
            Y = np.append(Y, y)
            Y_true = np.append(Y_true, y_true)
            Y_hat = np.append(Y_hat, y_hat)
            Y_tts = np.append(Y_tts, y_tts)
            
            plt.figure(figsize=(18, 6), dpi=150)
            plt.title(f"Gummy Worm Dataset - {model_name}: Primitive Approach for optimal ECE", fontsize=14, fontweight='bold')
            plt.ylabel("Metrics")
            plt.xlabel("Test Sample Size")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.plot(subsample_sizes, means["True ECE Grid (Binned - 100 Bins)"], label="True ECE (Grid - 100 Bins - 400.000)")
            plt.plot(subsample_sizes, means["True ECE Grid (Binned - 15 Bins)"],  label="True ECE (Grid - 15 Bins - 400.000)")
            plt.plot(subsample_sizes, means["True ECE Dists (Binned - 100 Bins)"], label="True ECE (Dists - 100 Bins - 400.000)")
            plt.plot(subsample_sizes, means["True ECE Dists (Binned - 15 Bins)"],  label="True ECE (Dists - 15 Bins - 400.000)")
            plt.plot(subsample_sizes, ece_values, label="ECE (15 Bins)")
            plt.scatter(first_plateau_sample_size, first_plateau_ece, label="First Plateau", color='black')
            plt.scatter(optimal_sample_size, optimal_ece, label="Optimal ECE", color='blue', marker='s')
            plt.scatter(approx_sample_size, approx_ece, label="Approx. ECE", color='red')
            plt.scatter(subsample_sizes[49], y_tts, label="ECE on 20% Test Set", color='yellow')
            plt.legend(title='Legend', loc='upper right', bbox_to_anchor=(1, 1))
            plt.show()


print("Mean Subsample Size ECE Approximation: ", np.mean(approx_subsample_sizes))
print("Std. Dev. Subsample Size ECE Approximation: ", np.std(approx_subsample_sizes))
print("Mean Subsample Size ECE Optimal: ", np.mean(optimal_subsample_sizes))
print("Std. Dev. Subsample Size ECE Optimal: ", np.std(optimal_subsample_sizes))

X = np.vstack(X)
Xs_and_Ys = np.hstack([X, Y.reshape(-1, 1), Y_hat.reshape(-1, 1), Y_true.reshape(-1, 1), Y_tts.reshape(-1, 1)])


now = datetime.now()

plt.clf()

norm = Normalize(vmin=X[:, 2].min(), vmax=X[:, 2].max())
fig = plt.figure(figsize=(12, 12), dpi=600)
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X[:, 0], X[:, 1], Y, c=X[:, 2], cmap='viridis', norm=norm, label='Optimal ECE')
ax.plot(X[:, 0], X[:, 1], Y, c='k', alpha=0.3)
ax.scatter(X[:, 0], X[:, 1], Y_hat, color='red', label='Approx. ECE')
ax.plot(X[:, 0], X[:, 1], Y_hat, color='red', label='Approx. ECE', alpha=0.4)
ax.scatter(X[:, 0], X[:, 1], Y_true, color='green', label='True ECE Dists (Binned - 100 Bins)')
ax.plot(X[:, 0], X[:, 1], Y_true, color='green', label='True ECE Dists (Binned - 100 Bins)', alpha=0.4)
plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1.05, 1))
ax.view_init(elev=15, azim=45)
plt.title(f"Optimal ECE (threshold: {min_plateau_threshold} - {max_plateau_threshold} -- steps: {plateau_steps})", fontsize=24, fontweight='bold')

plt.colorbar(sc, ax=ax, label='Relative ECE Falloff')

ax.set_xlabel('first plateau sample size', fontsize=12, labelpad=20, fontweight='bold')
ax.set_ylabel('mean accuracy', fontsize=12, labelpad=20, fontweight='bold')
ax.set_zlabel('Metrics', fontsize=12, labelpad=20, fontweight='bold')

plt.savefig(f'./test3d_{now.strftime("%Y%m%d_%H%M%S")}.png')

#2d plots
Xs_and_Ys = Xs_and_Ys[Xs_and_Ys[:, 0].argsort()]
plt.clf()
plt.figure(figsize=(12, 9), dpi=300)
plt.grid(True, linestyle='--', alpha=0.6)
plt.scatter(Xs_and_Ys[:, 0], Xs_and_Ys[:, 5], color='green', label='True ECE Dists (Binned - 100 Bins)', marker='s', s=50)
plt.plot(Xs_and_Ys[:, 0], Xs_and_Ys[:, 5], color='green', alpha=0.4)
plt.scatter(Xs_and_Ys[:, 0], Xs_and_Ys[:, 3], color='blue', label='Optimal ECE', s=50)
plt.plot(Xs_and_Ys[:, 0], Xs_and_Ys[:, 3], color='blue', alpha=0.4)
plt.scatter(Xs_and_Ys[:, 0], Xs_and_Ys[:, 4], color='red', label='Approx. ECE')
plt.plot(Xs_and_Ys[:, 0], Xs_and_Ys[:, 4], color='red', alpha=0.4)
plt.scatter(Xs_and_Ys[:, 0], Xs_and_Ys[:, 6], color='yellow', label='ECE on 20% Test Set')
plt.plot(Xs_and_Ys[:, 0], Xs_and_Ys[:, 6], color='yellow', alpha=0.4)
plt.legend(title='Legend', loc='upper right', bbox_to_anchor=(1.05, 1))
plt.title(f"Optimal ECE (threshold: {min_plateau_threshold} - {max_plateau_threshold} -- steps: {plateau_steps})", fontsize=24, fontweight='bold')

plt.xlabel('Subsample Size of First Plateau', fontsize=12, labelpad=20, fontweight='bold')
plt.ylabel('Metrics', fontsize=12, labelpad=20, fontweight='bold')

plt.savefig(f'./test_plateau_sample_size{now.strftime("%Y%m%d_%H%M%S")}.png')



plt.clf()
Xs_and_Ys = Xs_and_Ys[Xs_and_Ys[:, 1].argsort()]

plt.figure(figsize=(12, 9), dpi=300)
plt.grid(True, linestyle='--', alpha=0.6)
plt.scatter(Xs_and_Ys[:, 1], Xs_and_Ys[:, 5], color='green', label='True ECE Dists (Binned - 100 Bins)', marker='s', s=50)
plt.plot(Xs_and_Ys[:, 1], Xs_and_Ys[:, 5], color='green', alpha=0.4)
plt.scatter(Xs_and_Ys[:, 1], Xs_and_Ys[:, 3], color='blue', label='Optimal ECE', s=50)
plt.plot(Xs_and_Ys[:, 1], Xs_and_Ys[:, 3], color='blue', alpha=0.4)
plt.scatter(Xs_and_Ys[:, 1], Xs_and_Ys[:, 4], color='red', label='Approx. ECE')
plt.plot(Xs_and_Ys[:, 1], Xs_and_Ys[:, 4], color='red', alpha=0.4)
plt.scatter(Xs_and_Ys[:, 1], Xs_and_Ys[:, 6], color='yellow', label='ECE on 20% Test Set')
plt.plot(Xs_and_Ys[:, 1], Xs_and_Ys[:, 6], color='yellow', alpha=0.4)
plt.legend(title='Legend', loc='upper right', bbox_to_anchor=(1.05, 1))
plt.title(
    f"Optimal ECE (threshold: {min_plateau_threshold} - {max_plateau_threshold} -- steps: {plateau_steps})",
    fontsize=24, fontweight='bold'
)

plt.xlabel('Mean Accuracy', fontsize=12, labelpad=20, fontweight='bold')
plt.ylabel('Metrics', fontsize=12, labelpad=20, fontweight='bold')

plt.savefig(f'./test_mean_accuracy{now.strftime("%Y%m%d_%H%M%S")}.png')


plt.clf()
Xs_and_Ys = Xs_and_Ys[Xs_and_Ys[:, 2].argsort()]

plt.figure(figsize=(12, 9), dpi=300)
plt.grid(True, linestyle='--', alpha=0.6)
plt.scatter(Xs_and_Ys[:, 2], Xs_and_Ys[:, 5], color='green', label='True ECE Dists (Binned - 100 Bins)', marker='s', s=50)
plt.plot(Xs_and_Ys[:, 2], Xs_and_Ys[:, 5], color='green', alpha=0.4)
plt.scatter(Xs_and_Ys[:, 2], Xs_and_Ys[:, 3], color='blue', label='Optimal ECE', s=50)
plt.plot(Xs_and_Ys[:, 2], Xs_and_Ys[:, 3], color='blue', alpha=0.4)
plt.scatter(Xs_and_Ys[:, 2], Xs_and_Ys[:, 4], color='red', label='Approx. ECE')
plt.plot(Xs_and_Ys[:, 2], Xs_and_Ys[:, 4], color='red', alpha=0.4)
plt.scatter(Xs_and_Ys[:, 2], Xs_and_Ys[:, 6], color='yellow', label='ECE on 20% Test Set')
plt.plot(Xs_and_Ys[:, 2], Xs_and_Ys[:, 6], color='yellow', alpha=0.4)
plt.legend(title='Legend', loc='upper right', bbox_to_anchor=(1.05, 1))
plt.title(f"Optimal ECE (threshold: {min_plateau_threshold} - {max_plateau_threshold} -- steps: {plateau_steps})", fontsize=24, fontweight='bold')

plt.xlabel('Relative ECE Falloff', fontsize=12, labelpad=20, fontweight='bold')
plt.ylabel('Metrics', fontsize=12, labelpad=20, fontweight='bold')

plt.savefig(f'./test_ece_falloff{now.strftime("%Y%m%d_%H%M%S")}.png')


plt.clf()
df = pd.DataFrame({
    '1. Plateau': X[:, 0],
    'Mean Accuracy': X[:, 1],
    'Rel. ECE Falloff': X[:, 2],
    'ECE (20%)': Y_tts,
    'Approx. ECE': Y_hat,
    'Optimal ECE': Y,
    'True ECE (Dists - 100 bins)': Y_true
})

plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Between Features, True ECE, Optimal ECE, and Approx. ECE")
plt.show()

plt.clf()
plt.figure(figsize=(12, 12))

# Approx. ECE vs Y
plt.subplot(2, 2, 1)
sns.scatterplot(x=df['True ECE (Dists - 100 bins)'], y=df['Approx. ECE'])
plt.plot([df['True ECE (Dists - 100 bins)'].min(), df['True ECE (Dists - 100 bins)'].max()], [df['True ECE (Dists - 100 bins)'].min(), df['True ECE (Dists - 100 bins)'].max()], 'r--')
plt.xlabel("True ECE")
plt.ylabel("Approx. ECE")
plt.title("True ECE vs. Approx. ECE")

# Approx. ECE vs Optimal ECE
plt.subplot(2, 2, 2)
sns.scatterplot(x=df['Optimal ECE'], y=df['Approx. ECE'])
plt.plot([df['Optimal ECE'].min(), df['Optimal ECE'].max()], [df['Optimal ECE'].min(), df['Optimal ECE'].max()], 'r--')
plt.xlabel("Optimal ECE")
plt.ylabel("Approx. ECE")
plt.title("Optimal ECE vs. Approx. ECE")

plt.subplot(2, 2, 3)
sns.scatterplot(x=df['True ECE (Dists - 100 bins)'], y=df['ECE (20%)'])
plt.plot([df['True ECE (Dists - 100 bins)'].min(), df['True ECE (Dists - 100 bins)'].max()], [df['True ECE (Dists - 100 bins)'].min(), df['True ECE (Dists - 100 bins)'].max()], 'r--')
plt.xlabel("True ECE")
plt.ylabel("ECE (20% Test-Set")
plt.title("True ECE vs ECE (20% Test-Set")

# Y_hat vs Optimal ECE
plt.subplot(2, 2, 4)
sns.scatterplot(x=df['Optimal ECE'], y=df['ECE (20%)'])
plt.plot([df['Optimal ECE'].min(), df['Optimal ECE'].max()], [df['Optimal ECE'].min(), df['Optimal ECE'].max()], 'r--')
plt.xlabel("Optimal ECE")
plt.ylabel("ECE (20% Test-Set")
plt.title("Optimal ECE vs ECE (20% Test-Set")

plt.tight_layout()
plt.show()



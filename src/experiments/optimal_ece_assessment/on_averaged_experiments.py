import pickle

import numpy as np
from matplotlib import pyplot as plt

from src.metrics.primitive_ece_sample_assessment import approximate_optimal_ece_sample_size

svm_filename = "Gummy Worm Dataset__SVM__20_TrainTestSplits__AbsoluteValues__20250318_220436"
nn_filename = 'Gummy Worm Dataset__Neural Network__20_TrainTestSplits__AbsoluteValues__20250320_221632'
lr_filename = 'Gummy Worm Dataset__Logistic Regression__20_TrainTestSplits__AbsoluteValues__20250321_225921'
rf_filename = 'Gummy Worm Dataset__Random Forest__20_TrainTestSplits__AbsoluteValues__20250321_225921'

svm_filename_family = 'Gummy Worm Dataset__SVM__Gummy Worm Dataset Family__AbsoluteValues__20250311_015848'
nn_filename_family = 'Gummy Worm Dataset__Neural Network__Gummy Worm Dataset Family__AbsoluteValues__20250311_015848'
lr_filename_family = 'Gummy Worm Dataset__Logistic Regression__Gummy Worm Dataset Family__AbsoluteValues__20250311_015848'
rf_filename_family = 'Gummy Worm Dataset__Random Forest__Gummy Worm Dataset Family__AbsoluteValues__20250311_015848'

files = {
    'SVM': svm_filename,
    'Neural Network': nn_filename,
    'Logistic Regression': lr_filename,
    'Random Forest': rf_filename
}

files_family = {
    # 'SVM': svm_filename_family,  Currently the pickle file is corrupted
    'Neural Network': nn_filename_family,
    'Logistic Regression': lr_filename_family,
    'Random Forest': rf_filename_family
}

diffs_optimal_to_true_ece_dists_100bins = np.array([])
diffs_optimal_to_true_ece_dists_15bins = np.array([])
diffs_optimal_to_closest_true_ece = np.array([])

def interpolate_indexed_value_array(indexing_array: np.ndarray, value_array: np.ndarray, searched_for):
    index = 0
    while indexing_array[index] < searched_for:
        index += 1

    return np.abs((value_array[index - 1] + value_array[index]) / 2)

for file_info in [(files, 'varying_test_sample_size_train_test_split_seeds'), (files_family, 'varying_test_sample_size_dataset_family')]:
    for model_name, filename in file_info[0].items():
        with (open(f'../{file_info[1]}/data/{filename}.pkl', 'rb') as file):
            results = pickle.load(file)

            # Extract values
            means = results["Means"]
            std_devs = results["Std Devs"]
            subsample_sizes = np.linspace(100, 20000, 200, dtype=np.int64)
            ece_values = means["ECE"]
            accuracies = means["Accuracy"]

            # Calculate optimal ece sample size
            optimal_sample_size, iterations, first_plateau_sample_size, first_plateau_ece_value = (
                approximate_optimal_ece_sample_size(
                    0.0005, 30, subsample_sizes, ece_values, accuracies
                )
            )

            # Interpolate ece values
            optimal_ece_interpolated = interpolate_indexed_value_array(subsample_sizes, ece_values, optimal_sample_size)

            # Interpolate true_ece values
            true_ece_dists_100bins_interpolated = interpolate_indexed_value_array(subsample_sizes, means["True ECE Dists (Binned - 100 Bins)"], optimal_sample_size)
            true_ece_dists_15bins_interpolated = interpolate_indexed_value_array(subsample_sizes, means["True ECE Dists (Binned - 15 Bins)"], optimal_sample_size)
            true_ece_grid_100bins_interpolated = interpolate_indexed_value_array(subsample_sizes, means["True ECE Grid (Binned - 100 Bins)"], optimal_sample_size)
            true_ece_grid_15bins_interpolated = interpolate_indexed_value_array(subsample_sizes, means["True ECE Grid (Binned - 15 Bins)"], optimal_sample_size)

            diffs_optimal_to_true_ece_dists_100bins = np.append(
                diffs_optimal_to_true_ece_dists_100bins, np.abs(optimal_ece_interpolated - true_ece_dists_100bins_interpolated)
            )
            diffs_optimal_to_true_ece_dists_15bins = np.append(
                diffs_optimal_to_true_ece_dists_15bins, np.abs(optimal_ece_interpolated - true_ece_dists_15bins_interpolated)
            )
            diffs_optimal_to_closest_true_ece = np.append(
                diffs_optimal_to_closest_true_ece,
                np.min(
                    np.abs(
                        np.array([optimal_ece_interpolated] * 4) -
                        np.array([true_ece_dists_100bins_interpolated,
                         true_ece_dists_15bins_interpolated,
                         true_ece_grid_100bins_interpolated,
                         true_ece_grid_15bins_interpolated])
                    )
                )
            )

            plt.figure(figsize=(18, 6), dpi=150)
            plt.title(f"Gummy Worm Dataset - {model_name}: Primitive Approach for optimal ECE", fontsize=14, fontweight='bold')
            plt.ylabel("Metrics")
            plt.xlabel("Test Sample Size")
            plt.grid(True, linestyle='--', alpha=0.6)
            #if model_name == "Random Forest":
            plt.plot(subsample_sizes, means["True ECE Grid (Binned - 100 Bins)"], label="True ECE (Grid - 100 Bins - 400.000)")
            plt.plot(subsample_sizes, means["True ECE Grid (Binned - 15 Bins)"],  label="True ECE (Grid - 15 Bins - 400.000)")
            plt.plot(subsample_sizes, means["True ECE Dists (Binned - 100 Bins)"], label="True ECE (Dists - 100 Bins - 400.000)")
            plt.plot(subsample_sizes, means["True ECE Dists (Binned - 15 Bins)"],  label="True ECE (Dists - 15 Bins - 400.000)")
            plt.plot(subsample_sizes, ece_values, label="ECE (15 Bins)")
            plt.scatter(first_plateau_sample_size, first_plateau_ece_value, label="First Plateau", color='green')
            plt.scatter(optimal_sample_size, optimal_ece_interpolated, label="Optimal ECE", color='red')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout(pad=1.12)

            plt.show(block=False)

            print("Model: ", model_name)
            print(" Sample Sizes:", subsample_sizes)
            print(" Iterations:", iterations)
            print(" Optimal ECE Sample Size (1/2): ", optimal_sample_size)
            print(" Optimal ECE Sample Size (Exp): ", optimal_sample_size)
            print(" First Plateau Sample Size:", first_plateau_sample_size)
            print("")

print("Summary:")
print("Mean difference to True ECE Dists (Binned - 100 Bins):", np.mean(diffs_optimal_to_true_ece_dists_100bins))
print("Std. Dev. difference to True ECE Dists (Binned - 100 Bins):", np.std(diffs_optimal_to_true_ece_dists_100bins))
print("")
print("Mean difference to True ECE Dists (Binned - 15 Bins):", np.mean(diffs_optimal_to_true_ece_dists_15bins))
print("Std. Dev. difference to True ECE Dists (Binned - 15 Bins):", np.std(diffs_optimal_to_true_ece_dists_15bins))
print("")
print("Mean difference to closest True ECE:", np.mean(diffs_optimal_to_closest_true_ece))
print("Std. Dev. difference to closest True ECE:", np.std(diffs_optimal_to_closest_true_ece))
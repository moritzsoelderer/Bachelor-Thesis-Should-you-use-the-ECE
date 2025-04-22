import pickle

import numpy as np
from matplotlib import pyplot as plt

from src.metrics.primitive_ece_sample_assessment import approximate_optimal_ece_sample_size

dir = '../varying_test_sample_size_train_test_split_seeds/data/'
svm_filename = dir + "Gummy Worm Dataset__SVM__20_TrainTestSplits__AbsoluteValues__20250318_220436"
nn_filename = dir + 'Gummy Worm Dataset__Neural Network__20_TrainTestSplits__AbsoluteValues__20250320_221632'
lr_filename = dir + 'Gummy Worm Dataset__Logistic Regression__20_TrainTestSplits__AbsoluteValues__20250321_225921'
rf_filename = dir + 'Gummy Worm Dataset__Random Forest__20_TrainTestSplits__AbsoluteValues__20250321_225921'
svm_filename_exlamation_mark = dir + 'Exclamation Mark Dataset__SVM__Exclamation Mark Dataset Family__AbsoluteValues__20250405_200029'
nn_filename_exlamation_mark = dir + 'Exclamation Mark Dataset__Neural Network__Exclamation Mark Dataset Family__AbsoluteValues__20250406_143851'

dir_family = '../varying_test_sample_size_dataset_family/data/'
svm_filename_family = dir_family + 'Gummy Worm Dataset__SVM__Gummy Worm Dataset Family__AbsoluteValues__20250405_032919'
nn_filename_family = dir_family + 'Gummy Worm Dataset__Neural Network__Gummy Worm Dataset Family__AbsoluteValues__20250311_015848'
lr_filename_family = dir_family + 'Gummy Worm Dataset__Logistic Regression__Gummy Worm Dataset Family__AbsoluteValues__20250311_015848'
rf_filename_family = dir_family + 'Gummy Worm Dataset__Random Forest__Gummy Worm Dataset Family__AbsoluteValues__20250311_015848'

file_info = {
    'SVM': [svm_filename, svm_filename_exlamation_mark, svm_filename_family],
    'Neural Network': [nn_filename, nn_filename_exlamation_mark, nn_filename_family],
    'Logistic Regression': [lr_filename, lr_filename_family],
    'Random Forest': [rf_filename, rf_filename_family]
}

diffs_optimal_to_true_ece_dists_100bins = np.array([])
diffs_optimal_to_true_ece_dists_15bins = np.array([])
diffs_optimal_to_closest_true_ece = np.array([])

diffs_tts_to_true_ece_dists_100bins = np.array([])
diffs_tts_to_true_ece_dists_15bins = np.array([])
diffs_tts_to_closest_true_ece = np.array([])

def interpolate_indexed_value_array(indexing_array: np.ndarray, value_array: np.ndarray, searched_for):
    index = 0
    while indexing_array[index] < searched_for:
        index += 1

    return np.abs((value_array[index - 1] + value_array[index]) / 2)

for model_name, files in file_info.items():
    for file in files:
        with (open(f'{file}.pkl', 'rb') as file):
            results = pickle.load(file)

            print("Model: ", model_name)

            # Extract values
            means = results["Means"]
            std_devs = results["Std Devs"]
            subsample_sizes = np.linspace(100, 20000, 200, dtype=np.int64)
            ece_values = means["ECE"]
            accuracies = means["Accuracy"]

            # Calculate optimal ece sample size
            optimal_sample_size, iterations, first_plateau_sample_size, first_plateau_ece_value = (
                approximate_optimal_ece_sample_size(
                    0.0004, 30, subsample_sizes, ece_values, accuracies
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

            # Gather ECE Value if 0.2 train test split was performed
            subsample_size_tts = subsample_sizes[40]
            ece_value_tts = ece_values[40]

            diffs_tts_to_true_ece_dists_100bins = np.append(diffs_tts_to_true_ece_dists_100bins, np.abs(ece_value_tts - means["True ECE Dists (Binned - 100 Bins)"][40]))
            diffs_tts_to_true_ece_dists_15bins = np.append(diffs_tts_to_true_ece_dists_15bins, np.abs(ece_value_tts - means["True ECE Dists (Binned - 15 Bins)"][40]))
            diffs_tts_to_closest_true_ece = np.append(
                diffs_tts_to_closest_true_ece,
                np.min(
                    np.abs(
                        np.array([ece_value_tts] * 4) -
                        np.array([means["True ECE Dists (Binned - 100 Bins)"][40],
                         means["True ECE Dists (Binned - 15 Bins)"][40],
                         means["True ECE Grid (Binned - 100 Bins)"][40],
                         means["True ECE Grid (Binned - 15 Bins)"][40]])
                    )
                )
            )

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
            plt.scatter(first_plateau_sample_size, first_plateau_ece_value, label="First Plateau", color='green')
            plt.scatter(optimal_sample_size, optimal_ece_interpolated, label="Optimal ECE", color='red')
            plt.scatter(subsample_size_tts, ece_value_tts, label="ECE (0.2 - train test split)", color='orange')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout(pad=1.12)

            plt.show(block=False)

            print("")
            print(" Iterations:", iterations)
            print(" Optimal ECE Sample Size (Exp): ", optimal_sample_size)
            print(" First Plateau Sample Size:", first_plateau_sample_size)
            print("")
            print("")

print("Summary:")
print("Optimal ECE")
print(" Mean difference to True ECE Dists (Binned - 100 Bins):", np.mean(diffs_optimal_to_true_ece_dists_100bins))
print(" Std. Dev. difference to True ECE Dists (Binned - 100 Bins):", np.std(diffs_optimal_to_true_ece_dists_100bins))
print("")
print(" Mean difference to True ECE Dists (Binned - 15 Bins):", np.mean(diffs_optimal_to_true_ece_dists_15bins))
print(" Std. Dev. difference to True ECE Dists (Binned - 15 Bins):", np.std(diffs_optimal_to_true_ece_dists_15bins))
print("")
print(" Mean difference to closest True ECE:", np.mean(diffs_optimal_to_closest_true_ece))
print(" Std. Dev. difference to closest True ECE:", np.std(diffs_optimal_to_closest_true_ece))
print("")
print("")
print("ECE (0.2 - train test split)")
print(" Mean difference to True ECE Dists (Binned - 100 Bins):", np.mean(diffs_tts_to_true_ece_dists_100bins))
print(" Std. Dev. difference to True ECE Dists (Binned - 100 Bins):", np.std(diffs_tts_to_true_ece_dists_100bins))
print("")
print(" Mean difference to True ECE Dists (Binned - 15 Bins):", np.mean(diffs_tts_to_true_ece_dists_15bins))
print(" Std. Dev. difference to True ECE Dists (Binned - 15 Bins):", np.std(diffs_tts_to_true_ece_dists_15bins))
print("")
print(" Mean difference to closest True ECE:", np.mean(diffs_tts_to_closest_true_ece))
print(" Std. Dev. difference to closest True ECE:", np.std(diffs_tts_to_closest_true_ece))
import numpy as np
from sklearn.model_selection import train_test_split

from data_generation import datasets
from src.utilities.experiment_utils import train_svm, train_neural_network, train_logistic_regression, train_random_forest, \
    predict_sklearn, predict_tf, plot_bin_count_histogram, plot_probability_masks
from src.metrics.true_ece import true_ece_binned
from src.utilities.utils import sample_uniformly_within_bounds

dg = datasets.gummy_worm_dataset_hard()
X, y = dg.generate_data(10000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
X_grid = sample_uniformly_within_bounds([-5, -5], [15, 15], size=400000)

p_true_1 = dg.cond_prob(X_test, k=1)
p_true = np.column_stack((1 - p_true_1, p_true_1))

print("Calcualting grid probabilities")
p_true_grid_1 = dg.cond_prob(X_grid, k=1)
p_true_grid = np.column_stack((1 - p_true_grid_1, p_true_grid_1))

dg.scatter2d(0, 1, show=True)

true_ece_dists_15_bin_count_list = np.array([])
true_ece_dists_100_bin_count_list = np.array([])
true_ece_grid_15_bin_count_list = np.array([])
true_ece_grid_100_bin_count_list = np.array([])

print("Training models")
model_info = {
    "SVM": (train_svm(X_train, y_train), predict_sklearn),
    "Neural Network": (train_neural_network(X_train, y_train, sample_dim=dg.n_features), predict_tf),
    "Logistic Regression": (train_logistic_regression(X_train, y_train), predict_sklearn),
    "Random Forest": (train_random_forest(X_train, y_train), predict_sklearn)
}

for model_name, model_pred_fun_tuple in model_info.items():
    print("Model Name: ", model_name)

    p_pred = model_pred_fun_tuple[1](model_pred_fun_tuple[0], X_test)
    p_pred_grid = model_pred_fun_tuple[1](model_pred_fun_tuple[0], X_grid)

    print("Plotting Probability Masks")
    plot_probability_masks(X_grid, p_true_grid, p_pred_grid)

    print("Caluclating True ECE")
    _, true_ece_dists_15_bin_count = true_ece_binned(p_pred, p_true, np.linspace(0, 1, 15))
    _, true_ece_dists_100_bin_count = true_ece_binned(p_pred, p_true, np.linspace(0, 1, 100))
    _, true_ece_grid_15_bin_count = true_ece_binned(p_pred_grid, p_true_grid, np.linspace(0, 1, 15))
    _, true_ece_grid_100_bin_count = true_ece_binned(p_pred_grid, p_true_grid, np.linspace(0, 1, 100))

    print("Plotting Bin count")
    plot_bin_count_histogram(true_ece_dists_15_bin_count, f"{model_name} - Dists 15 bin count")
    plot_bin_count_histogram(true_ece_dists_100_bin_count, f"{model_name} - Dists 100 bin count")
    plot_bin_count_histogram(true_ece_grid_15_bin_count, f"{model_name} - Grid 15 bin count")
    plot_bin_count_histogram(true_ece_grid_100_bin_count, f"{model_name} - Grid 100 bin count")

    true_ece_dists_15_bin_count_list = np.append(true_ece_dists_15_bin_count_list, true_ece_dists_15_bin_count)
    true_ece_dists_100_bin_count_list = np.append(true_ece_dists_100_bin_count_list, true_ece_dists_100_bin_count)
    true_ece_grid_15_bin_count_list = np.append(true_ece_grid_15_bin_count_list, true_ece_grid_15_bin_count)
    true_ece_grid_100_bin_count_list = np.append(true_ece_grid_100_bin_count_list, true_ece_grid_100_bin_count)


print("Mean 15 bin count: ", np.mean(true_ece_dists_15_bin_count_list))
print("Mean 100 bin count: ", np.mean(true_ece_dists_100_bin_count_list))
print("")
print("Std Dev. 15 bin count: ", np.std(true_ece_dists_15_bin_count_list))
print("Std Dev. 100 bin count: ", np.std(true_ece_dists_100_bin_count_list))
print("")
print("")
print("Mean Grid 15 bin count: ", np.mean(true_ece_grid_15_bin_count_list))
print("Mean Grid 100 bin count: ", np.mean(true_ece_grid_100_bin_count_list))
print("")
print("Std Dev. Grid 15 bin count: ", np.std(true_ece_grid_15_bin_count_list))
print("Std Dev. Grid 100 bin count: ", np.std(true_ece_grid_100_bin_count_list))


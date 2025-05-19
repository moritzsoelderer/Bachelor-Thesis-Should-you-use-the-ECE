import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from src.utilities.experiment_utils import predict_sklearn, train_svm, train_neural_network, train_logistic_regression, \
    train_random_forest, predict_tf
from src.metrics.ece import ece
from src.experiments.optimal_ece_assessment.charting_approach.approximate_optimal_ece import approximate_optimal_ece
from src.metrics.true_ece import true_ece_binned
from src.data_generation.datasets import gummy_worm_dataset


dg = gummy_worm_dataset()
X, labels = dg.generate_data(n_examples=10000)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# True ECE Stuff
X_true_ece, _ = dg.generate_data(n_examples=100000, overwrite=False)
p_true_1 = dg.cond_prob(X_true_ece, k=1)
p_true = np.column_stack((1 - p_true_1, p_true_1))

steps = 100
n_bins = 15

model_info = {
    "SVM": (train_svm(X_train, y_train), predict_sklearn),
    "Neural Network": (train_neural_network(X_train, y_train, dg.n_features), predict_tf),
    "Logistic Regression": (train_logistic_regression(X_train, y_train), predict_sklearn),
    "Random Forest": (train_random_forest(X_train, y_train), predict_sklearn)
}

for model_name, model_pred_fun_tuple in model_info.items():
    optimal_ece, optimal_sample_size, iterations, ece_values, sample_sizes, first_plateau_sample_size, first_plateau_ece_value = (
        approximate_optimal_ece(
            model_pred_fun_tuple[0], model_pred_fun_tuple[1], X_test, y_test,
            0.0005, 30, 100, steps, n_bins
        )
    )

    true_ece_pred = model_pred_fun_tuple[1](model_pred_fun_tuple[0], X_true_ece)
    true_ece_val_100bins, _ = true_ece_binned(true_ece_pred, p_true, np.linspace(0, 1, 100))
    true_ece_val_15bins, _ = true_ece_binned(true_ece_pred, p_true, np.linspace(0, 1, 15))

    print("Optimal Ece:", optimal_ece)
    print("Optimal Ece sample size:", optimal_sample_size)
    print("Iterations:", iterations)
    print("True ECE:", true_ece_val_100bins)

    # fill up with remaining ECE values, for researching purposes
    missing_sample_sizes = np.arange(start=sample_sizes[-1] + steps, stop=X_test.shape[0], step=steps)
    missing_predictions = [model_pred_fun_tuple[1](model_pred_fun_tuple[0], X_test[:size]) for size in missing_sample_sizes]
    missing_labels = [y_test[:size] for size in missing_sample_sizes]
    missing_ece_values = [ece(tup[0], tup[1], n_bins) for tup in list(zip(missing_predictions, missing_labels))]

    sample_sizes = list(dict.fromkeys(np.append(np.append(sample_sizes, missing_sample_sizes), [optimal_sample_size])))
    ece_values = list(dict.fromkeys(np.append(np.append(ece_values, missing_ece_values), [optimal_ece])))

    sorted_samples_ece_values = sorted(list(zip(sample_sizes, ece_values)), key=lambda x: x[0])

    print(sorted_samples_ece_values)
    plt.figure(figsize=(18, 6), dpi=150)
    plt.title(f"{dg.title} - {model_name}: Primitive Approach for optimal ECE", fontsize=14, fontweight='bold')
    plt.ylabel("Metrics")
    plt.xlabel("Test Sample Size")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.plot([tup[0] for tup in sorted_samples_ece_values], [true_ece_val_100bins] * len(sorted_samples_ece_values), label="True ECE (Dists - 100 Bins - 400.000)")
    plt.plot([tup[0] for tup in sorted_samples_ece_values], [true_ece_val_15bins] * len(sorted_samples_ece_values), label="True ECE (Dists - 15 Bins - 400.000)")
    plt.plot([tup[0] for tup in sorted_samples_ece_values], [tup[1] for tup in sorted_samples_ece_values], label="Ece Values")
    plt.scatter(optimal_sample_size, optimal_ece, label="Optimal ECE", color='red')
    plt.scatter(first_plateau_sample_size, first_plateau_ece_value, label="First Plateau", color='green')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout(pad=1.12)

    plt.show(block=False)
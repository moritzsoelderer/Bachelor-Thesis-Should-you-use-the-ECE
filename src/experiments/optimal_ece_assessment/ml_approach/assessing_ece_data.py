import logging
import os
import pickle
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import src.utilities.experiment_utils as util
from src.experiments.optimal_ece_assessment.ml_approach import synthetic_data_generation as sdg
from src.metrics.ece import ece
from src.metrics.true_ece import true_ece_binned


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
            nominator = np.abs(ece_values[index] - true_ece_values[index - 1])
            denominator = np.abs(
                ece_values[index] - true_ece_values[index - 1] + true_ece_values[index] - ece_values[index - 1])
            interpolation_factor = 1 - nominator / denominator
            optimal_ece = ece_values[index - 1] - current_sign * (
                        interpolation_factor * (ece_values[index] - ece_values[index - 1]))
            true_ece = optimal_ece
            optimal_sample_size = sample_sizes[index - 1] + interpolation_factor * (
                        sample_sizes[index] - sample_sizes[index - 1])
            return optimal_ece, true_ece, optimal_sample_size
        elif np.abs(current_diff) < np.abs(last_diff):
            last_diff = current_diff
            closest_index = index
        index += 1
    return ece_values[closest_index], true_ece_values[closest_index], sample_sizes[closest_index]


def calculate_ece_on_subsets(y_pred, y_true, n_bins, subsets_sizes):
    return np.array([ece(np.array(y_pred[:size]), np.array(y_true[:size]), n_bins=n_bins) for size in subsets_sizes])


def generate_ece_data(
        num_samples: int = 40000,
        min_features: int = 2,
        max_features: int = 20,
        min_temperature: float = 0.5,
        max_temperature: float = 1.5,
        min_mask_ratio: float = 0.5,
        max_mask_ratio: float = 1.0,
):
    # Sample random variables
    num_features = np.random.randint(min_features, max_features)
    temperature = np.random.uniform(min_temperature, max_temperature)
    mask_ratio = np.random.uniform(min_mask_ratio, max_mask_ratio)

    logging.info(
        f"Generating ECE data with random parameters num_features={num_features},"
        f" temperature={temperature}, mask_ratio={mask_ratio}"
    )

    # Generate random dataset
    data_generator = sdg.SyntheticDataGenerator(num_features, num_classes=2, hidden_layers=[16])
    X, y, p_true = data_generator.generate_data(num_samples, temperature=temperature, mask_ratio=mask_ratio)

    ### Plot
    """
    cmap = plt.cm.viridis  # or 'plasma', 'coolwarm', etc.
    norm = plt.Normalize(0, 1)

    fig, ax = plt.subplots()
    mask1 = y == 1
    sc1 = ax.scatter(X[mask1, 0], X[mask1, 1], c=p_true[:, 1][mask1], cmap=cmap,
                     norm=norm, marker='o', edgecolor='k', label='y=1 (circle)')
    mask0 = y == 0
    sc0 = ax.scatter(X[mask0, 0], X[mask0, 1], c=p_true[:, 1][mask0], cmap=cmap,
                     norm=norm, marker='s', edgecolor='k', label='y=0 (square)')

    cbar = plt.colorbar(sc1, ax=ax)
    cbar.set_label('p_true')

    ax.set_xlabel('X[:, 0]')
    ax.set_ylabel('X[:, 1]')
    ax.legend()
    ax.set_title('Scatter plot of X (colored by p_true)')
    plt.tight_layout()
    plt.show()
    """

    # Split into train and test set
    X_train, X_test, y_p_train, y_p_test = train_test_split(X, list(zip(y, p_true)), test_size=0.25)
    y_train, p_true_train = zip(*y_p_train)
    y_test, p_true_test = zip(*y_p_test)

    # Train models and gather predictions
    p_pred_svm = util.train_svm(X_train, y_train).predict_proba(X_test)
    p_pred_nn = util.train_neural_network(X_train, y_train, sample_dim=num_features).predict(X_test)
    p_pred_lr = util.train_logistic_regression(X_train, y_train).predict_proba(X_test)
    p_pred_rf = util.train_random_forest(X_train, y_train).predict_proba(X_test)

    sample_sizes = np.linspace(100, len(X_test), 100).astype(np.int64)

    model_results = []

    # Calculate True ECE, ECEs and Optimal ECE
    for model, pred in [("SVM", p_pred_svm), ("Neural Network", p_pred_nn), ("Logistic Regression", p_pred_lr),
                        ("Random Forest", p_pred_rf)]:
        true_ece, _ = true_ece_binned(pred, p_true_test, np.linspace(0, 1, 15))

        eces = calculate_ece_on_subsets(pred, y_test, 15, sample_sizes)
        optimal_ece, _, optimal_sample_size = find_optimal_ece(eces, [true_ece] * len(sample_sizes), sample_sizes)
        accuracy = accuracy_score(y_test, np.argmax(pred, axis=1))

        model_results.append({
            "model_name": model,
            "p_test": pred,
            "True ECE Dists (Binned - 15 Bins)": true_ece,
            "ECEs": eces,
            "Optimal ECE": optimal_ece,
            "Optimal Sample Size": optimal_sample_size,
            "Accuracy": accuracy
        })

    return {
        "X_test": X_test,
        "y_test": y_test,
        "p_true_test": p_true_test,
        "Sample Sizes": sample_sizes,
        "model_results": model_results
    }



if __name__ == "__main__":
    # Configure Logging
    datetime_start = datetime.now()
    logging.basicConfig(
        level=logging.INFO,
        format="MS-Optimal ECE Assessment ML-Approach -- %(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f"./logs/{datetime_start.strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

    # Initialize Parameters
    num_samples = 40000
    min_features = 2
    max_features = 20
    min_temperature = 0.5
    max_temperature = 1.5
    min_mask_ratio = 0.0
    max_mask_ratio = 0.5

    logging.info(
        f"Executing Optimal ECE Assessment ML-Approach with parameters num_samples={num_samples},"
        f" min_features={min_features} and max_features={max_features}"
    )

    # Iterate
    batch_iterations = 128  # 128 to achieve 12512 data points
    batch_size = 24  # 24 to achieve 12512 data points
    for i in range(batch_iterations):
        with parallel_config(verbose=100):
            results = Parallel(n_jobs=-1, verbose=10)(  # n_jobs=-1 uses all available CPUs
                delayed(generate_ece_data)(num_samples, min_features, max_features, min_temperature, max_temperature,
                                           min_mask_ratio, max_mask_ratio)
                for j in range(batch_size)
            )

        dir_path = f"./data/{datetime_start.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(dir_path, exist_ok=True)
        file_path = f"{dir_path}/batch_{i}.pkl"

        # Dump pickle file
        with open(file_path, "wb") as f:
            pickle.dump(results, f)

        logging.info(
            f"Saved batch of length {len(results)} to file {file_path}"
        )



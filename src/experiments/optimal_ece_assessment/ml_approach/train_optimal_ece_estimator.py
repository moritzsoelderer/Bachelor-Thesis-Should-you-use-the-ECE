import pickle

import keras_tuner
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.metrics.ece import ece


def build_nn(hp) -> keras.Model:
    model = keras.Sequential()
    model.add(keras.layers.Flatten())
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(
            keras.layers.Dense(
                units=hp.Int("units", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )
        if hp.Boolean("dropout"):
            model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Dense(1, activation=hp.Choice("activation", ["relu", "tanh"])))

    loss = hp.Choice('loss', ['logCosh', 'mse'])
    if loss == 'logCosh':
        loss = keras.losses.LogCosh()

    optimizer_choice = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')

    optimizer = None
    if optimizer_choice == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["mae"],
    )
    return model


def interpolate_indexed_value_array(indexing_array: np.ndarray, value_array: np.ndarray, searched_for):
    index = None

    if indexing_array[0] > searched_for:
        return value_array[0]

    for i in range(1, len(indexing_array)):
        if indexing_array[i] > searched_for:
            index = i
            break

    if index is None:
        return value_array[-1]

    interpolation_factor = (searched_for - indexing_array[index - 1]) / (
            indexing_array[index] - indexing_array[index - 1])
    found = value_array[index - 1] + interpolation_factor * (value_array[index] - value_array[index - 1])
    return found


### Gather Data
data = []
files = ['20250509_013408.pkl']

for file_name in files:
    with (open(f'./data/{file_name}', 'rb') as file):
        # Flatten
        content = pickle.load(file)
        data = data + content

model_results = []
y_tests = []
sample_sizes = []
eces = np.array([])
for result in data:
    model_results = model_results + result["model_results"]
    for i in range(4):
        y_tests.append(result["y_test"])
        sample_sizes.append(result["Sample Sizes"])

y_tests = np.array(y_tests)
sample_sizes = np.array(sample_sizes)

print(y_tests.shape)
print(sample_sizes.shape)

p_tests = np.array([r["p_test"] for r in model_results])
eces = np.array([r["ECEs"] for r in model_results])

# Print Accuracy and other metrics
accuracies = np.array([result["Accuracy"] for result in model_results])
mean_accuracy = np.mean(accuracies)
std_dev_accuracy = np.std(accuracies)
print("Mean Accuracy", mean_accuracy)
print("Std. Dev. Accuracy", std_dev_accuracy)

dist_from_eces0 = np.array([np.linalg.norm(eces[0] - eces[i]) for i in range(len(eces)) if i != 0])
print("Distances from eces[0]", dist_from_eces0)
print("Mean distance from eces[0]", np.mean(dist_from_eces0))
print("Std. Dev distance from eces[0]", np.std(dist_from_eces0))

print("Mean True ECE", np.mean(np.array([result["True ECE Dists (Binned - 15 Bins)"] for result in model_results])))
print("Std. Dev True ECE", np.std(np.array([result["True ECE Dists (Binned - 15 Bins)"] for result in model_results])))

X = np.hstack((sample_sizes, eces))
scaler = StandardScaler()
y = np.array([result["Optimal Sample Size"] for result in model_results])
optimal_eces = np.array([result["Optimal ECE"] for result in model_results])

print("Input Shape", X.shape)
print("Output Shape", y.shape)

assert len(X) == len(y)

y = np.column_stack((y, optimal_eces))
print("Output Shape", y.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

optimal_eces_train = y_train[:, 1]
y_train = y_train[:, 0]
optimal_eces_test = y_test[:, 1]
y_test = y_test[:, 0]

# Transform input data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

tuner = keras_tuner.RandomSearch(
    hypermodel=build_nn,
    objective="val_loss",
    max_trials=50,
    executions_per_trial=2,
    directory="keras_tuner_logs",
    project_name="ece_neural_network",
)

tuner.search(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=128,
    callbacks=[keras.callbacks.EarlyStopping(patience=5)],
    verbose=1
)

best_model = tuner.get_best_models(num_models=1)[0]
y_pred = np.array(best_model.predict(X_test_scaled)).flatten()

# Print best Hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:")
for key in best_hps.values.keys():
    print(f"{key}: {best_hps.get(key)}")

# Train Regressor
model = XGBRegressor(n_estimators=500, max_depth=10)
model.fit(X_train_scaled, y_train)
y_pred_regressor = np.array(model.predict(X_test_scaled)).flatten()

# Calculate Metrics and display Dataframe
simple_strategy_preds = [
    (X_test[:, i], X_test[:, 100 + i]) for i in [0, 4, 9, 19, 49]]

preds = {
    "100 Samples": simple_strategy_preds[0],
    "500 Samples": simple_strategy_preds[1],
    "1000 Samples": simple_strategy_preds[2],
    "2000 Samples": simple_strategy_preds[3],
    "5000 Samples": simple_strategy_preds[4],
    "Neural Network": (
        # Probably results into index out of bounds for predicted sample sizes > 10000
        y_pred, np.array([ece(p_tests[i][:round(sample_size)], y_tests[i][:round(sample_size)], n_bins=15) for
                          i, sample_size in enumerate(y_pred)])),
    "XGBRegressor": (y_pred_regressor,
                     np.array([ece(p_tests[i][:round(sample_size)], y_tests[i][:round(sample_size)], n_bins=15) for
                               i, sample_size in enumerate(y_pred_regressor)])),
    "Y_Test": (y_test, optimal_eces_test)
}

# Compute stats
results = []
for name, (sample_sizes, eces) in preds.items():
    print(name, eces)
    results.append({
        "Name": name,
        "Mean (Samples)": np.mean(sample_sizes),
        "Std Dev (Samples)": np.std(sample_sizes),
        "MSE (Samples)": mean_squared_error(y_test, sample_sizes),
        "MAE (Samples)": mean_absolute_error(y_test, sample_sizes),
        "R2-Score (Samples)": r2_score(y_test, sample_sizes),
        "Mean (ECE)": np.mean(eces),
        "Std Dev (ECE)": np.std(eces),
        "MSE (ECE)": mean_squared_error(optimal_eces_test, eces),
        "MAE (ECE)": mean_absolute_error(optimal_eces_test, eces),
        "R2-Score (ECE)": r2_score(optimal_eces_test, eces),
    })

# Create DataFrame
pd.set_option('display.max_columns', None)
df = pd.DataFrame(results)
print(df)

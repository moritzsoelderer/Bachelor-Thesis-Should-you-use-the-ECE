import pickle
from pathlib import Path

import keras_tuner
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from keras.src.layers import Dropout
from keras.src.layers import Dense
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
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
    for i in range(hp.Int('num_layers', 10, 14)):
        model.add(
            Dense(
                units=hp.Int(f"units_{i}", min_value=704, max_value=1536, step=64),
                activation=hp.Choice(f"activation_{i}", ["relu", "linear"]),
            )
        )
    model.add(Dense(1, activation="relu"))
    model.add(Dropout(hp.Float("Dropout", 0.05, 0.25, sampling='log')))

    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', 1e-5, 5e-4, sampling='log')

    optimizer = None
    if optimizer_choice == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.LogCosh,
        metrics=["mae", "mse"],
    )
    return model


def interpolate_indexed_value_array(indexing_array: np.ndarray, value_array: np.ndarray, searched_for, strategy="linear"):
    f = interp1d(indexing_array, value_array, kind=strategy,
                 bounds_error=False, fill_value=(value_array[0], value_array[-1]))
    return f(searched_for)


def safe_unpickle_all(file):
    objects = []
    unpickler = pickle.Unpickler(file)
    while True:
        try:
            obj = unpickler.load()
            objects.append(obj)
        except EOFError:
            break  # End of file (incomplete last object)
        except Exception as e:
            print("Partial data recovered. Stopped at error:", e)
            break
    return objects


# Gather Data from different files
print("Gathering data")
data = []
dirs = ["./data/20250512_165251", "./data/20250511_095828", "./data/20250512_012245", "./data/20250510_185507", "./data/20250514_152839"]

for dir in dirs:
    files = Path(dir).glob("*.pkl")
    for file_name in files:
        with (open(f'{file_name}', 'rb') as file):
            try:
                content = pickle.load(file)
            except EOFError:
                print("Data is corrupted. Trying to partially recover data...")
                content = safe_unpickle_all(file)
                print(content)

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

print("y_tests shape", y_tests.shape)
print("sample sizes shape", sample_sizes.shape)

p_tests = np.array([r["p_test"] for r in model_results])
eces = np.array([r["ECEs"] for r in model_results])

print("p_tests shape", p_tests.shape)
print("eces shape", eces.shape)

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

print(sample_sizes)
print(eces)

# Prepare Data
X = np.hstack((sample_sizes, eces, accuracies.reshape(-1, 1)))
scaler = StandardScaler()
y_scaler = StandardScaler()
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
X_train_scaled = X_train[100:]
X_test_scaled = X_test[100:]
y_train_scaled = y_train
y_test_scaled = y_test

print("y_train_scaled mean:", np.mean(y_train_scaled))
print("y_train_scaled std:", np.std(y_train_scaled))


print("X_train_scaled", X_train_scaled)
# Perform Random Search
tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_nn,
    objective="val_loss",
    max_trials=5,
    executions_per_trial=1,
    directory="keras_tuner_logs_bayes_logcosh_dropout_only_ece_values",
    project_name="ece_neural_network_bayes2_logscosh_dropout_only_ece_values",
)
tuner.search(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=150,
    batch_size=256,
    callbacks=[keras.callbacks.EarlyStopping(patience=15)],
    verbose=1
)

best_model = tuner.get_best_models(num_models=1)[0]
y_pred_scaled = np.array(best_model.predict(X_test_scaled)).flatten()
y_pred = y_pred_scaled
print("y_pred_scaled NN", y_pred_scaled)

# Print best Hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:")
for key in best_hps.values.keys():
    print(f"{key}: {best_hps.get(key)}")

"""
model = keras.Sequential()
model.add(keras.layers.Flatten())
for i in range(4):
    model.add(
        Dense(
            units=480,
            activation="relu",
        )
    )
model.add(Dense(1, activation="linear"))

learning_rate = 0.0022035143136497045

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.LogCosh(),
    metrics=["mae"],
)

model.fit(X_train_scaled, y_train_scaled)
y_pred = y_scaler.inverse_transform(np.array(model.predict(X_test_scaled)).flatten().reshape(-1, 1)).ravel()
"""

print("y_pred NN", y_pred)

# Train Regressor
model = XGBRegressor(
    n_estimators=1000,
    max_depth=16,
    learning_rate=0.002,
    subsample=0.78,
    colsample_bytree=0.75,
    gamma=1,
    reg_alpha=0.15,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1,
    verbosity=1
)
model.fit(X_train_scaled[100:], y_train_scaled)
y_pred_scaled_regressor = np.array(model.predict(X_test_scaled[100:])).flatten()
y_pred_regressor = y_pred_scaled_regressor
print("y_pred_scaled regressor", y_pred_scaled_regressor)
print("y_pred_regressor", y_pred_regressor)

# Calculate Metrics and display Dataframe
simple_strategy_preds = [
    (f"{i * 100} Samples", (X_test[:, i], X_test[:, 100 + i])) for i in [0, 4, 9, 19, 49, 79, 99]]

def get_rounded_clipped_sample_size(sample_size):
    return max(15, min(10000, round(sample_size)))

preds = {}
for pred in simple_strategy_preds:
    preds[pred[0]] = pred[1]


ece_nn_interpolated = np.array([interpolate_indexed_value_array(X_test[i, :100], X_test[i, 100:200], sample_size) for
                                (i, sample_size) in enumerate(y_pred)])


ece_regressor_interpolated = np.array([interpolate_indexed_value_array(X_test[i, :100], X_test[i, 100:200], sample_size) for
                      (i, sample_size) in enumerate(y_pred_regressor)])

ece_nn_interpolated_quad = np.array([interpolate_indexed_value_array(X_test[i, :100], X_test[i, 100:200], sample_size, strategy="quadratic") for
                      (i, sample_size) in enumerate(y_pred)])

ece_regressor_interpolated_quad = np.array([interpolate_indexed_value_array(X_test[i, :100], X_test[i, 100:200], sample_size, strategy="quadratic") for
                                      (i, sample_size) in enumerate(y_pred_regressor)])
preds.update({
    "Neural Network": (
        y_pred, np.array([ece(p_tests[i][:get_rounded_clipped_sample_size(sample_size)], y_tests[i][:get_rounded_clipped_sample_size(sample_size)], n_bins=15) for
                          i, sample_size in enumerate(y_pred)])),
    "XGBRegressor": (y_pred_regressor,
                     np.array([ece(p_tests[i][:get_rounded_clipped_sample_size(sample_size)], y_tests[i][:get_rounded_clipped_sample_size(sample_size)], n_bins=15) for
                               i, sample_size in enumerate(y_pred_regressor)])),
    "Neural Network (Interpolated)": (y_pred, ece_nn_interpolated),
    "XGBRegressor (Interpolated)": (y_pred_regressor, ece_regressor_interpolated),
    "Interpolated Average": ((y_pred + y_pred_regressor)/ 2, (ece_nn_interpolated + ece_regressor_interpolated) / 2),
    "Neural Network (Interpolated - Quadratic)": (y_pred, ece_nn_interpolated_quad),
    "XGBRegressor (Interpolated - Quadratic)": (y_pred_regressor, ece_regressor_interpolated_quad),
    "Interpolated (Quadratic) Average": ((y_pred + y_pred_regressor) / 2, (ece_nn_interpolated_quad + ece_regressor_interpolated_quad) / 2),
    "Y_Test": (y_test, optimal_eces_test)
})

print("Optimal ECEs", optimal_eces_test)
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


# Plot Y_test against predictions
plt.hist(y_test, bins=50, alpha=0.5, label="True")
plt.hist(y_pred, bins=50, alpha=0.5, label="Predicted NN")
plt.hist(y_pred_regressor, bins=50, alpha=0.5, label="Predicted XGB")
plt.legend()
plt.title("Target vs Prediction Distribution")
plt.show()

# Plot Optimal ECE against ECE predictions
plt.hist(optimal_eces_test, bins=50, alpha=0.5, label="Optimal ECEs")
plt.hist(ece_nn_interpolated, bins=50, alpha=0.5, label="Predicted NN (Interpolated)")
plt.hist(ece_regressor_interpolated, bins=50, alpha=0.5, label="Predicted XGB (Interpolated)")
plt.legend()
plt.title("Target vs Prediction Distribution")
plt.show()


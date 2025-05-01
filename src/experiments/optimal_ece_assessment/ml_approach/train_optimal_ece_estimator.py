import pickle

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


data = []
files = ['20250430_155321.pkl', '20250430_151036.pkl']
for file_name in files:
    with (open(f'./data/{file_name}', 'rb') as file):
        # Flatten
        content = [item for sublist in pickle.load(file) for item in sublist]
        data = data + content


eces = np.array([result["ECEs"] for result in data])
sample_size = np.array([result["Sample Sizes"] for result in data])

X = np.hstack((eces, sample_size)) / 10000
y = np.array([result["Optimal Sample Size"] for result in data]) / 10000

print(X.shape)
print(y.shape)
assert len(X) == len(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(300, activation="tanh"))
model.add(tf.keras.layers.Dense(150, activation="tanh"))
model.add(tf.keras.layers.Dense(50, activation="tanh"))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
model.fit(X_train, y_train, epochs=15, batch_size=10, verbose=0)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Y pred (normalized)", y_pred)
print("Y (normlized)", y_test)
print("Y pred", y_pred * 10000)
print("Y", y_test * 10000)

print("MSE (normalized):", mse)
print("MSE (scaled):", mse * 10000)
print("MSE (scaled):", mean_squared_error(y_test * 10000, y_pred * 10000))
import numpy as np
from sklearn.model_selection import train_test_split

import data_generation as dg
import tensorflow as tf

from metrics import true_ece

n_datasets = 2

# calibration error metric variables
n_bins = np.array([10, 20, 50, 100, 200])
n_binned_metrics = len(["ece", "fce", "tce uniform", "tce pavabc", "ace"])
ce_matrix = np.zeros((n_binned_metrics, len(n_bins)), dtype=np.float32)
ksce_val = 0
balance_score_val = 0
true_ece_val = 0

iteration_metadata = np.zeros((n_datasets, 5), dtype=np.int64)

for i in range(n_datasets):

    # initialize parameters randomly
    n_classes = 2
    n_dists_per_class = np.random.randint(1, 10)
    n_uninformative_features = np.random.randint(1, 5)
    n_informative_features = np.random.randint(1, 5)
    n_examples_per_class_per_dist = np.random.randint(1000, 2000)  # change to realistic numbers

    # store metadata of current iteration
    iteration_metadata[i] = [n_classes, n_dists_per_class, n_informative_features,
                             n_uninformative_features, n_examples_per_class_per_dist]

    # generate and prepare data
    dataset = dg.DataGeneration.random(n_classes=n_classes, n_dists_per_class=n_dists_per_class,
                                       n_uninformative_features=n_uninformative_features,
                                       n_informative_features=n_informative_features)
    samples, labels = dataset.generate_data(n_examples=n_examples_per_class_per_dist)
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=.3)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    print("Number of training examples: ", len(X_train))
    print("Number of test examples: ", len(X_test))
    print("Number of training labels: ", len(y_train))
    print("Number of test labels: ", len(y_test))

    # train model
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(n_informative_features + n_uninformative_features, activation="tanh"))  # maybe add layers
    model.add(tf.keras.layers.Dense(100, activation="tanh"))  # maybe add layers
    model.add(tf.keras.layers.Dense(50, activation="tanh"))  # maybe add layers
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=15, batch_size=len(samples))

    # retrieve predictions, labels and true probabilities
    predictions = model.predict(X_test)
    y_test = np.argmax(y_test, axis=1)
    p_test_true = np.array([[dataset.cond_prob(x, k=0), dataset.cond_prob(x, k=1)] for x in X_test])

    # print dataframe (for debugging)
    # not_binned_df, binned_df = true_ece.print_calibration_error_summary_table(predictions, y_test,
    # p_test_true, n_bins)
    # print(not_binned_df)
    # print(binned_df)

    next_true_ece_val = true_ece.true_ece(predictions, p_test_true)
    next_ce_matrix, next_balance_score_val, next_ksce_val = true_ece.calibration_error_summary(predictions, y_test,
                                                                                               n_bins)

    true_ece_val += next_true_ece_val
    balance_score_val += next_balance_score_val
    ksce_val += next_ksce_val
    ce_matrix += next_ce_matrix

    # scatter dataset (for debugging)
    dataset.scatter2d(show=True)

# normalize values
true_ece_val = np.round(true_ece_val / n_datasets, 3)
balance_score_val = np.round(balance_score_val / n_datasets, 3)
ksce_val = np.round(ksce_val / n_datasets, 3)
ce_matrix = np.round(np.divide(ce_matrix, n_datasets), 3)

# average metadata
averaged_metadata = np.sum(iteration_metadata, axis=0) / len(iteration_metadata)

print("metrics (avg): ")
print(true_ece_val)
print(balance_score_val)
print(ksce_val)
print(ce_matrix)

print("iteration metadata: ")
print(iteration_metadata)

print("averaged metadata: ")
print(averaged_metadata)

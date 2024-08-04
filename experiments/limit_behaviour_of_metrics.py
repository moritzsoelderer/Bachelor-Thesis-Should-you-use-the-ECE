import numpy as np

import data_generation as dg

n_datasets = 4

for i in range(n_datasets):
    n_classes = np.random.randint(2, 10)
    n_dists_per_class = np.random.randint(1, 10)
    n_uninformative_features = np.random.randint(1, 5)
    n_informative_features = np.random.randint(1, 5)

    test = dg.DataGeneration.random(n_classes=n_classes, n_dists_per_class=n_dists_per_class,
                                    n_uninformative_features=n_uninformative_features,
                                    n_informative_features=n_informative_features)
    test.generate_data(n_examples=1000)
    test.scatter2d(show=True)
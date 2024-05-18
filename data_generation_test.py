import numpy as np
import scipy.stats as st

import data_generation as dg

dist1_1 = st.multivariate_normal(mean=[-6, -6], cov=1, allow_singular=True)
dist1_2 = st.multivariate_normal(mean=[6, 6], cov=1, allow_singular=True)

class_object1 = dg.ClassObject([dist1_1, dist1_2])

cov1 = [
    [5, 0],
    [0, 0]
]
cov2 = [
    [0, 0],
    [0, 1]
]
dist2_1 = st.multivariate_normal(mean=[0, 0], cov=cov1, allow_singular=True)
dist2_2 = st.multivariate_normal(mean=[0, 0], cov=cov2, allow_singular=True)

class_object2 = dg.ClassObject([dist2_1, dist2_2])

dist3_1 = st.multivariate_normal(mean=[-3, 2], cov=1, allow_singular=True)
dist3_2 = st.multivariate_normal(mean=[-5, 7], cov=1, allow_singular=True)

class_object3 = dg.ClassObject([dist3_1, dist3_2])

test = dg.DataGeneration(
    2, [class_object1, class_object2, class_object3],
    n_uninformative_features=5, title="dg-test"
)

n_samples_per_class_and_dist = [
    [10000, 10000],
    [10000, 10000],
    [10000, 10000]
]

samples, labels = test.generate_data(n_samples_per_class_and_dist)

colormap = np.array(['orange', 'blue', 'red'])

test.scatter2d(0, 1, colormap)
test.scatter2d(1, 2, colormap)

x = [1, 2, 3, 4, 5, 6, 7]
print("Conditional Probabilites: (X:", x, ")")
print("class 0:", test.cond_prob(x, 0, round_to=3))
print("class 1:", test.cond_prob(x, 1, round_to=3))

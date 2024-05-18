import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt

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

test = dg.DataGeneration(2, [class_object1, class_object2])

n_samples_per_class_and_dist = [
    [10000, 10000],
    [10000, 10000]
]

samples, labels = test.generate_data(n_samples_per_class_and_dist)

colormap = np.array(['orange', 'blue'])

test.scatter2d(0, 1, colormap)

print("Conditional Probabilites: (X: [6,6])")
print("class 0:", test.cond_prob([6, 6], 0, round_to=3))
print("class 1:", test.cond_prob([6, 6], 1, round_to=3))

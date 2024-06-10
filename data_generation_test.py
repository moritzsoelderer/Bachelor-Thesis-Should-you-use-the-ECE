import numpy as np
import scipy.stats as st

import data_generation as dg

dist1_1 = st.multivariate_normal(mean=[-6, -6], cov=1, allow_singular=True)
dist1_2 = st.multivariate_normal(mean=[6, 6], cov=1, allow_singular=True)

class_object1 = dg.ClassObject([dist1_1, dist1_2], None)

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

class_object2 = dg.ClassObject([dist2_1, dist2_2], [dg.MixtureInformation.empty(), dg.MixtureInformation.empty()])

dist3_1 = st.multivariate_normal(mean=[-3, 2], cov=1, allow_singular=True)
dist3_2 = st.multivariate_normal(mean=[-7], cov=1, allow_singular=True)

class_object3 = dg.ClassObject([dist3_1, dist3_2],
                               [dg.MixtureInformation.empty(), dg.MixtureInformation(1, 0, .5, 0)]
                               )

test = dg.DataGeneration([class_object1, class_object2, class_object3],
                         n_uninformative_features=1
                         )

n_samples_per_class_and_dist = [
    [100, 100],
    [100, 100],
    [100, 100]
]

samples, labels = test.generate_data(n_samples_per_class_and_dist)

print("sample check: ", samples)

colormap = np.array(['orange', 'blue', 'red'])

test.scatter2d(0, 1, colormap=colormap, show=True, axis1_label="test1", axis2_label="test2")
test.scatter2d(1, 2, colormap=colormap, show=True)

x = [1, 2, 3, 4, 5, 6, 7]
print("Conditional Probabilites: (X:", x, ")")
print("class 0:", test.cond_prob(x, 0, round_to=3))
print("class 1:", test.cond_prob(x, 1, round_to=3))

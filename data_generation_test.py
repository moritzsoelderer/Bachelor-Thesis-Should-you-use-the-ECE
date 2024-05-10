import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt

import data_generation as dg

test = dg.DataGeneration(2)

dist1_1 = st.multivariate_normal(mean=[-2, -2], cov=1, allow_singular=True)
dist1_2 = st.multivariate_normal(mean=[2, 2], cov=1, allow_singular=True)

class_object1 = dg.ClassObject([dist1_1, dist1_2])
dist2_1 = st.multivariate_normal(mean=[-2, 2], cov=2, allow_singular=True)
dist2_2 = st.multivariate_normal(mean=[20, -2], cov=2, allow_singular=True)

class_object2 = dg.ClassObject([dist2_1, dist2_2])

test.add_classobjects([class_object1, class_object2])

n_samples_per_class_and_dist = [
    [10000, 10000],
    [10000, 10000]
]

samples, labels = test.generate_data(n_samples_per_class_and_dist)

colormap = np.array(['red', 'blue'])

print(len(samples))
print(len(labels))
plt.scatter([s[0] for s in samples], [s[1] for s in samples], color=colormap[labels])
plt.show()

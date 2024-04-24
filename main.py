import math

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# class my_pdf(st.rv_continuous):
#   def _pdf(self, x):
#     return 3 * x ** 2  # Normalized over its range, in this case [0,1]


# my_cv = my_pdf(a=0, b=1, name='my_pdf')
s = 1000000

sample_size1 = s
sample_size2 = s
sample_size3 = s

dist1 = st.norm(loc=-1, scale=2)
y_dist1 = st.norm(loc=1, scale=4)
dist2 = st.norm(loc=-10, scale=10)
dist3 = st.norm(loc=-20, scale=10)

samples1 = np.array(dist1.rvs(size=sample_size1))
y1 = np.array(y_dist1.rvs(size=sample_size1))

samples2 = np.array(dist2.rvs(size=sample_size2))
samples3 = np.array(dist3.rvs(size=sample_size3))

sample_prob = max(dist1.pdf(-20), dist2.pdf(-20), dist3.pdf(-20))
print(dist3.pdf(-20))
# sample_prob_cum = (1/3.0)*dist1.cdf(100) + (1/3.0)*dist2.cdf(100) + (1/3.0)*dist3.cdf(100)

# np.sort(samples1)
np.sort(samples2)
np.sort(samples3)

samples_total = np.append(np.append(samples1, samples2), samples3)
np.sort(samples_total)

print(np.shape(samples1))
print(np.shape(samples2))
print(np.shape(samples1))
print(np.shape(samples_total))

print("sample prob:", sample_prob)
# print("sample prob cum:", sample_prob_cum)


# plt.hist(samples1, bins=math.ceil(sample_size1/10), label="Dist 1", density=False)
# plt.hist(samples2, bins=math.ceil(sample_size2/10), label="Dist 2", density=False)
# plt.hist(samples3, bins=math.ceil(sample_size3/10), label="Dist 3", density=False)
# plt.show()
# plt.clf()
# plt.hist(samples_total, bins=math.ceil(sample_size1+sample_size2/20), label="Merged Dist", density=True)


plt.hist2d(samples1, y1)
ax = plt.gca()
lim = 20
ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
plt.show()

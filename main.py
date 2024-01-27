from sklearn.datasets import make_classification
from matplotlib import pyplot as plt

X, y = make_classification()


plt.scatter(X[:, 0], X[:, 1], marker="o", c = y)
plt.show()
plt.scatter(X[:, 0], y, marker="o", c=y)
plt.show()
plt.clf()
plt.hist(y, 2, (0,1))
plt.show()

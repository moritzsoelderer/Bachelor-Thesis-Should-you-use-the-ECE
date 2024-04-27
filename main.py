import random

import scipy.stats as st
import matplotlib.pyplot as plt


def calc_prob(x=None, k=0, round_to=3):
    if x is None:
        raise ValueError("x is required")
    return round(sum([d.pdf(x) for d in dists[k]]) / sum([d.pdf(x) for dist in dists for d in dist]), round_to)


# initializing parameters
dists = []  # list of lists probability distributions
n_k = 3  # number of classes (=2 : binary classification)
n_k_clusters = 2  # number of clusters per class

n_examples = 10000  # number of examples per sample to draw
samples = []
plot_colors = ["blue", "red", "green"]  # colors for histograms

# initializing 1 distribution per class (for now, later it should be more)
for k in range(n_k):
    k_dists = []  # list of probability distributions of class
    for c in range(n_k_clusters):
        dist = st.norm(loc=(random.random()*20)-10, scale=random.random()*5)
        k_dists.append(dist)

        # plotting sample
        sample = dist.rvs(size=n_examples)
        samples.append(sample)
        plt.hist(sample, bins=int(n_examples / 10), label="class: " + str(k), color=plot_colors[k])
        plt.show()
        plt.clf()
    dists.append(k_dists)

# plotting all samples in one histogram
for sample in samples:
    plt.hist(sample, bins=int(n_examples / 10), alpha=0.75)
plt.show()
plt.clf()

# is code producing reasonable conditional probabilities?
test_examples = [-10, -5, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 5, 10]
for example in test_examples:
    print("X: ", example, "=> ", end='')
    for k in range(n_k):
        print("Class", k, ": ", calc_prob(example, k), "| ", end='')
    print("")

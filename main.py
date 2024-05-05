import random

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


# Calculates the conditional probability of a given class for a given input based on the input pdfs
# The idea is that by assigning input pdfs to class labels the conditional probability for a class
# is higher in regions where the input pdf takes on higher values, as there should be more points
# belonging to said class when sampling (if sample size is sufficiently big)
def calc_prob(x=None, k=0, round_to=3):
    if x is None:
        raise ValueError("x is required")
    return round(sum([d.pdf(x) for d in dists[k]]) / sum([d.pdf(x) for dist in dists for d in dist]), round_to)


# initializing parameters
dists = []  # list of lists probability distributions
n_k = 2  # number of classes (=2 : binary classification)
n_k_clusters = 2  # number of clusters per class

n_examples = 10000  # number of examples per sample to draw
samples = []
plot_colors = ["blue", "red", "green"]  # colors for histograms

# initializing 1 distribution per class (for now, later it should be more)
for k in range(n_k):
    k_dists = []  # list of probability distributions of class
    for c in range(n_k_clusters):
        dist = st.norm(loc=(random.random() * 20) - 10, scale=random.random() * 5)
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

### experiments

# do conditional probabilities provided by calc_prob coincide with relative frequencies of randomly drawn samples?
n_examples = 100000
samples = []

test_examples = [-10, -5, -2, -1, -0.5, -0.25, 0, 0.25, 0.5]

for test_example in test_examples:
    delta = .01  # delta inaccuracy for sample selection
    samples_near_test_example = []

    for dist_list in dists:
        samples_list = list(map(lambda distribution: sorted(distribution.rvs(size=n_examples)), dist_list))
        samples.append(samples_list)
        samples_list_near_test_example = \
            list(map(lambda sample: len(
                [example for example in sample if test_example + delta >= example >= test_example - delta]),
                     samples_list))
        samples_near_test_example.append(samples_list_near_test_example)

    test_example_cond_prob0 = calc_prob(test_example, 0)
    test_example_cond_prob1 = calc_prob(test_example, 1)

    test_example_rel_freq0 = (
            sum(samples_near_test_example[0]) / (sum(samples_near_test_example[0]) + sum(samples_near_test_example[1])))
    test_example_rel_freq1 = (
            sum(samples_near_test_example[1]) / (sum(samples_near_test_example[0]) + sum(samples_near_test_example[1])))

    print("Test example:", test_example)
    print("Class 0:", "Cond. Prob.:", test_example_cond_prob0, " - ", "Rel. Freq.:", test_example_rel_freq0)
    print("Class 1:", "Cond. Prob.:", test_example_cond_prob1, " - ", "Rel. Freq.:", test_example_rel_freq1)

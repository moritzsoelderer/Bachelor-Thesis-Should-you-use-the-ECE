import scipy.stats as st
import matplotlib.pyplot as plt


def calc_prob(x=None, k=0, round_to=3):
    if x is None:
        raise ValueError("x is required")
    return round(dists[k].pdf(x) / sum([d.pdf(x) for d in dists]), round_to)


# initializing parameters
dists = []  # list of probability distributions
n_k = 2  # number of classes (=2 : binary classification)
n_examples = 10000  # number of examples per sample to draw
plot_colors = ["blue", "red"]  # colors for histograms

# initializing 1 distribution per class (for now, later it should be more)
normal_means = [-1, 1]
for k in range(n_k):
    dist = st.norm(loc=normal_means[k], scale=1)
    dists.append(dist)

    # show sample
    sample = dist.rvs(size=n_examples)
    plt.hist(sample, bins=int(n_examples / 10), label=str(k), color=plot_colors[k])
    plt.legend()
    plt.show()
    plt.clf()

# is code producing reasonable conditional probabilities?
test_examples = [-10, -5, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 5, 10]
for example in test_examples:
    print(calc_prob(example, 0), " : ", calc_prob(example, 1))

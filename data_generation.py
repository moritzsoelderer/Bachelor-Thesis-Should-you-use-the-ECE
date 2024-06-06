import random

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import rv_continuous


class MixtureInformation:
    features_before: int
    features_before_value: float = None
    features_after: int
    features_after_value: float = None

    def __init__(self, features_before: int = 0, features_after: int = 0,
                 features_before_value: float = None, features_after_value: float = None):
        self.features_before = abs(features_before)
        self.features_after = abs(features_after)
        if features_before_value is not None:
            self.features_before_value = abs(features_before_value)
        if features_after_value is not None:
            self.features_after_value = abs(features_after_value)

    def add_before(self, samples: np.ndarray):
        n_examples = len(samples)
        if self.features_before_value is None:
            return np.array([np.append(st.uniform.rvs(size=n_examples), sample) for sample in samples])
        else:
            return np.array(
                [np.append([self.features_before_value] * self.features_before, sample) for sample in samples])

    def add_after(self, samples: np.ndarray):
        n_examples = len(samples)
        if self.features_before_value is None:
            return np.array([np.append(sample, st.uniform.rvs(size=n_examples)) for sample in samples])
        else:
            return np.array(
                [np.append(sample, [self.features_before_value] * self.features_before) for sample in samples])

    def remove_after(self, samples: np.ndarray):
        return np.array(samples[:len(samples) - self.features_after])

    def remove_before(self, samples: np.ndarray):
        return np.array(samples[self.features_before:])

    def trim(self, samples: np.ndarray):
        samples = self.remove_before(samples)
        samples = self.remove_after(samples)
        return samples

    @staticmethod
    def empty():
        return MixtureInformation(0, 0, None, None)


class ClassObject:
    distributions: list[rv_continuous]
    n_features: int
    mixture_information: list[MixtureInformation]
    samples: list[list]

    def __init__(self, distributions, mixture_information=None):
        if mixture_information is None:
            mixture_information = [MixtureInformation.empty()] * len(distributions)
        if len(distributions) != len(mixture_information):
            raise ValueError("distributions and mixture_information must be the same length")
        n_features_list = [self.get_n_features(distributions[i], mixture_information[i]) for i in range(len(distributions))]
        if len(set(n_features_list)) != 1:
            raise ValueError("all distributions must have the same number of features (mixture information included)")
        else:
            self.n_features = n_features_list[0]
        self.distributions = distributions
        self.mixture_information = mixture_information
        self.samples = [[]] * len(distributions)

    def sum_pdfs(self, x):
        return sum([self.distributions[i].pdf(self.mixture_information[i].trim(x)) for i in range(len(self.distributions))])

    def draw_samples(self, n_examples, index=None, overwrite=True):
        if index is None:
            samples = list(map(lambda distribution: distribution.rvs(n_examples), self.distributions))
            if overwrite:
                self.samples = samples
        elif index > len(self.distributions):
            raise ValueError("index larger than number of distributions")
        else:
            samples = self.distributions[index].rvs(n_examples)
            samples = self.mixture_information[index].add_before(samples)
            samples = self.mixture_information[index].add_after(samples)
            if overwrite:
                self.samples[index] = samples
        return samples

    @staticmethod
    def get_n_features(distribution, mixture_information):
        return len(distribution.mean) + mixture_information.features_before + mixture_information.features_after



class DataGeneration:
    title: str
    classes = list[ClassObject]
    n_informative_features: int
    n_uninformative_features: int

    samples: list[list] = None
    labels: list = None

    def __init__(self, class_objects: list[ClassObject], n_uninformative_features: int = 0,
                 title: str = None):
        n_features_list = [class_object.n_features for class_object in class_objects]
        if len(set(n_features_list)) != 1:
            raise ValueError("All class objects must have the same number of features")
        else:
            self.n_informative_features = n_features_list[0]
        self.classes = class_objects
        self.n_uninformative_features = n_uninformative_features

        if title is None:
            self.title = "DG-" + str(self.n_informative_features) + "-" + str(self.n_uninformative_features)
        else:
            self.title = str(title)

    def add_classobject(self, class_object):
        self.classes.append(class_object)

    def cond_prob(self, x, k=0, round_to=0):
        if x is None:
            raise ValueError("x is required")
        if len(x) < self.n_informative_features + self.n_uninformative_features:
            raise ValueError("x has to few features/components")
        if k > len(self.classes) - 1:
            raise ValueError("k is larger than number of classes")

        # slice to not consider uninformative features (if any)
        x = x[:self.n_informative_features]

        denominator = sum([class_object.sum_pdfs(x) for class_object in self.classes])
        if denominator == 0:
            return 0
        nominator = self.classes[k].sum_pdfs(x)

        if round_to != 0:
            return round(nominator / denominator, round_to)
        return nominator / denominator

    def generate_data(self, n_examples: list[list[int]], classes=None, overwrite=True):
        if classes is None:
            classes = [i for i in range(len(self.classes))]
        if len(n_examples) != len(classes):
            raise ValueError("Length on n_examples larger than number of classes to be considered")

        samples = [[]]
        labels = []

        for index in classes:
            for class_index in range(len(self.classes[index].distributions)):
                sample = self.classes[index].draw_samples(
                    n_examples=n_examples[index][class_index], index=class_index, overwrite=overwrite
                )
                if len(sample) != 0:
                    samples.append(sample)
                    labels += [index for _ in range(len(sample))]

        # flatten out sample list
        samples = [s for sample in samples for s in sample]

        # add uninformative features
        if self.n_uninformative_features > 0:
            samples = [np.append(sample, st.uniform.rvs(size=self.n_uninformative_features)) for sample in samples]
        if overwrite:
            self.samples = samples
            self.labels = labels
        return samples, labels

    def scatter2d(self, axis1=0, axis2=1, colormap=None, show=True):
        if colormap is None:
            colormap = np.array(['red', 'blue'])
        if len(colormap) < len(self.classes):
            diff = abs(len(self.classes) - len(colormap))
            for i in range(diff):
                colormap = np.append(colormap, colormap[-1])
        if self.samples is None:
            raise ValueError("There are no samples - maybe you need to generate some data first")
        if self.labels is None:
            raise ValueError("There are no labels - maybe you need to generate some data first")
        if axis1 > self.n_informative_features + self.n_uninformative_features - 1:
            raise ValueError("axis1 exceeds number of features")
        if axis2 > self.n_informative_features + self.n_uninformative_features - 1:
            raise ValueError("axis2 exceeds number of features")
        plt.scatter([s[axis1] for s in self.samples], [s[axis2] for s in self.samples], color=colormap[self.labels])
        plt.title(self.title)
        plt.xlabel("feature " + str(axis1))
        plt.ylabel("feature " + str(axis2))

        class_labels = [i for i in range(len(self.classes))]

        legend_elements = [
            Line2D([0], [0], marker='o', color='white', markerfacecolor=colormap[label], label=str(label))
            for label in class_labels
        ]
        plt.legend(handles=legend_elements, title="Classes", loc='upper left')
        if show:
            plt.show()
        return plt

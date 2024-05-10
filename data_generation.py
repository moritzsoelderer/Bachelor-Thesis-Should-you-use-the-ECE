import random

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous


class ClassObject:
    distributions = list[rv_continuous]
    samples: list[list]

    def __init__(self, distributions):
        self.distributions = distributions

    def sum_pdfs(self, x):
        return sum([dist.pdf(x) for dist in self.distributions])

    def draw_samples(self, n_examples, index=None, overwrite=True):
        if index is None:
            samples = list(map(lambda distribution: distribution.rvs(n_examples), self.distributions))
            if overwrite:
                self.samples = samples
        elif index > len(self.distributions):
            raise ValueError("index larger than number of distributions")
        else:
            samples = self.distributions[index].rvs(n_examples)
            if overwrite:
                self.samples[index] = samples
        return samples


class DataGeneration:
    classes = list[ClassObject]
    n_features: int

    def __init__(self, n_features: int, class_objects: list[ClassObject]):
        self.classes = class_objects
        self.n_features = n_features

    def add_classobject(self, class_object):
        self.classes.append(class_object)

    def cond_prob(self, x, k=0, round_to=0):
        if x is None:
            raise ValueError("x is required")
        if k > len(self.classes)-1:
            raise ValueError("k is larger than number of classes")
        denominator = sum([class_object.sum_pdfs(x) for class_object in self.classes])
        if denominator == 0:
            return 0
        nominator = self.classes[k].sum_pdfs(x)

        if round_to != 0:
            return round(nominator / denominator, round_to)
        return nominator / denominator

    def generate_data(self, n_examples: list[list[int]], classes=None, overwrite=False):
        if classes is None:
            classes = [i for i in range(len(self.classes))]
        elif len(n_examples) != len(classes):
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

        return [s for sample in samples for s in sample], labels

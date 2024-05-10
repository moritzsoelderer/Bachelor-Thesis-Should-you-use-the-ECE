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

    def __init__(self, n_features: int):
        self.classes = None
        self.n_features = n_features

    def add_classobjects(self, class_objects):
        self.classes = class_objects

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

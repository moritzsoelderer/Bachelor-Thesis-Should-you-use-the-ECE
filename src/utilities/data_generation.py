import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3D
from scipy.stats import rv_continuous


class MixtureInformation:
    features_before: int
    features_before_value: float = None
    features_after: int
    features_after_value: float = None

    def __init__(self, features_before: int = 0, features_after: int = 0,
                 features_before_value: float = None, features_after_value: float = None,
                 features_before_interval: tuple = (0.0, 1.0), features_after_interval: tuple = (0.0, 1.0), seed: int = 1):
        self.features_before = abs(features_before)
        self.features_after = abs(features_after)
        self.features_before_interval = features_before_interval
        self.features_after_interval = features_after_interval
        self.seed = seed
        if features_before_value is not None:
            self.features_before_value = abs(features_before_value)
        if features_after_value is not None:
            self.features_after_value = abs(features_after_value)

    def add_before(self, samples: np.ndarray):
        if self.features_before_value is None:
            scale = np.abs(self.features_before_interval[1] - self.features_before_interval[0])
            shift = self.features_before_interval[0]
            np.random.seed(seed=self.seed)
            uniform = np.array([np.append(scale * st.uniform.rvs(size=self.features_before) + shift, sample) for sample in samples])
            return uniform
        else:
            return np.array(
                [np.append([self.features_before_value] * self.features_before, sample) for sample in samples])

    def add_after(self, samples: np.ndarray):
        if self.features_after_value is None:
            scale = np.abs(self.features_after_interval[1] - self.features_after_interval[0])
            shift = self.features_after_interval[0]
            np.random.seed(seed=self.seed)
            return np.array([np.append(sample, scale * st.uniform.rvs(size=self.features_after) + shift) for sample in samples])
        else:
            return np.array(
                [np.append(sample, [self.features_after_value] * self.features_after) for sample in samples])

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
        n_features_list = [self.get_n_features(distributions[i], mixture_information[i]) for i in
                           range(len(distributions))]
        if len(set(n_features_list)) != 1:
            raise ValueError("all distributions must have the same number of features (mixture information included)")
        else:
            self.n_features = n_features_list[0]
        self.distributions = distributions
        self.mixture_information = mixture_information
        self.samples = [[]] * len(distributions)

    def sum_pdfs(self, x):
        return sum(
            [self.distributions[i].pdf(self.mixture_information[i].trim(x)) for i in range(len(self.distributions))])

    def draw_samples(self, n_examples, index=None, overwrite=True):
        if index is None:
            raise ValueError("index is required")
        elif index > len(self.distributions):
            raise ValueError("index larger than number of distributions")
        else:
            samples = self.distributions[index].rvs(n_examples)
            samples = self.mixture_information[index].add_before(samples)
            samples = self.mixture_information[index].add_after(samples)
            if overwrite:
                self.samples[index] = samples
        return np.array(samples)

    @staticmethod
    def get_n_features(distribution, mixture_information):
        return len(distribution.mean) + mixture_information.features_before + mixture_information.features_after

    @staticmethod
    def random(n_distributions=None, n_features=None):

        if n_distributions is None:
            n_distributions = np.random.randint(1, 10)
        if n_features is None:
            n_features = np.random.randint(1, 10)

        distributions = np.array([])
        mixture_information = np.array([])
        for i in range(n_distributions):
            n_informative_features = np.random.randint(1, n_features + 1)
            n_uninformative_features = n_features - n_informative_features

            mean = np.random.rand(n_informative_features) * 10
            matrix = np.array(
                [
                    [np.random.rand() if i == j else 0 for i in range(n_informative_features)]
                    for j in range(n_informative_features)
                ])
            cov = matrix * matrix.T

            distribution = st.multivariate_normal(mean=mean, cov=cov)

            n_uninformative_features_before = np.random.randint(0, n_uninformative_features + 1)
            n_uninformative_features_after = n_uninformative_features - n_uninformative_features_before
            mixture_info = MixtureInformation(features_before=n_uninformative_features_before, features_after=n_uninformative_features_after)

            distributions = np.append(distributions, [distribution])
            mixture_information = np.append(mixture_information, [mixture_info])

        return ClassObject(distributions, mixture_information)



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

    @staticmethod
    def random(title: str = None, n_classes: int = None, n_dists_per_class: int = None, n_informative_features: int = None, n_uninformative_features: int = None):

        if n_classes is None:
            n_classes = np.random.randint(1, 10)
        if n_dists_per_class is None:
            n_dists_per_class = np.random.randint(1, 10)
        if n_informative_features is None:
            n_informative_features = np.random.randint(1, 50)
        if n_uninformative_features is None:
            n_uninformative_features = np.random.randint(1, 50)

        class_objects = np.array([])

        for i in range(n_classes):
            class_object = ClassObject.random(n_dists_per_class, n_informative_features)
            class_objects = np.append(class_objects, [class_object])

        return DataGeneration(class_objects, n_uninformative_features, title)

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
        if n_examples is None:
            raise ValueError("n_examples is required")
        if not hasattr(n_examples, "__len__"):
            n_examples = [[n_examples] * len(self.classes[index].distributions) for index in classes]
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
        return np.array(samples), np.array(labels)

    def scatter2d(self, axis1=0, axis2=1, axis1_label=None, axis2_label=None, colormap=None, show=False, savePath=None):
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

        if axis1_label is None:
            axis1_label = "feature " + str(axis1)

        if axis2_label is None:
            axis2_label = "feature " + str(axis2)

        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        ax.scatter([s[axis1] for s in self.samples], [s[axis2] for s in self.samples], color=colormap[self.labels], s=0.9)
        plt.title(self.title, fontsize=14, fontweight='bold')

        plt.xlabel(axis1_label, fontsize=11)
        plt.ylabel(axis2_label, fontsize=11)

        class_labels = [i for i in range(len(self.classes))]

        legend_elements = [
            Line2D([0], [0], marker='o', color='white', markerfacecolor=colormap[label], label=str(label))
            for label in class_labels
        ]
        plt.legend(handles=legend_elements, title="Classes", loc='upper left')
        if savePath is not None:
            plt.savefig(fname=savePath)
        if show:
            plt.show()
        return plt

    def scatter3d(self, axis1=0, axis2=1, axis3=2, axis1_label=None, axis2_label=None, axis3_label=None, vert_angle=20, azimute_angle=45, colormap=None, show=False, savePath=None):
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
        if axis3 > self.n_informative_features + self.n_uninformative_features - 1:
            raise ValueError("axis3 exceeds number of features")

        if axis1_label is None:
            axis1_label = "feature " + str(axis1)

        if axis2_label is None:
            axis2_label = "feature " + str(axis2)

        if axis3_label is None:
            axis3_label = "feature " + str(axis3)

        fig = plt.figure(figsize=(9, 9), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter([s[axis1] for s in self.samples], [s[axis2] for s in self.samples], [s[axis3] for s in self.samples], color=colormap[self.labels], s=0.9)
        ax.view_init(elev=vert_angle, azim=azimute_angle)
        plt.title(self.title, fontsize=24, fontweight='bold')

        ax.set_xlabel(axis1_label, fontsize=12, labelpad=20, fontweight='bold')
        ax.set_ylabel(axis2_label, fontsize=12, labelpad=20, fontweight='bold')
        ax.set_zlabel(axis3_label, fontsize=12, labelpad=20, fontweight='bold')

        class_labels = [i for i in range(len(self.classes))]

        legend_elements = [
            Line3D([0], [0], [0], marker='o', color='white', markerfacecolor=colormap[label], label=str(label))
            for label in class_labels
        ]
        plt.legend(handles=legend_elements, title="Classes", loc='upper left')
        if savePath is not None:
            plt.savefig(fname=savePath)
        if show:
            plt.show()
        return plt



import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3D

from src.utilities.data_generation_utilities import ClassObject


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



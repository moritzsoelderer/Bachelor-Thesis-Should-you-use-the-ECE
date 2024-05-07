import random

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous


class ClassObject:
    index = int
    distributions = list[rv_continuous]
    samples = list[list]

    def __init__(self, index, distributions):
        self.index = index
        self.distributions = distributions


class DataGeneration:
    classes = list[ClassObject]

    def __init__(self):
        self.classes = None

    def add_classobjects(self, class_objects):
        index_list = [class_object.index for class_object in class_objects]
        if len(list(dict.fromkeys(index_list))) < len(index_list):
            raise ValueError("Duplicates in class objects indexes")

        self.classes = class_objects

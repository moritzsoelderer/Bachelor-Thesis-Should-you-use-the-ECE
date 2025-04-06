import numpy as np
import scipy.stats as st

from src.data_generation.data_generation import DataGeneration
from src.data_generation.data_generation_utilities import ClassObject


def dataset_parametrized(dataset_parameters: dict) -> DataGeneration:
    dist_info = dataset_parameters["dist_info"]
    dists_and_classes = [
        (st.multivariate_normal(mean=info["mean"], cov=info["cov"], allow_singular=True, seed=info["seed"]),
        info["class"])
        for info in dist_info
    ]

    distinct_classes = np.unique([clazz for _, clazz in dists_and_classes])
    class_objects = [
        ClassObject([info[0] for info in dists_and_classes if info[1] == clazz], None)
        for clazz in distinct_classes
    ]

    return DataGeneration(
            class_objects,
            n_uninformative_features=dataset_parameters["n_uninformative_features"],
            title=dataset_parameters["title"]
    )


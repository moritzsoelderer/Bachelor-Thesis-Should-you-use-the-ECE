from scipy import stats as st

from src.data_generation.data_generation import DataGeneration
from src.data_generation.data_generation_utilities import MixtureInformation, ClassObject
from src.data_generation.exclamation_mark_family import exclamation_mark_parameters, exclamation_mark_family_parameters
from src.data_generation.gummy_worm_dataset_family import gummy_worm_parameters, gummy_worm_family_parameters
from src.data_generation.util import dataset_parametrized


def gummy_worm_dataset() -> DataGeneration:
    return dataset_parametrized(gummy_worm_parameters)


def gummy_worm_dataset_family() -> [DataGeneration]:
    return [dataset_parametrized(parameters) for parameters in gummy_worm_family_parameters]


def exclamation_mark_dataset() -> DataGeneration:
    return dataset_parametrized(exclamation_mark_parameters)


def exclamation_mark_dataset_family() -> [DataGeneration]:
    return [dataset_parametrized(parameters) for parameters in exclamation_mark_family_parameters]


def imbalanced_gummy_worm_dataset() -> DataGeneration:
    dist1_1 = st.multivariate_normal(mean=[10, 10], cov=1, allow_singular=True, seed=42)
    dist1_2 = st.multivariate_normal(mean=[6, 2], cov=1.7, allow_singular=True, seed=13)
    dist1_3 = st.multivariate_normal(mean=[7, 10], cov=1, allow_singular=True, seed=165)
    dist2_2 = st.multivariate_normal(mean=[6, 6], cov=1.7, allow_singular=True, seed=37)

    class_object1 = ClassObject([dist1_1, dist1_2, dist1_3], None)
    class_object2 = ClassObject([dist2_2], None)
    return DataGeneration([class_object1, class_object2], n_uninformative_features=0, title="Imbalanced GummyWorm Dataset")


def sad_clown_dataset() -> DataGeneration:
    dist1_1 = st.multivariate_normal(mean=[-5, 9, -1.5], cov=4.5, allow_singular=True, seed=33)
    dist1_2 = st.multivariate_normal(mean=[6, -4], cov=8, allow_singular=True, seed=101)
    dist2_1 = st.multivariate_normal(mean=[7, 10, 3], cov=3, allow_singular=True, seed=57)
    dist2_2 = st.multivariate_normal(mean=[-8, -4, 5], cov=6, allow_singular=True, seed=92)
    dist1_3 = st.multivariate_normal(mean=[8, -10], cov=5, allow_singular=True, seed=1)
    dist2_3 = st.multivariate_normal(mean=[4, 4, 4], cov=2.7, allow_singular=True, seed=44)

    class_object1 = ClassObject([dist1_1, dist1_2, dist1_3],
    [
                        MixtureInformation.empty(),
                        MixtureInformation(features_after=1, features_after_interval=(-20, 10)),
                        MixtureInformation(features_before=1, features_before_interval=(-15, 10))
                    ]
    )
    class_object2 = ClassObject([dist2_1, dist2_2, dist2_3], None)
    return DataGeneration([class_object1, class_object2], n_uninformative_features=0, title="SadClown Dataset")


def imbalanced_sad_clown_dataset() -> DataGeneration:
    dist1_1 = st.multivariate_normal(mean=[-5, 9, -1.5], cov=4.5, allow_singular=True, seed=33)
    dist1_2 = st.multivariate_normal(mean=[6, -4], cov=8, allow_singular=True, seed=101)
    dist1_3 = st.multivariate_normal(mean=[8, -10], cov=5, allow_singular=True, seed=1)
    dist1_4 = st.multivariate_normal(mean=[4, 4, 4], cov=2.7, allow_singular=True, seed=44)

    dist2_1 = st.multivariate_normal(mean=[7, 10, 3], cov=3, allow_singular=True, seed=57)
    dist2_2 = st.multivariate_normal(mean=[-8, -4, 5], cov=6, allow_singular=True, seed=92)

    class_object1 = ClassObject([dist1_1, dist1_2, dist1_3, dist1_4],
    [
                        MixtureInformation.empty(),
                        MixtureInformation(features_after=1, features_after_interval=(-20, 10)),
                        MixtureInformation(features_before=1, features_before_interval=(-15, 10)),
                        MixtureInformation.empty()
                    ]
    )
    class_object2 = ClassObject([dist2_1, dist2_2], None)
    return DataGeneration([class_object1, class_object2], n_uninformative_features=0, title="Imbalanced SadClown Dataset")


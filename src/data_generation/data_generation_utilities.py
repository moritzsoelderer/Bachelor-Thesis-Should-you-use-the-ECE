import numpy as np
from scipy import stats as st
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

    def add_before(self, X: np.ndarray):
        if self.features_before_value is None:
            scale = np.abs(self.features_before_interval[1] - self.features_before_interval[0])
            shift = self.features_before_interval[0]
            np.random.seed(seed=self.seed)
            uniform = np.array([np.append(scale * st.uniform.rvs(size=self.features_before) + shift, sample) for sample in X])
            return uniform
        else:
            return np.array(
                [np.append([self.features_before_value] * self.features_before, sample) for sample in X])

    def add_after(self, X: np.ndarray):
        if self.features_after_value is None:
            scale = np.abs(self.features_after_interval[1] - self.features_after_interval[0])
            shift = self.features_after_interval[0]
            np.random.seed(seed=self.seed)
            return np.array([np.append(sample, scale * st.uniform.rvs(size=self.features_after) + shift) for sample in X])
        else:
            return np.array(
                [np.append(sample, [self.features_after_value] * self.features_after) for sample in X])

    def remove_after(self, X: np.ndarray):
        return np.array(X[:, :(X.shape[1] - self.features_after)])

    def remove_before(self, X: np.ndarray):
        return np.array(X[:, self.features_before:])

    def trim(self, X: np.ndarray):
        X = self.remove_before(X)
        X = self.remove_after(X)
        return X

    @staticmethod
    def empty():
        return MixtureInformation(0, 0, None, None)


class ClassObject:
    distributions: list[rv_continuous]
    n_features: int
    mixture_information: list[MixtureInformation]
    X: list[list]

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
        self.X = [[]] * len(distributions)

    def sum_pdfs(self, x):
        return np.sum(
            [dist.pdf(self.mixture_information[index].trim(x)) for index, dist in enumerate(self.distributions)], axis=0
        )

    def draw_samples(self, n_examples, index=None, overwrite=True):
        if index is None:
            raise ValueError("index is required")
        elif index > len(self.distributions):
            raise ValueError("index larger than number of distributions")
        else:
            X = self.distributions[index].rvs(n_examples)
            X = self.mixture_information[index].add_before(X)
            X = self.mixture_information[index].add_after(X)
            if overwrite:
                self.X[index] = X
        return np.array(X)

    def get_n_distributions(self):
        return len(self.distributions)

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

import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset


class PCA(Transformer):
    """
    It performs Principal Component Analysis (PCA) on the dataset.
    PCA is a linear algebra technique used to reduce the dimensions of the dataset.

    Parameters
    ----------
    dataset: Dataset
        The dataset to perform PCA on.
    n_components: int
        Number of components.

    Attributes
    ----------
    mean: np.ndarray
        Mean of the samples.
    components: np.ndarray
        The principal components (a matrix where each row is an eigenvector corresponding to a principal component).
    explained_variance: np.ndarray
        The amount of variance explained by each principal component (a vector of eigenvalues).
    """
    def __init__(self, dataset: Dataset, n_components: int = None, **kwargs):
        """
        PCA algorithm.

        Parameters
        ----------
        dataset: Dataset
            The dataset to perform PCA on.
        n_components: int
            Number of components.
        """
        super().__init__(**kwargs)
        if not 0 < n_components <= len(dataset.features):
            raise ValueError(f"n_components={n_components} must be greater than 0 and less or equal than the number of features={len(dataset.features)}.")
        
        self.n_components = n_components

        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, dataset: Dataset) -> 'PCA':
        """
        It fits PCA on the dataset.
        It estimates the mean, principal components, and explained variance

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        PCA
            PCA object.
        """
        self.mean = dataset.get_mean()
        centered_data = dataset.X - self.mean

        covariance_matrix = np.cov(centered_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        sorted_indexes = np.argsort(eigenvalues)[::-1]
        n = self.n_components
        self.components = eigenvectors[:, sorted_indexes[:n]]

        self.explained_variance = eigenvalues[sorted_indexes[:n]] / np.sum(eigenvalues)

        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset.
        It calculates the reduced dataset using the principal components.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        Dataset
            Transformed dataset.
        """
        centered_data = dataset.X - self.mean

        X_reduced = np.dot(centered_data, self.components)

        return X_reduced
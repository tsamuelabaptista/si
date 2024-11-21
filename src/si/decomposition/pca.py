import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset


class PCA(Transformer):
    def __init__(self, dataset: Dataset, n_components: int = None, **kwargs):
        super().__init__(**kwargs)
        if not 0 < n_components <= len(dataset.features):
            raise ValueError(f"n_components={n_components} must be greater than 0 and less or equal than the number of features={len(dataset.features)}.")
        
        self.n_components = n_components

        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, dataset: Dataset) -> 'PCA':
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
        centered_data = dataset.X - self.mean

        X_reduced = np.dot(centered_data, self.components)

        return X_reduced
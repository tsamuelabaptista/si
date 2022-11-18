from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class NN:
    """
    The NN is the Neural Network model.
    It comprehends the model topology including several neural network layers.
    The algorithm for fitting the model is based on backpropagation.

    Parameters
    ----------
    layers: list
        List of layers in the neural network.
    """
    def __init__(self, layers: list):
        """
        Initialize the neural network model.

        Parameters
        ----------
        layers: list
            List of layers in the neural network.
        """
        self.layers = layers

    def fit(self, dataset: Dataset) -> 'NN':
        """
        It fits the model to the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: NN
            The fitted model
        """
        X = dataset.X

        # forward propagation
        for layer in self.layers:
            X = layer.forward(X)

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the output of the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.ndarray
            The predicted output
        """
        X = dataset.X

        # forward propagation
        for layer in self.layers:
            X = layer.forward(X)

        return X

    def score(self, dataset: Dataset, scoring_func: Callable = accuracy) -> float:
        """
        It computes the score of the model on the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the score on
        scoring_func: Callable
            The scoring function to use

        Returns
        -------
        score: float
            The score of the model
        """
        y_pred = self.predict(dataset)
        return scoring_func(dataset.y, y_pred)


if __name__ == '__main__':
    pass

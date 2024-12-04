from typing import Callable, Union

import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    KNN Regressor
    The k-Nearest Neighbors regressor is a machine learning model that predicts the target value of new samples 
    based on a similarity measure (e.g., distance functions). This algorithm estimates the target value of a new 
    sample by averaging (or applying another aggregation function to) the target values of the k-nearest samples 
    in the training data.

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use

    Attributes
    ----------
    dataset: np.ndarray
        The training data
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Initialize the KNN regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        # parameters
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self
    
    def _get_closest_value(self, sample: np.ndarray) -> Union[float, int]:
        """
        It returns the closest value of the given sample

        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest value of

        Returns
        -------
        value: float or int
            The closest value
        """
        # compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # get the values of the k nearest neighbors
        k_nearest_neighbors_values = self.dataset.y[k_nearest_neighbors]

        # calculate the average of the values
        value = np.average(k_nearest_neighbors_values)
        return value
    
    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the target values of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the target values of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """
        predictions = np.apply_along_axis(self._get_closest_value, axis=1, arr=dataset.X)
        return predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        It returns the root mean squared error (rmse) of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on
        
        predictions: np.ndarray
            An array with the predictions 

        Returns
        -------
        accuracy: float
            The root mean squared error (rmse) of the model
        """
        return rmse(dataset.y, predictions)


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN classifier
    knn = KNNRegressor(k=3)

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The root mean squared error (rmse) of the model is: {score}')
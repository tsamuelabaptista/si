from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into stratified training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The stratified training dataset
    test: Dataset
        The stratified testing dataset
    """

    # set random state
    np.random.seed(random_state)

    # get unique class labels and their counts
    labels, counts = np.unique(dataset.y, return_counts=True)

    # initialize empty lists for train and test indices
    train_idxs = []
    test_idxs = []
    
    # loop through unique labels and their counts:
    for label, count in zip(labels, counts):
        # calculate the number of test samples for the current class
        num_test_samples = int(count * test_size)
        # get indices for the current class
        class_idxs = np.where(dataset.y == label)[0]
        # shuffle indices for the current class
        np.random.shuffle(class_idxs)
        # select indices for the current class
        test_class_idxs = class_idxs[:num_test_samples]
        # add selected indices for the current class to the test indices
        test_idxs.extend(test_class_idxs)
        # select the remaining indices for the current class
        train_class_idxs = class_idxs[num_test_samples:]
        # add the remaining indices for the current class to the train indices
        train_idxs.extend(train_class_idxs)

    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)

    return train, test
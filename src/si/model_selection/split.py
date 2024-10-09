import numpy as np

from si.data.dataset import Dataset

def train_test_split(self, dataset: Dataset, test_size: float, random_state: int) -> tuple:

    np.random.seed(random_state)
    
    permutations = np.random.permutation(dataset.shape()[0])

    test_sample_size = int(dataset.shape()[0] * test_size)

    test_idx = permutations[:test_sample_size]
    train_idx = permutations[test_sample_size:]

    train_dataset = Dataset(X=dataset.X[train_idx, :], y=dataset.y[train_idx], features=dataset.features, label=dataset.label)
    test_dataset = Dataset(X=dataset.X[test_idx, :], y=dataset.y[test_idx], features=dataset.features, label=dataset.label)
    return train_dataset, test_dataset
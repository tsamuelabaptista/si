import numpy as np
from typing import Callable
from si.data.dataset import Dataset
from random import seed


def k_fold_cross_validation(model: Callable, dataset: Dataset, scoring: Callable, cv: int = 5, random_state: int = 42) -> list[float]:
    """
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    num_samples = dataset.X.shape[0]

    sample_per_fold = num_samples // cv
    
    scores = []

    indexes = np.arange(num_samples)

    np.random.shuffle(indexes)

    for i in range(cv):
        start = sample_per_fold * i
        end = sample_per_fold * (i+1)

        test_indexes = indexes[start:end]
        train_indexes = np.concatenate((indexes[:start], indexes[end:]))

        train_dataset = Dataset(X=dataset.X[train_indexes, :], y=dataset.y[train_indexes])
        test_dataset = Dataset(X=dataset.X[test_indexes, :], y=dataset.y[test_indexes])

        model.fit(train_dataset)
        predictions = model.predict(test_dataset)

        score = scoring(test_dataset.y, predictions)

        scores.append(score)

    return scores
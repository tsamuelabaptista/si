import itertools
from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation


def grid_search_cv(model: Callable, dataset: Dataset, hyperparameter_grid: dict[str : list[list]], scoring: Callable, cv: int = None) -> dict:
    """
    """
    for hyperparameter in hyperparameter_grid:
        if not hasattr(model, hyperparameter):
            raise AttributeError("")
        
    combinations = itertools.product(*hyperparameter_grid.values())

    results = {"scores": [],
               "hyperparameters": []}

    for combination in combinations:

        for parameter_name, parameter_value in zip(hyperparameter_grid.keys(), combination):

            setattr(model, parameter_name, parameter_value)

        scores = np.array(k_fold_cross_validation(model, dataset, scoring, cv))
        results["scores"].append(np.mean(scores))
        results["hyperparameters"].append(combination)

    best_score_index = np.argmax(results["scores"])
    results["best_hyperparameters"] = results["best_hyperparameters"][best_score_index]
    results["best-score"] = np.max(results["scores"])

    return results

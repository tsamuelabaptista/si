import itertools
from typing import Any, Callable, Dict, Tuple

import numpy as np
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation


def randomized_search_cv(model, 
                         dataset: Dataset, 
                         hyperparameter_grid: Dict[str, Tuple], 
                         scoring: Callable = None, 
                         cv: int = 5, 
                         n_iter: int = 10) -> Dict[str, Any]:
    """
    Performs a randomized search cross validation on a model.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    hyperparameter_grid: Dict[str, Tuple]
        The hyperparameter grid to use.
    scoring: Callable
        The scoring function to use.
    cv: int
        The cross validation folds.
    n_iter: int
        Number of hyperparameter random combinations to test

    Returns
    -------
    results: Dict[str, Any]
        The results of the grid search cross validation. Includes the scores, hyperparameters,
        best hyperparameters and best score.
    """
    # validate the parameter grid
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")
        
    results = {'scores': [], 'hyperparameters': []}

    # get all possible hyperparameter combinations
    all_combinations = list(itertools.product(*hyperparameter_grid.values()))

    # randomly select n_iter combinations
    random_combinations = np.random.choice(len(all_combinations), size=n_iter, replace=False)

    # for each combination index
    for idx in random_combinations:
        # select the corresponding combination
        combination = all_combinations[idx]

        # parameter configuration
        parameters = {}

        # set the parameters
        for parameter, value in zip(hyperparameter_grid.keys(), combination):
            setattr(model, parameter, value)
            parameters[parameter] = value

        # cross validate the model
        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # add the score
        results['scores'].append(np.mean(score))

        # add the hyperparameters
        results['hyperparameters'].append(parameters)

    results['best_hyperparameters'] = results['hyperparameters'][np.argmax(results['scores'])]
    results['best_score'] = np.max(results['scores'])
    return results


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.models.logistic_regression import LogisticRegression

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)

    # initialize the Logistic Regression model
    knn = LogisticRegression()

    # parameter grid
    parameter_grid_ = {
        'l2_penalty': np.linspace(1, 10, 10),
        'alpha': np.linspace(0.001, 0.0001, 100),
        'max_iter': np.linspace(1000, 2000, 200)
    }

    # cross validate the model
    results_ = randomized_search_cv(knn,
                              dataset_,
                              hyperparameter_grid=parameter_grid_,
                              cv=3,
                              n_iter=10)

    # print the results
    print(results_)

    # get the best hyperparameters
    best_hyperparameters = results_['best_hyperparameters']
    print(f"Best hyperparameters: {best_hyperparameters}")

    # get the best score
    best_score = results_['best_score']
    print(f"Best score: {best_score}")
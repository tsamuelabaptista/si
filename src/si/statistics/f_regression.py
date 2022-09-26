from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from si.data.dataset import Dataset


def f_regression(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray],
                                            Tuple[float, float]]:
    """
    Scoring function for regression problems. It computes r's pearson correlation coefficients between
    each feature and the target variable. This is also called the R^2 score or Residual Sum of Squares.
    Then, it computes the F-value using the following F-statistics formula:
    F = (R^2 / (1 - R^2)) * (n - 2)
    p-values are computed using scipy.stats.f.sf (survival function) which is the same as 1 - cdf.

    Parameters
    ----------
    dataset: Dataset
        A labeled dataset

    Returns
    -------
    r: np.array, shape (n_features,)
        Pearson's correlation coefficients
    p: np.array, shape (n_features,)
        p-values
    """
    deg_of_freedom = dataset.shape()[0] - 2

    correlation_coefficient = []
    for i in range(dataset.X.shape[1]):
        r, _ = stats.pearsonr(dataset.X[:, i], dataset.y)
        correlation_coefficient.append(r)

    correlation_coefficient = np.array(correlation_coefficient)

    corr_coef_squared = correlation_coefficient ** 2
    F = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom
    p = stats.f.sf(F, 1, deg_of_freedom)
    return F, p


if __name__ == '__main__':
    from si.data.dataset import Dataset
    from si.io.csv import read_csv

    # df = pd.read_csv(r'C:\Users\BiSBII\Dropbox\SIB-2022-2023\prática\si\datasets\cpu.csv')
    # X = df.drop('perf', axis=1).to_numpy()
    # y = df['perf'].to_numpy()
    # dataset = Dataset(X, y, features=df.drop('perf', axis=1).columns, label='perf')
    dataset = read_csv(r'C:\Users\BiSBII\Dropbox\SIB-2022-2023\prática\si\datasets\cpu.csv',
                       features=True, label=True)

    F, p = f_regression(dataset)
    print(F)
    print(p)

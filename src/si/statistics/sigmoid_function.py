import numpy as np

def sigmoid_function(x: np.ndarray) -> np.ndarray:
    """
    It calculates the sigmoid function of a point (x) following the formula:
        f(x) = 1 / 1 + e^-(x)
        ...

    Parameters
    ----------
    x: np.ndarray
        Point.

    Returns
    -------
    np.ndarray
        The probability of each point in x to have the value of 1.
    """
    return 1 / (1 + np.exp(-x))

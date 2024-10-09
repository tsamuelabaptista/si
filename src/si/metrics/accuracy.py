import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    """
    return sum(y_true==y_pred) / len(y_true)
import numpy as np

def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    It computes the cosine distance of a point (x) to a set of points y.
        distance = 1 - similarity
        similarity(A,B) = cos(theta) = A.B/||A||.||B|| = sum(Ai.Bi)/sqrt(sum(Ai^2).sum(Bi^2))
        ...

    Parameters
    ----------
    x: np.ndarray
        Point.
    y: np.ndarray
        Set of points.

    Returns
    -------
    np.ndarray
        Cosine distance for each point in y.
    """
    similarity = np.sum(x * y, axis=1) / (np.sqrt(np.sum(x ** 2) * np.sum(y ** 2, axis=1)))
    similarity[np.isnan(similarity)] = 0.0  # Set NaN values to 0.0
    return 1 - similarity
    

if __name__ == '__main__':
    # test cosine_distance
    x = np.array([1, 2, 3])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    our_distance = cosine_distance(x, y)
    # using sklearn
    # to test this snippet, you need to install sklearn (pip install -U scikit-learn)
    from sklearn.metrics.pairwise import cosine_distances
    sklearn_distance = cosine_distances(x.reshape(1, -1), y)
    assert np.allclose(our_distance, sklearn_distance)
    print(our_distance, sklearn_distance)
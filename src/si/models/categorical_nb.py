import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class CategoricalNB(Model):
    """
    Categorical Naive Bayes Classifier
    A Naive Bayes classifier is a probabilistic ML algorithm that makes predictions 
    by applying Bayes' theorem with the "naive" assumption of feature independence.

    Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)
    P(A|B) -> Posterior: The probability of "A" being True, given "B" is True.
    P(B|A) -> Likelihood: The probability of "B" being True, given "A" is True.
    P(A) -> Prior: The probability of "A" being True. This is the knowledge.
    P(B) -> Marginalization: The probability of "B" being True.

    Parameters
    ----------
    smothing: float
        Laplace smoothing to avoid zero probabilities

    Attributes
    ----------
    class_prior: list
        Prior probabilities for each class
    feature_probs: list
        Probabilities for each feature for each class being present / being 1
    """
    def __init__(self, smothing: float = 1.0, **kwargs):    
        """
        Initialize the Categorical Naive-Bayes

        Parameters
        ----------
        smothing: float
            Laplace smoothing to avoid zero probabilities
        """
        # parameters
        super().__init__(**kwargs)
        self.smothing = smothing

        # attributes
        self.class_prior = None
        self.feature_probs = None

    def _fit(self, dataset: Dataset) -> 'CategoricalNB':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: CategoricalNB
            The fitted model
        """
        # define n_samples, n_features, n_classes
        n_samples, n_features = dataset.shape()
        n_classes = len(np.unique(dataset.y))

        # initialize class_counts, feature_counts, and class_prior
        class_counts = np.zeros(n_classes)
        feature_counts = np.zeros((n_classes, n_features))
        self.class_prior = np.zeros(n_classes)

        # compute class_counts, feature_counts, and class_prior
        classes, class_counts = np.unique(dataset.y, return_counts=True)
        for i, cls in enumerate(classes):
            feature_counts[i] = dataset.X[dataset.y == cls].sum(axis=0)
        self.class_prior = class_counts / n_samples

        # Apply Laplace smoothing to feature_counts and class_counts to avoid zero probabilities
        feature_counts = feature_counts + self.smothing
        # reshape class_counts as a column vector to ensure compatibility for broadcasting with feature_counts
        class_counts = class_counts[:, None] + self.smothing * n_classes # for binary classification n_classes = 2

        # compute feature_probs
        self.feature_probs = feature_counts / class_counts

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """
        # compute n_samples
        n_samples = dataset.shape()[0]
        # initialize an array to store predictions for all samples
        predictions = np.zeros(n_samples)

        # iterate over each sample
        for i in range(n_samples):
            sample = dataset.X[i] # select the i-th sample (feature vector)
            
            # initialize an array to store class probabilities for each sample
            class_probs = np.zeros(len(self.class_prior))
            # compute probabilities for each class
            for c in range(len(self.class_prior)):
                class_probs[c] = np.prod(sample * self.feature_probs[c] + (1 - sample) * (1 - self.feature_probs[c])) * self.class_prior[c]
            
            # pick the class with highest probability as the predicted class for each sample
            predictions[i] = np.argmax(class_probs)
        
        # return the array of predictions for all samples
        return predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        It returns the accuracy of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on
        
        predictions: np.ndarray
            An array with the predictions 

        Returns
        -------
        accuracy: float
            The accuracy of the model
        """
        return accuracy(dataset.y, predictions)
    

if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the Categorical NB classifier
    nb = CategoricalNB()

    # fit the model to the train dataset
    nb.fit(dataset_train)

    # evaluate the model on the test dataset
    score = nb.score(dataset_test)
    print(f'The accuracy of the model is: {score}')
        
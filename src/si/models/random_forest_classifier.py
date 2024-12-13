from typing import Literal

import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier(Model):
    """
    Class representing a random forest classifier.

    Random Forest is an ensemble machine learning technique that combines multiple decision trees to improve prediction accuracy and reduce overfitting.
    
    Parameters
    ----------
    n_estimators: int
        number of decision trees to use
    max_features: int
        maximum number of features to use per tree
    min_sample_split: int
        minimum number of samples required to split an internal node
    max_depth: int
        maximum depth of the trees.
    mode: Literal['gini', 'entropy']
        the mode to use for calculating the information gain
    seed: int
        random seed to use to assure reproducibility

    Attributes
    ----------
    trees: list[tuple(np.ndarray, DecisionTreeClassifier)]
        the trees of the random forest and respective features used for training
        a list of tuples where each tuple contains:
        - a numpy array of feature indices used for training the decision tree.
        - the trained DecisionTreeClassifier instance.
    """
    def __init__(self, n_estimators: int = 100, max_features: int = None, min_sample_split: int = 2, max_depth: int = 10, 
                 mode: Literal['gini', 'impurity'] = 'gini', seed: int = 42, **kwargs):
        """
        Creates a RandomForestClassifier object.

        Parameters
        ----------
        n_estimators: int
            number of decision trees to use
        max_features: int
            maximum number of features to use per tree
        min_sample_split: int
            minimum number of samples required to split an internal node
        max_depth: int
            maximum depth of the trees.
        mode: Literal['gini', 'entropy']
            the mode to use for calculating the information gain
        seed: int
            random seed to use to assure reproducibility
        """
        # parameters
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        # attributes
        self.trees = []

    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Fits the random forest classifier to a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to.

        Returns
        -------
        RandomForestClassifier
            The fitted model.
        """
        # set random state
        np.random.seed(self.seed)

        # get n_samples and n_features
        n_samples, n_features = dataset.shape()

        # define self.max_features
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        # repeat the steps for all trees in the forest (n_estimators)
        for _ in range(self.n_estimators):     
            
            # create a bootstrap dataset
            sample_idxs = np.random.choice(n_samples, size=n_samples, replace=True)
            feature_idxs = np.random.choice(n_features, size=self.max_features, replace=False)
            new_X = dataset.X[sample_idxs][:, feature_idxs]
            new_y = dataset.y[sample_idxs]
            new_features = np.array(dataset.features)[feature_idxs]
            bootstrap_dataset = Dataset(X=new_X, y=new_y, features=new_features, label=dataset.label)

            # create and train a decision tree with the bootstrap dataset
            tree = DecisionTreeClassifier(min_sample_split=self.min_sample_split, max_depth=self.max_depth, mode=self.mode)
            tree.fit(bootstrap_dataset)

            # append a tuple containing the features used and the trained tree
            self.trees.append((feature_idxs, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Makes predictions for a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to make predictions for.

        Returns
        -------
        np.ndarray
            The predicted values.
        """
        # get predictions for each tree using the respective set of features
        predictions_ = []
        for feature_idxs, tree in self.trees:
            # extract corresponding features for each tree
            tree_X = dataset.X[:, feature_idxs]
            # create a new dataset object with the new features
            tree_dataset = Dataset(X=tree_X, y=dataset.y, features=np.array(dataset.features)[feature_idxs], label=dataset.label)
            # get predictions for each tree
            tree_predictions = tree.predict(tree_dataset)
            predictions_.append(tree_predictions)

        predictions_ = np.array(predictions_) # shape(n_trees, n_samples)

        # get the most common predicted class for each sample
        # for each sample, count occurrences of each class label across all trees and choose the most common one
        n_samples = dataset.shape()[0]
        predictions = []
        for i in range(n_samples):
            # get predictions for each sample from all trees
            sample_predictions = predictions_[:, i]
            # count occurrences of each label
            unique_labels, counts = np.unique(sample_predictions, return_counts=True)
            # get the label with the highest count (most common predicted class for each sample)
            most_common_label = unique_labels[np.argmax(counts)]
            predictions.append(most_common_label)

        predictions = np.array(predictions)
        return predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculates the accuracy of the model on a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to calculate the accuracy on.
        predictions: np.ndarray
            Predictions

        Returns
        -------
        float
            The accuracy of the model on the dataset.
        """
        return accuracy(dataset.y, predictions)
    

if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split

    data = read_csv('datasets/iris/iris.csv', sep=',', features=True, label=True)
    train, test = train_test_split(data, test_size=0.33, random_state=42)
    model = RandomForestClassifier(min_sample_split=3, max_depth=3, mode='gini')
    model.fit(train)
    print(model.score(test))
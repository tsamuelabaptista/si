import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier(Model):
    """
    The StackingClassifier model harnesses an ensemble of models to generate predictions.
    These predictions are subsequently employed to train another model, the final model.
    The final model can then be used to predict the output variable (y).
    
    Parameters
    ----------
    models: array-like, shape = [n_models]
        Different models for the ensemble.
    final_model: Model
        The model to make the final predictions

    Attributes
    ----------
    """
    def __init__(self, models, final_model, **kwargs):
        """
        Initialize the ensemble classifier.

        Parameters
        ----------
        models: array-like, shape = [n_models]
            Different models for the ensemble.
        final_model: Model
            The model to make the final predictions

        """
        # parameters
        super().__init__(**kwargs)
        self.models = models
        self.final_model = final_model

    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit the models according to the given training data.

        Parameters
        ----------
        dataset: Dataset
            The training data.

        Returns
        -------
        self: VotingClassifier
            The fitted model.
        """
        # train the initial set of models
        for model in self.models:
            model.fit(dataset)

        # get predictions from the initial set of models
        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        # wrap the predictions as a Dataset for final_model
        features = [f"{model}" for model in range(len(self.models))]
        predictions_dataset = Dataset(X=predictions, y=dataset.y, features=features, label=dataset.label)

        # train the final model with the predictions of the initial set of models
        self.final_model.fit(predictions_dataset)

        return self
    
    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        dataset: Dataset
            The test data.

        Returns
        -------
        y: array-like, shape = [n_samples]
            The predicted class labels.
        """
        # get predictions from the initial set of models
        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        # wrap the predictions as a Dataset for final_model
        features = [f"{model}" for model in range(len(self.models))]
        predictions_dataset = Dataset(X=predictions, y=None, features=features)

        # get the final predictions using the final model and the predictions of the initial set of models
        return self.final_model.predict(predictions_dataset)
    
    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        dataset: Dataset
            The test data.
        predictions: np.ndarray
            Predictions

        Returns
        -------
        score: float
            Mean accuracy
        """
        return accuracy(dataset.y, predictions)
    

if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    from si.models.knn_classifier import KNNClassifier
    from si.models.logistic_regression import LogisticRegression
    from si.models.decision_tree_classifier import DecisionTreeClassifier

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN, Logistic and Decision Tree classifier
    knn = KNNClassifier(k=3)
    lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    dt = DecisionTreeClassifier(min_sample_split=2, max_depth=10, mode='gini')

    # initialize the second KNN classifier (final model)
    knn_final = KNNClassifier(k=3)

    # initialize the Stacking classifier
    stacking = StackingClassifier([knn, lg, dt], knn_final)

    stacking.fit(dataset_train)

    # compute the score
    score = stacking.score(dataset_test)
    print(f"Score: {score}")

    print(stacking.predict(dataset_test))
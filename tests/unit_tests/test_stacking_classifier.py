from unittest import TestCase

from datasets import DATASETS_PATH

import os

from si.ensemble.stacking_classifier import StackingClassifier
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression

class TestStackingClassifier(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):

        knn = KNNClassifier()
        logistic_regression = LogisticRegression()
        decision_tree = DecisionTreeClassifier()
        knn_final = KNNClassifier()

        sc=StackingClassifier(models = [knn, logistic_regression, decision_tree], final_model = knn_final)

        self.assertEqual(sc.models[2].min_sample_split, 2)
        self.assertEqual(sc.models[2].max_depth, 10)

    def test_predict(self):
        knn = KNNClassifier()
        logistic_regression = LogisticRegression()
        decision_tree = DecisionTreeClassifier()
        knn_final = KNNClassifier()

        sc=StackingClassifier(models = [knn, logistic_regression, decision_tree], final_model = knn_final)

        sc.fit(self.train_dataset)

        predictions = sc.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])
    
    def test_score(self):
        knn = KNNClassifier()
        logistic_regression = LogisticRegression()
        decision_tree = DecisionTreeClassifier()
        knn_final = KNNClassifier()

        sc=StackingClassifier(models = [knn, logistic_regression, decision_tree], final_model = knn_final)
        sc.fit(self.train_dataset)
        accuracy_ = sc.score(self.test_dataset)

        self.assertEqual(round(accuracy_, 2), 0.95)
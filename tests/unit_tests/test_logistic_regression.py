from unittest import TestCase

import numpy as np
from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv

from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.logistic_regression import LogisticRegression

class TestLogisticRegressor(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):

        logreg = LogisticRegression()
        logreg.fit(self.train_dataset)

        self.assertEqual(logreg.theta.shape[0], self.train_dataset.shape()[1])
        self.assertNotEqual(logreg.theta_zero, None)
        self.assertNotEqual(len(logreg.cost_history), 0)
        self.assertNotEqual(len(logreg.mean), 0)
        self.assertNotEqual(len(logreg.std), 0)

    def test_predict(self):
        logreg = LogisticRegression()
        logreg.fit(self.train_dataset)

        predictions = logreg.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])
    
    def test_score(self):
        logreg = LogisticRegression()
        logreg.fit(self.train_dataset)
        accuracy_ = logreg.score(self.test_dataset)

        self.assertEqual(round(accuracy_, 2), 0.96)
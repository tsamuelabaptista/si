from unittest import TestCase

import numpy as np


from datasets import DATASETS_PATH

import os
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.categorical_nb import CategoricalNB


class TestCategoricalNB(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):
        nb = CategoricalNB()

        nb.fit(self.dataset)

        self.assertIsNotNone(nb.class_prior)
        self.assertIsNotNone(nb.feature_probs)

    def test_predict(self):
        nb = CategoricalNB()

        nb.fit(self.train_dataset)
        predictions = nb.predict(self.test_dataset)
        self.assertEqual(predictions.shape[0], self.test_dataset.y.shape[0])

    def test_score(self):
        nb = CategoricalNB()

        nb.fit(self.train_dataset)
        score = nb.score(self.test_dataset)
        self.assertEqual(round(score, 2), 0.29)
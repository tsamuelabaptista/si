from unittest import TestCase

from datasets import DATASETS_PATH

import os

from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.random_forest_classifier import RandomForestClassifier

class TestRandomForest(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):

        random_forest = RandomForestClassifier()
        random_forest.fit(self.train_dataset)

        self.assertEqual(random_forest.min_sample_split, 2)
        self.assertEqual(random_forest.max_depth, 10)


    def test_predict(self):
        random_forest = RandomForestClassifier()
        random_forest.fit(self.train_dataset)

        predictions = random_forest.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])
    
    def test_score(self):
        random_forest = RandomForestClassifier()
        random_forest.fit(self.train_dataset)
        accuracy_ = random_forest.score(self.test_dataset)
        print(accuracy_)

        self.assertEqual(round(accuracy_, 2), 0.97)
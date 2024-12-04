from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv

from si.model_selection.split import train_test_split, stratified_train_test_split

class TestSplits(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_train_test_split(self):

        train, test = train_test_split(self.dataset, test_size = 0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)

    def test_stratified_train_test_split(self):

        test_size = 0.2
        train, test = stratified_train_test_split(self.dataset, test_size=test_size, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * test_size)

        # ensure test and train sets shape matches the expected size accounting for a discrespancy of 1 sample due to rounding during stratified splitting
        self.assertTrue(abs(test.shape()[0] - test_samples_size) <= 1)
        self.assertTrue(abs(train.shape()[0] - (self.dataset.shape()[0] - test_samples_size)) <= 1)

        # ensure the total number of samples in the test and train sets matches the original dataset size
        self.assertEqual(test.shape()[0] + train.shape()[0], self.dataset.shape()[0])
        
        _, counts = np.unique(self.dataset.y, return_counts=True)
        _, test_counts = np.unique(test.y, return_counts=True)
        _, train_counts = np.unique(train.y, return_counts=True)

        test_proportion = test_counts / counts
        train_proportion = train_counts / counts

        # ensure the total counts of train and test sets for each unique class matches the total number of counts for each unique class
        self.assertTrue(np.array_equal(test_counts + train_counts, counts))

        # ensure that each unique class has the same and expected proportion allowing small discrepancies
        self.assertTrue(np.allclose(test_proportion, test_size, atol=0.01))
        self.assertTrue(np.allclose(train_proportion, 1 - test_size, atol=0.01))
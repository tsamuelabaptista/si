import os
from unittest import TestCase

import numpy as np
from datasets import DATASETS_PATH
from si.decomposition.pca import PCA
from si.io.csv_file import read_csv

class TestPCA(TestCase):
    
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        n_components = 2
        pca = PCA(dataset=self.dataset, n_components=n_components)
        pca = pca.fit(self.dataset)

        self.assertEqual(pca.explained_variance.shape[0], n_components)
        self.assertLessEqual(sum(pca.explained_variance), 1)
        self.assertGreater(sum(pca.explained_variance), 0)

    def test_transform(self):
        n_components = 2
        pca = PCA(dataset=self.dataset, n_components=n_components)
        pca = pca.fit(self.dataset)
        X_reduced = pca.transform(self.dataset)

        self.assertEqual(X_reduced.shape, (self.dataset.X.shape[0], n_components))        



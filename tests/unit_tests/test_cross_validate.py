import os
import numpy as np
from unittest import TestCase

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.metrics.accuracy import accuracy
from si.model_selection.cross_validate import k_fold_cross_validation
from si.models.logistic_regression import LogisticRegression

class TestCrossValidate(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_cross_validate(self):
        model = LogisticRegression()
        cross_validation = np.array(k_fold_cross_validation(model=model, dataset=self.dataset, scoring=accuracy, cv=5))

        self.assertEqual(round(np.mean(cross_validation), 2), 0.97)
        self.assertEqual(round(np.std(cross_validation), 2), 0.02)
        # OR
        self.assertAlmostEqual(np.mean(cross_validation), 0.97, places=2)
        self.assertAlmostEqual(np.std(cross_validation), 0.02, places=2)
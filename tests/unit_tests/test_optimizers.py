import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH
from si.io.data_file import read_data_file
from si.neural_networks.optimizers import Adam, SGD


class TestOptimizers(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        # initialize dummy weights and gradients
        self.w = np.random.rand(self.dataset.X.shape[1], 1)
        self.grad_loss_w = np.random.rand(self.dataset.X.shape[1], 1)

    def testSGD(self):

        sgd = SGD()
        new_w = sgd.update(self.w, self.grad_loss_w)

        # ensure initial weights and new weights have the same shape
        self.assertEqual(new_w.shape, self.w.shape)
        # ensure weights are updated
        self.assertIsNotNone(new_w)
        # ensure weights have changed
        self.assertTrue(np.all(new_w != self.w))

    def testAdam(self):
        
        adam = Adam()
        new_w = adam.update(self.w, self.grad_loss_w)

        # ensure initial weights and new weights have the same shape
        self.assertEqual(new_w.shape, self.w.shape)
        # ensure weights are updated
        self.assertIsNotNone(new_w)
        # ensure weights have changed
        self.assertTrue(np.all(new_w != self.w))
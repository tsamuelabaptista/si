from unittest import TestCase

import numpy as np

from si.statistics.sigmoid_function import sigmoid_function

class TestSigmoidFunction(TestCase):
    def test_sigmoid_function(self):

        x = np.array([1.9, 10.4, 75])

        x_sigmoid = sigmoid_function(x)

        self.assertTrue(all(x_sigmoid >= 0))
        self.assertTrue(all(x_sigmoid <= 1))
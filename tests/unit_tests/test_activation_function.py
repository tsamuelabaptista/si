from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH

import os

from si.io.data_file import read_data_file
from si.neural_networks.activation import ReLUActivation, SigmoidActivation, SoftmaxActivation, TanhActivation

class TestSigmoidLayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_activation_function(self):

        sigmoid_layer = SigmoidActivation()
        result = sigmoid_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= 0 and i <= 1 for j in range(result.shape[1]) for i in result[:, j]]))
        self.assertEqual(result.shape[0], self.dataset.X.shape[0])
        self.assertEqual(result.shape[1], self.dataset.X.shape[1])

    def test_derivative(self):
        sigmoid_layer = SigmoidActivation()
        derivative = sigmoid_layer.derivative(self.dataset.X)
        self.assertTrue(all([i >= 0 and i <= 0.25 for j in range(derivative.shape[1]) for i in derivative[:, j]]))
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])

class TestRELULayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_activation_function(self):

        relu_layer = ReLUActivation()
        result = relu_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= 0 for j in range(result.shape[1]) for i in result[:, j]]))
        self.assertEqual(result.shape[0], self.dataset.X.shape[0])
        self.assertEqual(result.shape[1], self.dataset.X.shape[1])

    def test_derivative(self):
        sigmoid_layer = ReLUActivation()
        derivative = sigmoid_layer.derivative(self.dataset.X)
        self.assertTrue(np.all((derivative == 0) | (derivative == 1)))
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])

class TestTanhLayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_activation_function(self):

        tanh_layer = TanhActivation()
        result = tanh_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= -1 and i <= 1 for j in range(result.shape[1]) for i in result[:, j]]))
        self.assertEqual(result.shape[0], self.dataset.X.shape[0])
        self.assertEqual(result.shape[1], self.dataset.X.shape[1])

    def test_derivative(self):
        tanh_layer = TanhActivation()
        derivative = tanh_layer.derivative(self.dataset.X)
        self.assertTrue(all([i >= 0 and i <= 1 for j in range(derivative.shape[1]) for i in derivative[:, j]]))
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])

class TestSoftmaxLayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_activation_function(self):

        softmax_layer = SoftmaxActivation()
        result = softmax_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= 0 and i <= 1 for j in range(result.shape[1]) for i in result[:, j]]))
        self.assertTrue(np.allclose(np.sum(result, axis=1), 1.0))
        self.assertEqual(result.shape[0], self.dataset.X.shape[0])
        self.assertEqual(result.shape[1], self.dataset.X.shape[1])

    def test_derivative(self):
        softmax_layer = SoftmaxActivation()
        derivative = softmax_layer.derivative(self.dataset.X)
        self.assertTrue(all([i >= 0 and i <= 0.25 for j in range(derivative.shape[1]) for i in derivative[:, j]]))
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])
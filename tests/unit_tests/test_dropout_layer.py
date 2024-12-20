from unittest import TestCase

import numpy as np

from si.data.dataset import Dataset
from si.neural_networks.layers import Dropout


class TestDenseLayer(TestCase):

    def setUp(self):

        self.dataset = Dataset.from_random(600, 100)

    def test_forward_propagation(self):

        dropout_layer = Dropout(probability=0.2)
        dropout_layer.set_input_shape((self.dataset.X.shape[1], ))

        # test forward propagation in training mode
        training_mode_output = dropout_layer.forward_propagation(self.dataset.X, training=True)
        # check if the shape of the output matches the input shape
        self.assertEqual(training_mode_output.shape, self.dataset.X.shape)
        # check if some values are dropped out (set to 0)
        self.assertTrue(np.any(training_mode_output == 0))
        # ensure scaling is applied correctly
        scaling_factor = 1 / (1 - dropout_layer.probability)
        scaled_input = self.dataset.X * dropout_layer.mask * scaling_factor
        self.assertTrue(np.allclose(training_mode_output, scaled_input))

        # test forward propagation in inference mode
        inference_mode_output = dropout_layer.forward_propagation(self.dataset.X, training=False)
        # check if the output matches the input exactly in inference mode
        self.assertTrue(np.array_equal(inference_mode_output, self.dataset.X))

    def test_backward_propagation(self):
        dropout_layer = Dropout(probability=0.2)
        dropout_layer.set_input_shape((self.dataset.X.shape[1], ))
        dropout_layer.forward_propagation(self.dataset.X, training=True)
        output_error = np.random.random((self.dataset.X.shape))
        input_error = dropout_layer.backward_propagation(output_error)
        # check if the shape of the input error matches the shape of the output error
        self.assertEqual(input_error.shape, output_error.shape)
        # check if the input error is scaled correctly by the mask
        expected_error = output_error * dropout_layer.mask
        self.assertTrue(np.allclose(input_error, expected_error))
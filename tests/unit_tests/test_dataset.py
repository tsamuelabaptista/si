import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

    def test_dropna(self):

        X = np.array([[np.nan, 1.0, 1.0], [1.0, 1.0, 1.0]])
        y = np.array([1, 2])
        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        original_shape_x = dataset.X.shape[0]
        original_shape_y = dataset.y.shape[0]
        dataset.dropna()

        self.assertGreater(original_shape_x, dataset.X.shape[0]) # Ensure original dataset X shape is greater than new dataset X shape
        self.assertGreater(original_shape_y, dataset.y.shape[0]) # Ensure original dataset y shape is greater than new dataset y shape  
        self.assertEqual(dataset.X.shape[0], dataset.y.shape[0]) # Ensure shape of X consistency with shape of y
        self.assertFalse(np.isnan(dataset.X).any()) # Ensure the dataset has no NaNs

    def test_fillna(self):
        X = np.array([[np.nan, 1.0, 1.0], [1.0, 1.0, 1.0]])
        y = np.array([1, 2])
        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        # Test fillna with a specified value
        value = 1.0
        dataset.fillna(value=value)
        self.assertTrue(np.allclose(dataset.X[~np.isnan(X)], X[~np.isnan(X)])) # Ensure original values are unchanged
        nan_positions = np.isnan(X) # Get positions of NaNs in the original dataset
        self.assertTrue(np.all(dataset.X[nan_positions] == value)) # Ensure all NaN positions were replaced with the specified value
        self.assertFalse(np.isnan(dataset.X).any()) # Ensure the dataset has no NaNs
    
        dataset = Dataset(X, y, features, label)

        # Test fillna with mean
        col_means = np.nanmean(X, axis=0)
        dataset.fillna(value='mean')
        self.assertTrue(np.allclose(dataset.X[~np.isnan(X)], X[~np.isnan(X)])) # Ensure original values are unchanged
        self.assertTrue(np.allclose(dataset.X[np.isnan(X)], col_means[np.isnan(X).any(axis=0)])) # Ensure NaN replaced with means

        dataset = Dataset(X, y, features, label)

        # Test fillna with median
        col_medians = np.nanmedian(X, axis=0)
        dataset.fillna(value='median')
        self.assertTrue(np.allclose(dataset.X[~np.isnan(X)], X[~np.isnan(X)])) # Ensure original values are unchanged
        self.assertTrue(np.allclose(dataset.X[np.isnan(X)], col_medians[np.isnan(X).any(axis=0)])) # Ensure NaN replaced with medians

    def test_remove_by_index(self):

        X = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        y = np.array([1, 2])
        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        original_shape_x = dataset.X.shape[0]
        original_shape_y = dataset.y.shape[0]
        
        index = 0
        expected_X = np.delete(X, index, axis=0)
        expected_y = np.delete(y, index)

        dataset.remove_by_index(index=index)

        self.assertEqual(original_shape_x - 1, dataset.X.shape[0]) # Ensure removal of exactly one row on X
        self.assertEqual(original_shape_y - 1, dataset.y.shape[0]) # Ensure removal of exactly one row on y
        self.assertEqual(dataset.X.shape[0], dataset.y.shape[0]) # Ensure shape of X is consistent with shape of y
        self.assertTrue(np.array_equal(dataset.X, expected_X) and np.array_equal(dataset.y, expected_y)) # Ensure the dataset matches the expected results

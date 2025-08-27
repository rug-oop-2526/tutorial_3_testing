import unittest


import numpy as np


from preprocessing.min_max_scaler import MinMaxScaler


class TestMinMaxScaler(unittest.TestCase):
    def test_scale_correctly(self):
        """Test fitting and transforming on standard data."""
        scaler = MinMaxScaler()
        data = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
        expected_output = np.array([
            [0, 0], [0.25, 0.25], [0.5, 0.5], [1.0, 1.0]
        ])

        scaler.fit(data)
        transformed = scaler.transform(data)

        np.testing.assert_allclose(transformed, expected_output)

    def test_scale_single_instance(self):
        """Test scaling a single 1D array."""
        scaler = MinMaxScaler()
        data = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
        single_instance = np.array([-0.5, 6])
        expected_output = np.array([0.25, 0.25])

        scaler.fit(data)
        transformed = scaler.transform(single_instance)

        self.assertEqual(transformed.shape, (2,))
        np.testing.assert_allclose(transformed, expected_output)

    def test_fit_and_transform_constant_features_with_default_fill_value(self):
        """
        Test that features with zero range are scaled to the default fill_value (0.0).
        """
        scaler = MinMaxScaler()
        data = np.array([[1, 1], [1, 1]])
        transform_data = np.array([[1, 1], [2, 2]])
        expected_output = np.array([[0, 0], [0, 0]])

        scaler.fit(data)
        transformed = scaler.transform(transform_data)

        np.testing.assert_allclose(transformed, expected_output)

    def test_fit_and_transform_constant_features_with_custom_fill_value(self):
        """
        Test that features with zero range are scaled to a custom fill_value.
        """
        scaler = MinMaxScaler(fill_value=0.5)
        data = np.array([[1, 1], [1, 1]])
        transform_data = np.array([[1, 1], [2, 2]])
        expected_output = np.array([[0.5, 0.5], [0.5, 0.5]])
        
        scaler.fit(data)
        transformed = scaler.transform(transform_data)
        
        np.testing.assert_allclose(transformed, expected_output)

    def test_raises_value_error_wrong_dim(self):
        """Test that fit raises a ValueError for a 3D array."""
        scaler = MinMaxScaler()
        with self.assertRaises(ValueError):
            scaler.fit(np.array([[[1.]]]))

    def test_raises_runtime_error_not_fitted(self):
        """Test that transform raises a RuntimeError before being fitted."""
        scaler = MinMaxScaler()
        with self.assertRaises(RuntimeError):
            scaler.transform(np.array([1.]))

    def test_raises_value_error_wrong_num_features(self):
        """Test that transform raises a ValueError for a feature count mismatch."""
        scaler = MinMaxScaler()
        scaler.fit(np.array([[1, 1], [1, 1]]))
        
        with self.assertRaises(ValueError):
            scaler.transform(np.array([1.]))


if __name__ == "__main__":
    unittest.main()
import unittest


import numpy as np


from preprocessing.mean_inputer import MeanImputer


class TestMeanImputer(unittest.TestCase):
    def test_fit_and_transform_standard(self):
        """Test fitting and transforming on standard data with NaNs."""
        imputer = MeanImputer()
        data = np.array([
            [1.0, 2.0, np.nan],
            [2.0, np.nan, 3.0],
            [np.nan, 4.0, 5.0]
        ])
        expected_output = np.array([
            [1.0, 2.0, 4.0],
            [2.0, 3.0, 3.0],
            [1.5, 4.0, 5.0]
        ])
        
        imputer.fit(data)
        transformed_data = imputer.transform(data)
        
        np.testing.assert_array_almost_equal(transformed_data, expected_output)

    def test_fit_and_transform_no_nans(self):
        """Test that data with no NaNs remains unchanged."""
        imputer = MeanImputer()
        data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        imputer.fit(data)
        transformed_data = imputer.transform(data)
        
        np.testing.assert_array_equal(transformed_data, data)

    def test_fit_and_transform_all_nans_with_fill_value(self):
        """
        Test handling of a feature with all NaN values using a specific fill_value.
        """
        imputer = MeanImputer(fill_value=0.0)
        data = np.array([
            [1.0, np.nan, 3.0],
            [2.0, np.nan, 4.0],
            [3.0, np.nan, 5.0]
        ])
        expected_output = np.array([
            [1.0, 0.0, 3.0],
            [2.0, 0.0, 4.0],
            [3.0, 0.0, 5.0]
        ])

        imputer.fit(data)
        transformed_data = imputer.transform(data)
        
        np.testing.assert_array_equal(transformed_data, expected_output)

    def test_fit_and_transform_all_nans_without_fill_value(self):
        """
        Test handling of all-NaN features when fill_value is None.
        The default behavior should be to leave NaNs as NaNs.
        """
        imputer = MeanImputer(fill_value=None)
        data = np.array([
            [1.0, np.nan, 3.0],
            [2.0, np.nan, 4.0],
            [3.0, np.nan, 5.0]
        ])
        expected_output = np.array([
            [1.0, np.nan, 3.0],
            [2.0, np.nan, 4.0],
            [3.0, np.nan, 5.0]
        ])

        imputer.fit(data)
        transformed_data = imputer.transform(data)
        
        np.testing.assert_array_equal(transformed_data, expected_output)

    def test_fit_and_transform_with_zero_mean_data(self):
        """Test imputer with data where the mean is zero."""
        imputer = MeanImputer()
        data = np.array([
            [-1.0, 10.0],
            [np.nan, np.nan],
            [1.0, np.nan]
        ])
        expected_output = np.array([
            [-1.0, 10.0],
            [0.0, 10.0],
            [1.0, 10.0]
        ])
        
        imputer.fit(data)
        transformed_data = imputer.transform(data)
        
        np.testing.assert_array_equal(transformed_data, expected_output)

    def test_transform_before_fit(self):
        """Test that transform raises a RuntimeError before being fitted."""
        imputer = MeanImputer()
        data = np.array([[1.0, 2.0]])
        with self.assertRaises(RuntimeError):
            imputer.transform(data)

    def test_invalid_num_features_transform(self):
        """Test ValueError for mismatched feature counts."""
        imputer = MeanImputer()
        imputer.fit(np.zeros((10, 5)))
        
        data = np.zeros((5, 3))
        with self.assertRaises(ValueError):
            imputer.transform(data)


if __name__ == "__main__":
    unittest.main()
import unittest
from pyfakefs.fake_filesystem_unittest import TestCase
import numpy as np
from unittest.mock import Mock, patch
import os


from preprocessing.numeric_cleaner import NumericCleaner


class TestNumericCleaner(TestCase):
    def setUp(self) -> None:
        self.setUpClassPyfakefs()

        self.save_path = os.path.join('fakedir', 'dump.pkl')

        self.mk_imputer = Mock(spec_set=['fit', 'transform', 'is_fitted'])
        self.mk_scaler  = Mock(spec_set=['fit', 'transform', 'is_fitted'])

        self.mk_imputer.fit.side_effect = lambda _: self.mk_imputer
        self.mk_scaler.fit.side_effect  = lambda _: self.mk_scaler
        self.mk_imputer.transform.return_value = np.array([[0.0, 2.0]])
        self.mk_scaler.transform.return_value  = np.array([[0.5, 1.0]])
        self.mk_imputer.is_fitted.return_value = True
        self.mk_scaler.is_fitted.return_value  = True

        self.cleaner = NumericCleaner(self.mk_imputer, self.mk_scaler)

    def test_fit_sequential_calls(self):
        """
        Tests that fit() and transform() call the dependencies in the correct order.
        """

        data = np.array([[np.nan, 2.0], [3.0, np.nan]])

        # Test the fit method
        self.cleaner.fit(data)

        # Test methods are called with correct data and order
        self.mk_imputer.fit.assert_called_once_with(data)
        self.mk_scaler.fit.assert_called_once_with(self.mk_imputer.transform.return_value)

    def test_transform_sequential_calls(self):
        """
        Tests that fit() and transform() call the dependencies in the correct order.
        """

        data = np.array([[np.nan, 2.0], [3.0, np.nan]])

        result = self.cleaner.transform(data)

        # Test methods are called with correct data and order
        self.mk_imputer.transform.assert_called_once_with(data)
        self.mk_scaler.transform.assert_called_once_with(self.mk_imputer.transform.return_value)

        # Test return value corresponds to scaler transform return value
        np.testing.assert_allclose(result, self.mk_scaler.transform.return_value)

    def test_call_method_correct_flow(self):
        """
        Tests that the __call__ method correctly calls fit() and transform().
        """
        data = np.array([[np.nan, 2.0], [3.0, np.nan]])
        result = self.cleaner(data)

        self.mk_imputer.fit.assert_called_once_with(data)
        self.mk_imputer.transform.assert_called_with(data)
        self.mk_scaler.fit.assert_called_once_with(self.mk_imputer.transform.return_value)
        self.mk_scaler.transform.assert_called_with(self.mk_imputer.transform.return_value)

        np.testing.assert_allclose(result, self.mk_scaler.transform.return_value)
    

    def test_save_opens_in_binary_and_calls_pickle_dump_with_self(self):
        self.fs.makedirs('fakedir', exist_ok=True)

        ret = object()

        with patch('pickle.dump', return_value=ret) as save_mock:
            self.cleaner.save(self.save_path)
            assert os.path.exists(self.save_path)
            save_mock.assert_called_once()


    def test_load_opens_in_binary_and_returns_result_of_pickle_load(self):
        self.fs.makedirs('fakedir', exist_ok=True)
        
        obj = object()

        self.fs.create_file(self.save_path)

        with patch("pickle.load", return_value=obj) as load_mock:
            ret = NumericCleaner.load(self.save_path)
            load_mock.assert_called_once()
            assert obj is ret

if __name__ == "__main__":
    unittest.main()
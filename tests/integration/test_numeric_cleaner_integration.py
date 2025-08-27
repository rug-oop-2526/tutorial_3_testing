import os
import unittest
import tempfile
import numpy as np

from preprocessing.numeric_cleaner import NumericCleaner
from preprocessing.mean_inputer import MeanImputer
from preprocessing.min_max_scaler import MinMaxScaler


class TestNumericCleanerIntegration(unittest.TestCase):
    def test_expected_scaled_values(self):
        """
        Verifies that for a known input, the cleaner produces exactly the expected scaled values.
        """
        X = np.array([
            [1.0, np.nan],
            [3.0, 5.0],
            [np.nan, 7.0]
        ], dtype=float)

        # Mean imputation:
        # Col 0 mean = (1+3)/2 = 2.0  -> imputes NaN in row 2
        # Col 1 mean = (5+7)/2 = 6.0  -> imputes NaN in row 0
        # After imputation:
        # [[1.0, 6.0],
        #  [3.0, 5.0],
        #  [2.0, 7.0]]
        #
        # Min-max scaling per column:
        # Col 0 min=1, range=2  -> [0.0, 1.0, 0.5]
        # Col 1 min=5, range=2  -> [0.5, 0.0, 1.0]
        expected = np.array([
            [0.0, 0.5],
            [1.0, 0.0],
            [0.5, 1.0]
        ])

        Xt = NumericCleaner.from_defaults(fill_value=None).fit(X).transform(X)

        np.testing.assert_allclose(Xt, expected, rtol=1e-9, atol=1e-9)

    def test_end_to_end_fit_transform_basic(self):
        X = np.array([
            [1.0,    np.nan, 10.0],
            [3.0,    5.0,    20.0],
            [np.nan, 7.0,    30.0],
        ], dtype=float)

        cleaner = NumericCleaner(
            imputer=MeanImputer(fill_value=None),
            scaler=MinMaxScaler(fill_value=0.0),
        ).fit(X)

        Xt = cleaner.transform(X)

        self.assertEqual(Xt.shape, X.shape)
        self.assertTrue(np.isfinite(Xt).all())
        # Values should be in [0, 1] per feature (non-zero range)
        self.assertTrue(((Xt.min(axis=0) >= 0.0) & (Xt.max(axis=0) <= 1.0)).all())

    def test_zero_range_feature_scaled_to_fill_value(self):
        # Second column constant -> range zero
        X = np.array([
            [1.0, 42.0],
            [3.0, 42.0],
            [5.0, 42.0],
        ], dtype=float)

        fill = 0.25
        cleaner = NumericCleaner(
            imputer=MeanImputer(fill_value=None),
            scaler=MinMaxScaler(fill_value=fill),
        ).fit(X)

        Xt = cleaner.transform(X)

        self.assertEqual(Xt.shape, X.shape)
        self.assertEqual(float(Xt[:, 0].min()), 0.0)
        self.assertEqual(float(Xt[:, 0].max()), 1.0)
        np.testing.assert_allclose(Xt[:, 1], np.full(X.shape[0], fill))

    def test_all_nan_column_imputed_then_zero_range_scaled_to_given_fill_value(self):
        # Last column all NaN -> imputed to fill, becomes constant -> scaled to scaler fill
        X = np.array([
            [1.0, 2.0, np.nan],
            [2.0, 3.0, np.nan],
            [3.0, 4.0, np.nan],
        ], dtype=float)

        fill = 5.0
        cleaner = NumericCleaner.from_defaults(fill_value=fill).fit(X)
        Xt = cleaner.transform(X)

        np.testing.assert_allclose(Xt[:, 2], np.full(X.shape[0], fill))
        self.assertEqual(float(Xt[:, 0].min()), 0.0)
        self.assertEqual(float(Xt[:, 0].max()), 1.0)
        self.assertEqual(float(Xt[:, 1].min()), 0.0)
        self.assertEqual(float(Xt[:, 1].max()), 1.0)

    def test_call_equivalent_to_fit_then_transform(self):
        X = np.array([
            [np.nan, 2.0],
            [3.0,    5.0],
            [4.0,    np.nan],
        ], dtype=float)

        c1 = NumericCleaner.from_defaults(fill_value=0.0)
        y_call = c1(X)

        c2 = NumericCleaner.from_defaults(fill_value=0.0)
        y_manual = c2.fit(X).transform(X)

        np.testing.assert_allclose(y_call, y_manual)

    def test_transform_before_fit_raises_runtime_error(self):
        X = np.array([[1.0, 2.0]], dtype=float)
        cleaner = NumericCleaner.from_defaults(fill_value=0.0)
        with self.assertRaises(RuntimeError) as ctx:
            cleaner.transform(X)
        self.assertIn("must be fit() before calling", str(ctx.exception))

    def test_mismatched_feature_count_raises_value_error(self):
        X_train = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]], dtype=float)
        cleaner = NumericCleaner.from_defaults(fill_value=0.0).fit(X_train)

        X_bad = np.array([[1.0, 2.0]], dtype=float)
        with self.assertRaises(ValueError):
            cleaner.transform(X_bad)

    def test_pickle_round_trip_preserves_behavior(self):
        X = np.array([
            [1.0, np.nan, 10.0],
            [3.0, 5.0,    20.0],
            [7.0, 9.0,    30.0],
        ], dtype=float)

        cleaner = NumericCleaner.from_defaults(fill_value=0.0).fit(X)
        Xt = cleaner.transform(X)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cleaner.pkl")
            cleaner.save(path)
            reloaded = NumericCleaner.load(path)
            Xt2 = reloaded.transform(X)

        np.testing.assert_allclose(Xt, Xt2, rtol=1e-7, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
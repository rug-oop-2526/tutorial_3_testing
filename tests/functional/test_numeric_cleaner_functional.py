import os
import tempfile
import unittest
import numpy as np

from preprocessing.numeric_cleaner import NumericCleaner


class TestNumericCleanerFunctional(unittest.TestCase):
    def test_roundtrip_produces_expected_output(self):
        X = np.array([
            [1.0,   np.nan],
            [3.0,   5.0   ],
            [np.nan, 7.0  ],
        ])

        # Expected:
        # Imputation (means): col0 mean=(1+3)/2=2 -> [1,3,2]; col1 mean=(5+7)/2=6 -> [6,5,7]
        # Scaling per column to [0,1]:
        # col0 min=1, max=3 -> [0.0, 1.0, 0.5]
        # col1 min=5, max=7 -> [0.5, 0.0, 1.0]
        expected = np.array([
            [0.0, 0.5],
            [1.0, 0.0],
            [0.5, 1.0],
        ])

        # Build cleaner with deterministic defaults (MeanImputer(fill=None), MinMaxScaler(fill=0))
        cleaner = NumericCleaner.from_defaults(fill_value=None).fit(X)

        # Transform before saving (sanity check)
        Xt_before = cleaner.transform(X)
        np.testing.assert_allclose(Xt_before, expected, rtol=1e-9, atol=1e-9)

        # Example of NOT using pyfake fs (more verbose and complex)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cleaner.pkl")
            cleaner.save(path)
            self.assertTrue(os.path.exists(path), "Pickle file not created")

            reloaded = NumericCleaner.load(path)
            Xt_after = reloaded.transform(X)

        # Compare reloaded output to both expected and pre-save output
        np.testing.assert_allclose(Xt_after, expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(Xt_after, Xt_before, rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
import numpy as np
import pickle

from preprocessing.mean_inputer import MeanImputer
from preprocessing.min_max_scaler import MinMaxScaler
from preprocessing.preprocess import Preprocess

class NumericCleaner:
    """
    A composite transformer that first imputes missing values and then scales numeric features.

    This class chains a `MeanImputer` and a `MinMaxScaler` to provide a single,
    convenient interface for common numeric data cleaning tasks. The two transformers
    are fitted sequentially on the training data and then applied in the same order
    to new data.
    """
    _imputer: Preprocess
    _scaler:  Preprocess

    def __init__(self, imputer: Preprocess, scaler: Preprocess) -> None:
        self._imputer = imputer
        self._scaler  = scaler

    @classmethod
    def from_defaults(cls, fill_value: float | None = None) -> 'NumericCleaner':
        return cls(
            MeanImputer(fill_value),
            MinMaxScaler(fill_value if fill_value is not None else 0)
        )

    def fit(self, data: np.ndarray) -> 'NumericCleaner':
        """
        Fits the imputer and scaler sequentially on the input data.

        The imputer is fitted on the original data, and the scaler is then
        fitted on the data after it has been imputed. This ensures the scaler
        learns from clean data.

        :param data: The training data to fit the cleaner on, with shape (n_samples, n_features).
        :returns: The fitted cleaner instance.
        """
        transformed = self._imputer.fit(data).transform(data)
        self._scaler.fit(transformed)

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Sequentially applies the fitted imputer and scaler to the data.

        The transformations are applied in the same order as they were fitted.

        :param data: The array to be transformed.
        :returns: The transformed NumPy array.
        :raises RuntimeError: If the cleaner has not been fitted before being called.
        """
        try:
            out = self._imputer.transform(data)
            out = self._scaler.transform(out)
        except RuntimeError:
            raise RuntimeError(f'{self.__class__.__name__} must be fit() before calling.')
        
        return out

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        A convenience method that performs a combined fit and transform.

        This method first fits the cleaner to the data and then returns the
        transformed result.

        :param data: The array to be fitted and transformed.
        :returns: The transformed NumPy array.
        """
        self.fit(data)

        return self.transform(data)
    
    def save(self, file_path: str):
        """
        Saves the fitted cleaner object to a file.

        :param file_path: The path to the file where the cleaner will be saved.
        """
        if not self._imputer.is_fitted() or not self._scaler.is_fitted():
            raise RuntimeError('The cleaner must be fitted before saving.')
        
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> 'NumericCleaner':
        """
        Loads a fitted cleaner object from a file.

        :param file_path: The path to the file to load the cleaner from.
        :returns: The loaded NumericCleaner instance.
        """
        with open(file_path, 'rb') as f:
            cleaner = pickle.load(f)
        return cleaner

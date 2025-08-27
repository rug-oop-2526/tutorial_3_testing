from abc import ABC, abstractmethod

import numpy as np

class Transform(ABC):
    _requires_fit: bool = False
    _is_fitted: bool = False

    def fit(self, data: np.ndarray) -> 'Transform':
        """
        Validates the input data and prepares the transform.

        This base implementation validates that the input is a 2D array.
        It should be called by subclasses to ensure consistency.

        :param data: The input array for validation, of shape (n_samples, n_features).
        :returns: self.
        :raises ValueError: if the input data is not a 2D NumPy array.
        """
        if data.ndim != 2:
            raise ValueError('fit expects a 2D array of shape (n_samples, n_features).')
        
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transforms the input data using the fitted transformation.

        This method handles pre-conditions and shape validation, then delegates
        the actual transformation to the `_apply_transform` method. It
        automatically handles 1D arrays by reshaping them to 2D before
        passing them to the core transformation logic.

        :param data: The array to be transformed. Can be 1D (n_features) or 2D (n_samples, n_features).
        :returns: The transformed array with the same dimensions as `data`.
        :raises RuntimeError: if the transformer requires fitting and has not been fitted.
        :raises ValueError: if the input data has more than two dimensions.
        """
        if self._requires_fit and not self._is_fitted:
            raise RuntimeError(f'{self.__class__.__name__} must be fit() before calling.')
        
        if data.ndim > 2:
            raise ValueError("__call__ expects 1D or 2D array input.")
        
        if data.ndim == 1:
            out = self._apply_transform(data[np.newaxis, :])
            return out[0]
        
        return self._apply_transform(data)
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Applies fit and transform.

        This allows the transformer to be called directly, e.g., `scaler(data)`.
        It first checks if the transformer requires fitting and, if so, calls
        the `fit` method before calling `transform`.

        :param data: The array to be transformed.
        :returns: The transformed array.
        """
        if self._requires_fit:
            self.fit(data)
        
        return self.transform(data)
    
    def is_fitted(self) -> bool:
        return not self._requires_fit or self._is_fitted
    
    def _raise_invalid_num_features(self, expected: int, received: int) -> None:
        """
        Raises a ValueError for a feature count mismatch.

        This is a helper method to provide a consistent and descriptive error message
        when the number of features in the input data does not match the number of
        features the transformer was fitted on.

        :param expected: The number of features the transformer was fitted on.
        :param received: The number of features in the new input data.
        :raises ValueError: Always raises a ValueError with a formatted message.
        """
        raise ValueError(f'invalid number of features, expected {expected} but received {received}')
    
    @abstractmethod
    def _apply_transform(self, data: np.ndarray) -> np.ndarray:
        """Implement the 2D transform (n_samples, n_features)."""
        ...


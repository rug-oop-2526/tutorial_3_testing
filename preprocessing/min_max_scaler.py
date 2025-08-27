import numpy as np


from preprocessing.transform import Transform


class MinMaxScaler(Transform):
    """
    A transformer that scales features to a given range, typically [0, 1].

    This scaler calculates the minimum and range (max - min) for each feature
    in the training data during the `fit` stage. During the `transform` stage,
    it scales new data based on these fitted values. Features with a zero range
    are handled by being scaled to a user-defined fill value, which defaults to 0.
    """
    _requires_fit: bool = True

    _min_vals: np.ndarray | None
    _rng_vals: np.ndarray | None

    _fill_value: float

    
    def __init__(self, fill_value: float = 0.):
        """
        Initializes the MinMaxScaler.

        :param fill_value: The value to which features with a zero range will be
                           scaled. Defaults to 0.0.
        """
        self._min_vals = None
        self._rng_vals = None

        self._fill_value = fill_value

    def fit(self, data: np.ndarray) -> 'MinMaxScaler':
        """
        Calculates the per-feature minimum and range for scaling.

        This method fits the scaler to the provided data by computing
        the minimum value and the range (max - min) for each feature along axis 0.
        These values are stored internally and used later for transforming data.

        :param data: A NumPy array of shape (n_samples, n_features).
        :returns: The fitted scaler instance.
        :raises ValueError: if the input data does not have two dimensions.
        """
        super().fit(data)
        
        self._min_vals = np.min(data, axis=0)
        self._rng_vals = np.max(data, axis=0) - self._min_vals
        
        self._is_fitted = True

        return self

    def _apply_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the fitted Min-Max scaling to a NumPy array.

        This method scales the input data using the minimum and range values
        calculated during the fit stage. Features with a range of zero will be
        scaled to the `_fill_value`.

        :param data: A NumPy array of shape (n_samples, n_features) to be scaled.
        :returns: A new NumPy array with the same shape as the input, containing the scaled data.
        :raises ValueError: if the number of features in the input data does not
                            match the number of features the scaler was fitted on.
        """
        assert self._min_vals is not None
        assert self._rng_vals is not None

        if data.shape[1] != self._min_vals.shape[0]:
            self._raise_invalid_num_features(self._min_vals.shape[0], data.shape[1])
        
        out = np.full_like(data, fill_value=self._fill_value, dtype=float)

        np.divide(data - self._min_vals, self._rng_vals, where=self._rng_vals != 0, out=out)

        return out
    
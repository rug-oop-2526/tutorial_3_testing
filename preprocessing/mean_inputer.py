import numpy as np


from preprocessing.preprocess import Preprocess

    
class MeanImputer(Preprocess):
    """
    A transformer that imputes missing values (NaNs) using the mean of each feature.

    This imputer calculates the mean for each feature (column) in the training data,
    while ignoring any missing values (`NaNs`). During the transform step, it
    replaces any `NaNs` in the input data with the corresponding feature means.
    For features that contain only `NaN`s, the missing values are replaced by a
    user-defined fill value, which defaults to `np.nan`.
    """
    _requires_fit: bool = True

    _means: np.ndarray | None

    _fill_value: float | None
    
    def __init__(self, fill_value: float | None = None):
        """
        Initializes the MeanImputer.

        :param fill_value: The value to use for imputing features that contain
                           only NaNs. If `None`, those features will remain as NaNs.
                           Defaults to `None`.
        """
        self._means = None
        self._fill_value = fill_value

    def fit(self, X: np.ndarray) -> 'MeanImputer':
        """
        Calculates the mean of each feature from the input data.

        :param data: The training data to fit the imputer on, with shape (n_samples, n_features).
        :returns: The fitted imputer instance.
        :raises ValueError: if the input data does not have two dimensions.
        """
        super().fit(X)

        means = np.nanmean(X, axis=0)

        if self._fill_value is None:
            self._means = means
        else:
            self._means = np.where(np.isnan(means), self._fill_value, means)

        self._is_fitted = True

        return self

    def _apply_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Replaces missing values (NaNs) in the data with the fitted feature means.

        This method performs the core imputation logic, replacing `NaN` values
        in the input array with the corresponding pre-calculated feature means.

        :param data: The array to be imputed, with shape (n_samples, n_features).
        :returns: A new NumPy array with missing values replaced by the means.
        :raises ValueError: if the number of features in the input data does
                            not match the number of features the imputer was fitted on.
        """
        assert self._means is not None

        if X.shape[1] != self._means.shape[0]:
            self._raise_invalid_num_features(self._means[0], X.shape[1])
        
        y = np.where(np.isnan(X), self._means, X)
        
        return y
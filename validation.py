"""Input validation helpers shared across adcheck.

These are plain functions, not classes, and none of them start with a
leading underscore. "Internal" is communicated by what is exported from
``adcheck.__init__``, not by naming convention.
"""

from __future__ import annotations

import warnings

import numpy as np

from .exceptions import NotFittedError


def check_array(X, name: str = "X") -> np.ndarray:
    """Validate and convert input to a 2D float64 numpy array.

    Parameters
    ----------
    X : array-like
        Input data to validate.
    name : str, default="X"
        Name to use in error messages.

    Returns
    -------
    numpy.ndarray
        Validated array of shape (n_samples, n_features) and dtype
        float64.

    Raises
    ------
    ValueError
        If X cannot be converted to a numeric array, is not
        2-dimensional, is empty, or contains NaN or infinite values.
    """
    try:
        X = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} could not be converted to a numeric array: {exc}") from exc

    if X.ndim == 1:
        raise ValueError(
            f"{name} must be a 2D array of shape (n_samples, n_features), "
            f"got a 1D array of shape {X.shape}. Reshape it first, for "
            f"example X.reshape(-1, 1) for a single feature or "
            f"X.reshape(1, -1) for a single sample."
        )
    if X.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D array of shape (n_samples, n_features), "
            f"got an array with {X.ndim} dimensions."
        )
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must not be empty, got shape {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError(
            f"{name} contains NaN or infinite values, which are not "
            f"supported. Remove or impute them before calling this method."
        )
    return X


def check_n_features(X: np.ndarray, n_features_expected: int, name: str = "X") -> None:
    """Check that X has the expected number of features.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
    n_features_expected : int
    name : str, default="X"
        Name to use in the error message.

    Raises
    ------
    ValueError
        If X.shape[1] does not equal n_features_expected.
    """
    if X.shape[1] != n_features_expected:
        raise ValueError(
            f"{name} has {X.shape[1]} feature(s), but this estimator was "
            f"fitted with {n_features_expected} feature(s)."
        )


def check_is_fitted(estimator, attributes: list[str]) -> None:
    """Raise NotFittedError if any of the given attributes are missing.

    Parameters
    ----------
    estimator : object
        The estimator instance to check.
    attributes : list of str
        Attribute names that should be present after fitting.

    Raises
    ------
    NotFittedError
        If any attribute in ``attributes`` is missing from ``estimator``.
    """
    missing = [a for a in attributes if not hasattr(estimator, a)]
    if missing:
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet. "
            f"Call 'fit' with appropriate training data before using "
            f"this method."
        )


def warn_on_constant_columns(X: np.ndarray, name: str = "X") -> None:
    """Emit a UserWarning if any column of X has zero variance.

    A constant column does not break the leverage or kNN-distance
    calculations (a pseudo-inverse and ordinary distances both remain
    well-defined), but it usually signals a redundant feature worth
    knowing about.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
    name : str, default="X"
        Name to use in the warning message.
    """
    stds = X.std(axis=0)
    constant_idx = np.where(stds == 0)[0]
    if constant_idx.size > 0:
        warnings.warn(
            f"{name} has {constant_idx.size} constant (zero-variance) "
            f"column(s) at index/indices {constant_idx.tolist()}. This "
            f"does not break the calculation, a pseudo-inverse is used, "
            f"but it usually indicates a redundant feature.",
            UserWarning,
            stacklevel=3,
        )

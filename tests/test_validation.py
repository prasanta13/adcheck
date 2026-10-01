"""Tests for adcheck.validation."""

import numpy as np
import pytest

from adcheck.exceptions import NotFittedError
from adcheck.validation import (
    check_array,
    check_is_fitted,
    check_n_features,
    warn_on_constant_columns,
)


def test_check_array_accepts_valid_input():
    X = check_array([[1.0, 2.0], [3.0, 4.0]])
    assert X.shape == (2, 2)
    assert X.dtype == np.float64


def test_check_array_rejects_non_numeric():
    with pytest.raises(ValueError, match="could not be converted"):
        check_array([["a", "b"], ["c", "d"]])


def test_check_array_rejects_1d():
    with pytest.raises(ValueError, match="2D array"):
        check_array([1.0, 2.0, 3.0])


def test_check_array_rejects_3d():
    with pytest.raises(ValueError, match="2 dimensions|2D array"):
        check_array(np.zeros((2, 2, 2)))


def test_check_array_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        check_array(np.zeros((0, 3)))


def test_check_array_rejects_nan():
    with pytest.raises(ValueError, match="NaN or infinite"):
        check_array([[1.0, np.nan], [3.0, 4.0]])


def test_check_array_rejects_inf():
    with pytest.raises(ValueError, match="NaN or infinite"):
        check_array([[1.0, np.inf], [3.0, 4.0]])


def test_check_n_features_mismatch():
    X = np.zeros((5, 3))
    with pytest.raises(ValueError, match="feature"):
        check_n_features(X, n_features_expected=4)


def test_check_n_features_match_does_not_raise():
    X = np.zeros((5, 3))
    check_n_features(X, n_features_expected=3)  # should not raise


def test_check_is_fitted_raises_when_missing():
    class Dummy:
        pass

    with pytest.raises(NotFittedError):
        check_is_fitted(Dummy(), ["some_attr_"])


def test_check_is_fitted_passes_when_present():
    class Dummy:
        some_attr_ = 1

    check_is_fitted(Dummy(), ["some_attr_"])  # should not raise


def test_warn_on_constant_columns_warns():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    with pytest.warns(UserWarning, match="constant"):
        warn_on_constant_columns(X)


def test_warn_on_constant_columns_silent_when_none():
    X = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
    with warnings_as_errors():
        warn_on_constant_columns(X)  # should not raise/warn


class warnings_as_errors:
    """Small context manager so the "no warning" test actually fails
    if a warning is raised, instead of silently passing either way.
    """

    def __enter__(self):
        import warnings

        self._catch = warnings.catch_warnings()
        self._catch.__enter__()
        warnings.simplefilter("error")
        return self

    def __exit__(self, *exc_info):
        self._catch.__exit__(*exc_info)

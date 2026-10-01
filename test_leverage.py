"""Tests for adcheck.leverage.

test_hand_worked_example derives its expected values explicitly in the
comments below rather than by trusting a second computation, per the
project's testing policy.
"""

import numpy as np
import pytest

from adcheck.leverage import compute_leverage, leverage_threshold


def test_hand_worked_example(plus_shape_X_train):
    """Leverage on a symmetric 4-point "plus" shape, derived by hand.

    X_train = [[ 1, 0], [0, 1], [-1, 0], [0, -1]]

    X_train.T @ X_train:
        column sums of squares and cross products, by symmetry the
        off-diagonal terms are 0 (e.g. the x*y cross term is
        1*0 + 0*1 + (-1)*0 + 0*(-1) = 0) and each diagonal term is
        1^2 + 0^2 + (-1)^2 + 0^2 = 2, giving

            X^T X = [[2, 0],
                     [0, 2]]  =  2 * I

    so (X^T X)^-1 = 0.5 * I exactly (no pseudo-inverse subtlety needed
    here, X^T X is already full rank).

    Leverage of a point x is x @ (X^T X)^-1 @ x.T = 0.5 * (x[0]**2 + x[1]**2).

    For every training point, x[0]**2 + x[1]**2 = 1 (each is a unit
    vector along an axis), so each training point has leverage
    0.5 * 1 = 0.5 exactly.

    For query (0.5, 0.5): 0.5 * (0.25 + 0.25) = 0.5 * 0.5 = 0.25
    For query (10, 0):    0.5 * (100 + 0)     = 50.0
    """
    X_train = plus_shape_X_train

    train_leverage = compute_leverage(X_train, X_train)
    np.testing.assert_allclose(train_leverage, [0.5, 0.5, 0.5, 0.5])

    X_query = np.array([[0.5, 0.5], [10.0, 0.0]])
    query_leverage = compute_leverage(X_train, X_query)
    np.testing.assert_allclose(query_leverage, [0.25, 50.0])


def test_sum_of_training_leverages_equals_n_features(plus_shape_X_train, random_X_train):
    """The diagonal of a hat matrix H = X(X^TX)^+X^T is a projection
    matrix onto the column space of X, so its trace, the sum of the
    leverages of the points that generated it, equals the rank of that
    column space. For a reference set with linearly independent
    columns, that is exactly n_features. This holds regardless of the
    specific data, unlike the hand-worked example above.
    """
    for X_train in (plus_shape_X_train, random_X_train):
        n_features = X_train.shape[1]
        train_leverage = compute_leverage(X_train, X_train)
        assert train_leverage.sum() == pytest.approx(n_features)


def test_leverage_threshold_formula():
    """h* = 3(p+1)/n, checked against values computed by hand."""
    assert leverage_threshold(n_train=4, n_features=2) == pytest.approx(2.25)
    assert leverage_threshold(n_train=100, n_features=9) == pytest.approx(0.3)
    assert leverage_threshold(n_train=10, n_features=0) == pytest.approx(0.3)


def test_leverage_handles_collinear_reference_set():
    """A genuinely singular X^T X (two identical columns, so X^T X has
    rank 1 instead of 2) must not raise, and must still produce finite,
    non-negative leverage values via the pseudo-inverse.
    """
    X_train = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [-1.0, -1.0],
        ]
    )
    leverage = compute_leverage(X_train, X_train)
    assert np.all(np.isfinite(leverage))
    assert np.all(leverage >= 0.0)


def test_leverage_handles_constant_column():
    """A constant column (zero variance) also makes X^T X singular in
    that direction; this must likewise not raise and must still
    produce finite leverage values.
    """
    X_train = np.array(
        [
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0],
            [4.0, 5.0],
        ]
    )
    leverage = compute_leverage(X_train, X_train)
    assert np.all(np.isfinite(leverage))

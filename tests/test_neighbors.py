"""Tests for adcheck.neighbors."""

import numpy as np
import pytest

from adcheck.neighbors import (
    knn_threshold_mean_std,
    knn_threshold_percentile,
    leave_one_out_distances,
    nearest_neighbour_distance,
    pairwise_euclidean,
)


def test_pairwise_euclidean_known_distances():
    A = np.array([[0.0, 0.0]])
    B = np.array([[3.0, 4.0], [0.0, 0.0]])
    D = pairwise_euclidean(A, B)
    # A 3-4-5 triangle, and the distance from a point to itself is 0.
    np.testing.assert_allclose(D, [[5.0, 0.0]])


def test_leave_one_out_distances_hand_worked(plus_shape_X_train):
    """On the "plus" shape, each point's nearest OTHER point is one of
    the two adjacent arms, at distance sqrt((1-0)**2 + (0-1)**2) =
    sqrt(2); the opposite arm is farther away, at distance 2. So every
    leave-one-out distance equals sqrt(2) exactly.
    """
    loo = leave_one_out_distances(plus_shape_X_train, k=1)
    np.testing.assert_allclose(loo, [np.sqrt(2)] * 4)


def test_nearest_neighbour_distance_hand_worked(plus_shape_X_train):
    """Query (0.5, 0.5) is distance sqrt(0.5**2 + 0.5**2) = sqrt(0.5)
    from both (1, 0) and (0, 1), its two nearest training points, and
    farther from the other two. Query (10, 0) is distance 9 from its
    nearest training point (1, 0).
    """
    X_query = np.array([[0.5, 0.5], [10.0, 0.0]])
    dist = nearest_neighbour_distance(plus_shape_X_train, X_query, k=1)
    np.testing.assert_allclose(dist, [np.sqrt(0.5), 9.0])


def test_knn_threshold_percentile_of_identical_values():
    """The 95th percentile of four identical values is just that value."""
    loo = np.array([1.4142135623730951] * 4)
    assert knn_threshold_percentile(loo, percentile=95.0) == pytest.approx(np.sqrt(2))


def test_knn_threshold_mean_std_formula():
    loo = np.array([1.0, 2.0, 3.0])
    expected = loo.mean() + 2.0 * loo.std()
    assert knn_threshold_mean_std(loo, z=2.0) == pytest.approx(expected)


def test_k_must_not_exceed_reference_size_for_query_distance(plus_shape_X_train):
    with pytest.raises(ValueError, match="must not exceed"):
        nearest_neighbour_distance(plus_shape_X_train, plus_shape_X_train, k=5)


def test_k_must_be_smaller_than_reference_size_for_leave_one_out(plus_shape_X_train):
    with pytest.raises(ValueError, match="must be smaller than"):
        leave_one_out_distances(plus_shape_X_train, k=4)


def test_k_must_be_positive(plus_shape_X_train):
    with pytest.raises(ValueError, match="positive integer"):
        nearest_neighbour_distance(plus_shape_X_train, plus_shape_X_train, k=0)
    with pytest.raises(ValueError, match="positive integer"):
        leave_one_out_distances(plus_shape_X_train, k=0)

"""k-nearest-neighbour distance applicability domain criterion."""

from __future__ import annotations

import numpy as np


def pairwise_euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances between rows of A and rows of B.

    Parameters
    ----------
    A : numpy.ndarray of shape (n_a, n_features)
    B : numpy.ndarray of shape (n_b, n_features)

    Returns
    -------
    numpy.ndarray of shape (n_a, n_b)
        Entry [i, j] is the Euclidean distance between row i of A and
        row j of B.
    """
    a_sq = np.sum(A**2, axis=1)[:, None]
    b_sq = np.sum(B**2, axis=1)[None, :]
    cross = A @ B.T
    # Clip at 0 because floating point error can otherwise make an
    # identical pair come out as a tiny negative number under the sqrt.
    sq_dist = np.maximum(a_sq + b_sq - 2.0 * cross, 0.0)
    return np.sqrt(sq_dist)


def nearest_neighbour_distance(X_ref: np.ndarray, X_query: np.ndarray, k: int = 1) -> np.ndarray:
    """Mean distance from each query point to its k nearest reference points.

    Parameters
    ----------
    X_ref : numpy.ndarray of shape (n_ref, n_features)
    X_query : numpy.ndarray of shape (n_query, n_features)
    k : int, default=1
        Number of nearest reference points to average over. If k=1 this
        is simply the nearest-neighbour distance.

    Returns
    -------
    numpy.ndarray of shape (n_query,)

    Raises
    ------
    ValueError
        If k is not a positive integer, or k exceeds the number of
        reference samples.
    """
    n_ref = X_ref.shape[0]
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}.")
    if k > n_ref:
        raise ValueError(f"k ({k}) must not exceed the number of reference samples ({n_ref}).")

    D = pairwise_euclidean(X_query, X_ref)
    D.sort(axis=1)
    return D[:, :k].mean(axis=1)


def leave_one_out_distances(X_ref: np.ndarray, k: int = 1) -> np.ndarray:
    """For each reference point, the mean distance to its k nearest
    other reference points (itself excluded).

    This characterises how closely packed the reference set is, and is
    used to derive a data-driven kNN-distance threshold rather than
    relying on an arbitrary fixed number.

    Parameters
    ----------
    X_ref : numpy.ndarray of shape (n_ref, n_features)
    k : int, default=1

    Returns
    -------
    numpy.ndarray of shape (n_ref,)

    Raises
    ------
    ValueError
        If k is not a positive integer, or k is not smaller than the
        number of reference samples (there must be at least k other
        points to compare against).
    """
    n_ref = X_ref.shape[0]
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}.")
    if k >= n_ref:
        raise ValueError(
            f"k ({k}) must be smaller than the number of reference "
            f"samples ({n_ref}); there are not enough other points to "
            f"compute a leave-one-out distance to {k} neighbour(s)."
        )

    D = pairwise_euclidean(X_ref, X_ref)
    np.fill_diagonal(D, np.inf)
    D.sort(axis=1)
    return D[:, :k].mean(axis=1)


def knn_threshold_percentile(loo_distances: np.ndarray, percentile: float = 95.0) -> float:
    """Threshold as a percentile of the reference set's own
    leave-one-out nearest-neighbour distances.

    Parameters
    ----------
    loo_distances : numpy.ndarray
        Output of :func:`leave_one_out_distances`.
    percentile : float, default=95.0
        Percentile in the range [0, 100].

    Returns
    -------
    float
    """
    return float(np.percentile(loo_distances, percentile))


def knn_threshold_mean_std(loo_distances: np.ndarray, z: float = 3.0) -> float:
    """Threshold as mean + z standard deviations of the reference set's
    own leave-one-out nearest-neighbour distances.

    Parameters
    ----------
    loo_distances : numpy.ndarray
        Output of :func:`leave_one_out_distances`.
    z : float, default=3.0
        Number of standard deviations above the mean.

    Returns
    -------
    float
    """
    return float(np.mean(loo_distances) + z * np.std(loo_distances))

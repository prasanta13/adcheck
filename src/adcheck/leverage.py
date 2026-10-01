"""Leverage (hat-matrix) applicability domain criterion."""

from __future__ import annotations

import numpy as np


def compute_leverage(X_ref: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """Compute hat-matrix leverage of each row of X_query against X_ref.

    Leverage measures how far a point lies from the centre and spread of
    the reference set, in the metric defined by the reference set's own
    covariance structure. For a reference matrix X_ref of shape
    (n_ref, n_features), the leverage of a point x is::

        h(x) = x @ pinv(X_ref.T @ X_ref) @ x.T

    The Moore-Penrose pseudo-inverse is used instead of a plain matrix
    inverse so that collinear or rank-deficient reference data does not
    raise a numerical error; see Notes.

    Parameters
    ----------
    X_ref : numpy.ndarray of shape (n_ref, n_features)
        Reference data defining the leverage metric, typically a model's
        training set. No centering or scaling is applied internally;
        standardise beforehand if that is what you want, see the
        package README for the exact convention used here.
    X_query : numpy.ndarray of shape (n_query, n_features)
        Points to compute leverage for. Must have the same number of
        features as X_ref.

    Returns
    -------
    numpy.ndarray of shape (n_query,)
        Leverage value for each row of X_query.

    Notes
    -----
    This is the diagonal of the regression hat matrix
    H = X (X^T X)^+ X^T, generalised to query points that need not be
    part of the reference set, following the applicability domain
    literature for QSAR models (see README references: Gramatica 2007,
    Roy, Kar and Ambure 2015).

    Using the pseudo-inverse means leverage stays well-defined when
    columns of X_ref are collinear or nearly so. It is not a substitute
    for having enough reference samples relative to the number of
    features for the result to be meaningful; see
    :func:`leverage_threshold`.

    For a point that is itself a row of X_ref, the sum of all such
    leverages equals the number of features, a useful property for
    sanity-checking an implementation.
    """
    XtX_pinv = np.linalg.pinv(X_ref.T @ X_ref)
    return np.einsum("ij,jk,ik->i", X_query, XtX_pinv, X_query)


def leverage_threshold(n_train: int, n_features: int) -> float:
    """Standard warning-leverage threshold h* = 3(p+1)/n.

    This is the conventional critical leverage used in applicability
    domain work for QSAR models (Gramatica, 2007), where points with
    leverage above this value are considered statistical outliers
    relative to the reference set's own spread. p is taken as the
    number of features, with no separate allowance for an intercept
    term; see the package README for the exact convention and
    references.

    Parameters
    ----------
    n_train : int
        Number of reference (training) samples.
    n_features : int
        Number of features.

    Returns
    -------
    float
        The threshold h* = 3 * (n_features + 1) / n_train.
    """
    return 3.0 * (n_features + 1) / n_train

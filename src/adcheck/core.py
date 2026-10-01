"""The ApplicabilityDomain estimator."""

from __future__ import annotations

from typing import Union

import numpy as np

from .exceptions import NotFittedError  # noqa: F401  (re-exported for convenience)
from .leverage import compute_leverage
from .leverage import leverage_threshold as default_leverage_threshold
from .neighbors import (
    knn_threshold_mean_std,
    knn_threshold_percentile,
    leave_one_out_distances,
    nearest_neighbour_distance,
)
from .results import ADResult
from .validation import check_array, check_is_fitted, check_n_features, warn_on_constant_columns


class ApplicabilityDomain:
    """Leverage and k-nearest-neighbour applicability domain.

    Given the training set of a fitted model, decide whether a new query
    point lies inside the region the model was trained on, following the
    leverage-based and distance-based applicability domain criteria
    described in OECD guidance for QSAR models (see the package README
    for full references).

    Two criteria are available and can be used separately or combined:

    - Leverage: how far a point lies from the centre and spread of the
      reference set, via the hat-matrix diagonal, compared against the
      conventional warning threshold h* = 3(p+1)/n.
    - k-nearest-neighbour distance: how far a point lies from its
      closest reference points, compared against a threshold derived
      from the reference set's own internal point spacing.

    A query point is considered in the domain only if it passes every
    enabled criterion.

    Parameters
    ----------
    leverage_threshold : "auto", float, or None, default="auto"
        Leverage threshold h*. "auto" computes the standard
        3(p+1)/n_train value at fit time. A float sets an explicit
        threshold. None disables the leverage criterion entirely.
    knn_k : int, default=1
        Number of nearest neighbours to average over for the
        kNN-distance criterion.
    knn_threshold : "auto", "mean_std", float, or None, default="auto"
        kNN-distance threshold. "auto" uses the knn_percentile-th
        percentile of the reference set's own leave-one-out
        nearest-neighbour distances. "mean_std" uses
        mean + knn_z * std of the same distribution. A float sets an
        explicit threshold. None disables the kNN criterion entirely.
    knn_percentile : float, default=95.0
        Percentile used when knn_threshold="auto".
    knn_z : float, default=3.0
        Number of standard deviations used when knn_threshold="mean_std".

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    n_samples_fit_ : int
        Number of reference samples seen during fit.
    leverage_threshold_ : float or None
        Resolved numeric leverage threshold, or None if the leverage
        criterion is disabled.
    knn_threshold_ : float or None
        Resolved numeric kNN-distance threshold, or None if the kNN
        criterion is disabled.

    Examples
    --------
    >>> import numpy as np
    >>> from adcheck import ApplicabilityDomain
    >>> X_train = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    >>> ad = ApplicabilityDomain().fit(X_train)
    >>> ad.predict(np.array([[0.5, 0.5], [10, 0]]))
    array([ True, False])
    """

    def __init__(
        self,
        leverage_threshold: Union[str, float, None] = "auto",
        knn_k: int = 1,
        knn_threshold: Union[str, float, None] = "auto",
        knn_percentile: float = 95.0,
        knn_z: float = 3.0,
    ) -> None:
        self.leverage_threshold = leverage_threshold
        self.knn_k = knn_k
        self.knn_threshold = knn_threshold
        self.knn_percentile = knn_percentile
        self.knn_z = knn_z

    def fit(self, X_train) -> "ApplicabilityDomain":
        """Fit the applicability domain to a reference (training) set.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Reference data, typically the training set of the model
            being checked. No centering or scaling is applied
            internally.

        Returns
        -------
        self : ApplicabilityDomain
            The fitted instance.

        Raises
        ------
        ValueError
            If X_train is not a valid 2D numeric array, is empty,
            contains NaN or infinite values, has too few samples
            relative to its number of features for the leverage
            criterion to be meaningful, if knn_k is not smaller than
            the number of samples, or if both criteria are disabled.
        """
        if self.leverage_threshold is None and self.knn_threshold is None:
            raise ValueError(
                "At least one of leverage_threshold or knn_threshold "
                "must be enabled (not None); an ApplicabilityDomain "
                "with both criteria disabled would never reject "
                "anything."
            )

        X_train = check_array(X_train, name="X_train")
        n_samples, n_features = X_train.shape

        if self.leverage_threshold is not None and n_samples <= n_features:
            raise ValueError(
                f"X_train has {n_samples} sample(s) and {n_features} "
                f"feature(s); the leverage criterion requires more "
                f"samples than features (n_samples > n_features). Pass "
                f"leverage_threshold=None to fit on this data using "
                f"only the kNN criterion."
            )

        if self.knn_threshold is not None and self.knn_k >= n_samples:
            raise ValueError(
                f"knn_k ({self.knn_k}) must be smaller than the number "
                f"of training samples ({n_samples})."
            )

        warn_on_constant_columns(X_train, name="X_train")

        self.X_train_ = X_train
        self.n_features_in_ = n_features
        self.n_samples_fit_ = n_samples

        if self.leverage_threshold is None:
            self.leverage_threshold_ = None
        elif self.leverage_threshold == "auto":
            self.leverage_threshold_ = default_leverage_threshold(n_samples, n_features)
        else:
            self.leverage_threshold_ = float(self.leverage_threshold)

        if self.knn_threshold is None:
            self.knn_threshold_ = None
        else:
            loo = leave_one_out_distances(X_train, k=self.knn_k)
            if self.knn_threshold == "auto":
                self.knn_threshold_ = knn_threshold_percentile(loo, self.knn_percentile)
            elif self.knn_threshold == "mean_std":
                self.knn_threshold_ = knn_threshold_mean_std(loo, self.knn_z)
            else:
                self.knn_threshold_ = float(self.knn_threshold)

        return self

    def evaluate(self, X_query) -> ADResult:
        """Evaluate query points against the fitted applicability domain.

        Parameters
        ----------
        X_query : array-like of shape (n_samples, n_features)
            Query points to evaluate. Must have the same number of
            features as the data passed to fit.

        Returns
        -------
        ADResult
            Per-query leverage and kNN-distance values, per-criterion
            pass/fail flags, and the combined in_domain verdict.

        Raises
        ------
        NotFittedError
            If called before fit.
        ValueError
            If X_query is not a valid 2D numeric array, is empty,
            contains NaN or infinite values, or has a different number
            of features than the data passed to fit.
        """
        check_is_fitted(self, ["X_train_"])
        X_query = check_array(X_query, name="X_query")
        check_n_features(X_query, self.n_features_in_, name="X_query")

        n_query = X_query.shape[0]

        if self.leverage_threshold_ is None:
            leverage_values = np.full(n_query, np.nan)
            leverage_ok = np.ones(n_query, dtype=bool)
        else:
            leverage_values = compute_leverage(self.X_train_, X_query)
            leverage_ok = leverage_values <= self.leverage_threshold_

        if self.knn_threshold_ is None:
            knn_values = np.full(n_query, np.nan)
            knn_ok = np.ones(n_query, dtype=bool)
        else:
            knn_values = nearest_neighbour_distance(self.X_train_, X_query, k=self.knn_k)
            knn_ok = knn_values <= self.knn_threshold_

        in_domain = leverage_ok & knn_ok

        return ADResult(
            in_domain=in_domain,
            leverage=leverage_values,
            leverage_ok=leverage_ok,
            knn_distance=knn_values,
            knn_ok=knn_ok,
        )

    def predict(self, X_query) -> np.ndarray:
        """Combined in-domain verdict for each query point.

        Equivalent to ``self.evaluate(X_query).in_domain`` but returns
        only the combined boolean array.

        Parameters
        ----------
        X_query : array-like of shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray of bool, shape (n_samples,)
        """
        return self.evaluate(X_query).in_domain

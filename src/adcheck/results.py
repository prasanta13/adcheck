"""Result container returned by ApplicabilityDomain.evaluate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ADResult:
    """Per-query applicability domain evaluation.

    All arrays have one entry per query point, in the same order as the
    input passed to :meth:`adcheck.ApplicabilityDomain.evaluate`.

    Attributes
    ----------
    in_domain : numpy.ndarray of bool, shape (n_query,)
        Combined verdict: True only if the query point passes every
        enabled criterion.
    leverage : numpy.ndarray of float, shape (n_query,)
        Leverage value for each query point. Filled with NaN if the
        leverage criterion was disabled.
    leverage_ok : numpy.ndarray of bool, shape (n_query,)
        Whether each point's leverage is at or below the fitted
        leverage threshold. All True if the leverage criterion was
        disabled.
    knn_distance : numpy.ndarray of float, shape (n_query,)
        Distance (mean over k neighbours) from each query point to its
        nearest reference points. Filled with NaN if the kNN criterion
        was disabled.
    knn_ok : numpy.ndarray of bool, shape (n_query,)
        Whether each point's kNN distance is at or below the fitted
        kNN threshold. All True if the kNN criterion was disabled.
    """

    in_domain: np.ndarray
    leverage: np.ndarray
    leverage_ok: np.ndarray
    knn_distance: np.ndarray
    knn_ok: np.ndarray

    def __len__(self) -> int:
        return len(self.in_domain)

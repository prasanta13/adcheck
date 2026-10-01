"""Shared pytest fixtures."""

import numpy as np
import pytest


@pytest.fixture
def plus_shape_X_train():
    """The 4-point "plus" shape used for the hand-worked example in
    test_leverage.py, test_neighbors.py and test_core.py.

    X_train = [[ 1,  0],
               [ 0,  1],
               [-1,  0],
               [ 0, -1]]

    By symmetry every training point has leverage exactly 0.5 and a
    leave-one-out nearest-neighbour distance of exactly sqrt(2); see
    test_leverage.py for the full derivation.
    """
    return np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )


@pytest.fixture
def random_X_train():
    """A larger, non-trivial reference set for property-based checks
    that do not depend on the exact hand-worked numbers, seeded for
    reproducibility.
    """
    rng = np.random.default_rng(0)
    return rng.normal(size=(50, 5))

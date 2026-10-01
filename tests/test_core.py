"""Integration tests for adcheck.core.ApplicabilityDomain.

These exercise the full fit / predict / evaluate flow, including the
dual-criterion logic and the edge cases listed in the project README:
singular or near-singular reference data, n_train <= n_features,
constant columns, and NaN input.
"""

import numpy as np
import pytest

from adcheck import ADResult, ApplicabilityDomain, NotFittedError


# ---------------------------------------------------------------------
# Core behaviour, against the same hand-worked example as test_leverage.py
# and test_neighbors.py
# ---------------------------------------------------------------------


def test_fit_returns_self(plus_shape_X_train):
    ad = ApplicabilityDomain()
    assert ad.fit(plus_shape_X_train) is ad


def test_fitted_attributes(plus_shape_X_train):
    ad = ApplicabilityDomain().fit(plus_shape_X_train)
    assert ad.n_features_in_ == 2
    assert ad.n_samples_fit_ == 4
    assert ad.leverage_threshold_ == pytest.approx(2.25)
    assert ad.knn_threshold_ == pytest.approx(np.sqrt(2))


def test_predict_matches_hand_worked_example(plus_shape_X_train):
    ad = ApplicabilityDomain().fit(plus_shape_X_train)
    X_query = np.array([[0.5, 0.5], [10.0, 0.0]])
    np.testing.assert_array_equal(ad.predict(X_query), [True, False])


def test_evaluate_returns_ADResult_with_matching_values(plus_shape_X_train):
    ad = ApplicabilityDomain().fit(plus_shape_X_train)
    X_query = np.array([[0.5, 0.5], [10.0, 0.0]])
    result = ad.evaluate(X_query)

    assert isinstance(result, ADResult)
    assert len(result) == 2
    np.testing.assert_allclose(result.leverage, [0.25, 50.0])
    np.testing.assert_allclose(result.knn_distance, [np.sqrt(0.5), 9.0])
    np.testing.assert_array_equal(result.leverage_ok, [True, False])
    np.testing.assert_array_equal(result.knn_ok, [True, False])
    np.testing.assert_array_equal(result.in_domain, [True, False])


def test_dual_criterion_requires_both_to_pass():
    """A point can pass one criterion and fail the other; in_domain
    must be False whenever either one fails. Construct such a point
    deliberately: low leverage (close to the reference centre) but far
    from any individual reference point, by using a reference set with
    a gap in the middle.
    """
    X_train = np.array(
        [
            [5.0, 0.0],
            [-5.0, 0.0],
            [0.0, 5.0],
            [0.0, -5.0],
        ]
    )
    ad = ApplicabilityDomain(knn_threshold=0.5).fit(X_train)
    # The origin has leverage 0 (it is the centroid) but is far
    # (distance 5) from every single reference point under a tight
    # explicit kNN threshold of 0.5.
    result = ad.evaluate(np.array([[0.0, 0.0]]))
    assert result.leverage_ok[0]
    assert not result.knn_ok[0]
    assert not result.in_domain[0]


# ---------------------------------------------------------------------
# Disabling a criterion
# ---------------------------------------------------------------------


def test_leverage_criterion_can_be_disabled(plus_shape_X_train):
    ad = ApplicabilityDomain(leverage_threshold=None).fit(plus_shape_X_train)
    assert ad.leverage_threshold_ is None
    result = ad.evaluate(np.array([[1000.0, 1000.0]]))
    assert np.isnan(result.leverage[0])
    assert result.leverage_ok[0]  # disabled criterion always passes


def test_knn_criterion_can_be_disabled(plus_shape_X_train):
    ad = ApplicabilityDomain(knn_threshold=None).fit(plus_shape_X_train)
    assert ad.knn_threshold_ is None
    result = ad.evaluate(np.array([[1000.0, 1000.0]]))
    assert np.isnan(result.knn_distance[0])
    assert result.knn_ok[0]  # disabled criterion always passes


def test_both_criteria_disabled_raises(plus_shape_X_train):
    with pytest.raises(ValueError, match="At least one"):
        ApplicabilityDomain(leverage_threshold=None, knn_threshold=None).fit(plus_shape_X_train)


# ---------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------


def test_explicit_float_thresholds_are_used_as_is(plus_shape_X_train):
    ad = ApplicabilityDomain(leverage_threshold=1.0, knn_threshold=2.0).fit(plus_shape_X_train)
    assert ad.leverage_threshold_ == 1.0
    assert ad.knn_threshold_ == 2.0


def test_mean_std_threshold_rule(plus_shape_X_train):
    ad = ApplicabilityDomain(knn_threshold="mean_std", knn_z=0.0).fit(plus_shape_X_train)
    # With z=0 the threshold is just the mean of the leave-one-out
    # distances, which on this symmetric shape are all sqrt(2).
    assert ad.knn_threshold_ == pytest.approx(np.sqrt(2))


# ---------------------------------------------------------------------
# Edge cases named in the project README: singular matrices, n <= p,
# constant columns, NaN input, calling before fit, feature mismatch.
# ---------------------------------------------------------------------


def test_not_fitted_error_on_predict():
    ad = ApplicabilityDomain()
    with pytest.raises(NotFittedError):
        ad.predict(np.array([[0.0, 0.0]]))


def test_not_fitted_error_on_evaluate():
    ad = ApplicabilityDomain()
    with pytest.raises(NotFittedError):
        ad.evaluate(np.array([[0.0, 0.0]]))


def test_feature_mismatch_at_predict_time(plus_shape_X_train):
    ad = ApplicabilityDomain().fit(plus_shape_X_train)
    with pytest.raises(ValueError, match="feature"):
        ad.predict(np.array([[1.0, 2.0, 3.0]]))


def test_n_train_not_greater_than_n_features_raises():
    X_train = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 samples, 3 features
    with pytest.raises(ValueError, match="more samples than features"):
        ApplicabilityDomain().fit(X_train)


def test_n_train_not_greater_than_n_features_ok_with_leverage_disabled():
    X_train = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ad = ApplicabilityDomain(leverage_threshold=None).fit(X_train)
    assert ad.n_samples_fit_ == 2


def test_knn_k_too_large_raises(plus_shape_X_train):
    with pytest.raises(ValueError, match="knn_k"):
        ApplicabilityDomain(knn_k=10).fit(plus_shape_X_train)


def test_singular_reference_set_does_not_raise():
    """Two perfectly collinear columns; X^T X is singular. Must still
    fit and evaluate successfully via the pseudo-inverse.
    """
    X_train = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [-1.0, -1.0],
        ]
    )
    ad = ApplicabilityDomain().fit(X_train)
    result = ad.evaluate(np.array([[1.5, 1.5]]))
    assert np.isfinite(result.leverage[0])


def test_constant_column_warns_but_does_not_raise():
    X_train = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
    with pytest.warns(UserWarning, match="constant"):
        ad = ApplicabilityDomain().fit(X_train)
    result = ad.evaluate(np.array([[2.5, 5.0]]))
    assert np.isfinite(result.leverage[0])


def test_nan_in_training_data_raises():
    X_train = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    with pytest.raises(ValueError, match="NaN or infinite"):
        ApplicabilityDomain().fit(X_train)


def test_nan_in_query_data_raises(plus_shape_X_train):
    ad = ApplicabilityDomain().fit(plus_shape_X_train)
    with pytest.raises(ValueError, match="NaN or infinite"):
        ad.predict(np.array([[np.nan, 0.0]]))


def test_1d_input_raises_with_reshape_hint(plus_shape_X_train):
    ad = ApplicabilityDomain().fit(plus_shape_X_train)
    with pytest.raises(ValueError, match="reshape"):
        ad.predict(np.array([1.0, 2.0]))

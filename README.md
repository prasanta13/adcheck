# adcheck

![CI](https://github.com/prasanta13/adcheck/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Leverage and k-nearest-neighbour applicability domain checking for trained models.

## The problem

A model is only as trustworthy as the data it was trained on. If you train a model
on one region of feature space and then ask it to predict something far outside
that region, the prediction is an extrapolation, not an interpolation, and the
model usually has no way of telling you that on its own. It will return a number
with the same apparent confidence whether the input looked like its training data
or nothing like it at all.

The applicability domain of a model is the region of feature space its training
data actually covers. Checking whether a new point falls inside that region, before
trusting the prediction for it, is standard practice in some fields (notably
QSAR modelling in cheminformatics, where it is required by OECD guidance) and
largely absent in others.

## What this package does

Given a reference set of points, typically a model's training data, `adcheck`
decides whether new query points lie inside the region that reference set covers,
using two criteria that can be used separately or combined:

- **Leverage**: how far a point lies from the centre and spread of the reference
  set, via the diagonal of the regression hat matrix, compared against a
  standard threshold.
- **k-nearest-neighbour distance**: how far a point lies from its closest
  reference points, compared against a threshold derived from how closely packed
  the reference set already is.

The package makes no assumption about what the features represent. It operates on
plain numeric arrays of shape `(n_samples, n_features)`.

## Installation

```
pip install -e .
```

or, to also install the test dependencies:

```
pip install -e ".[dev]"
```

Requires Python 3.10 or later. The only runtime dependency is NumPy.

## Worked example

```python
import numpy as np
from adcheck import ApplicabilityDomain

rng = np.random.default_rng(0)
X_train = rng.normal(size=(200, 5))

ad = ApplicabilityDomain(knn_k=3).fit(X_train)

X_query = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],      # the centre of the training distribution
    [1.0, -1.0, 0.5, 0.0, -0.5],    # an unremarkable point
    [8.0, 8.0, 8.0, 8.0, 8.0],      # far outside the training distribution
])

result = ad.evaluate(X_query)
print(result.in_domain)
```

This is `examples/basic_usage.py` in full, and its actual output, copied from a
real run and not retyped by hand, is:

```
Fitted on 200 reference points, 5 features
leverage_threshold_ (h*)      = 0.0900
knn_threshold_ (95th pct, k=3) = 1.7726

point               leverage   leverage_ok    knn_dist    knn_ok   in_domain
----------------------------------------------------------------------------
origin                0.0000          True      0.7339      True        True
typical point         0.0152          True      0.9767      True        True
far outlier           1.6354         False     15.4123     False       False

predict() returns just the combined verdict:
[ True  True False]
```

`predict` gives you the one-line answer. `evaluate` gives you everything behind
it, including which specific criterion failed, which is usually the more useful
thing to know.

## The method

### Leverage

For a reference matrix `X_ref` of shape `(n_ref, n_features)`, the leverage of a
point `x` is

```
h(x) = x @ pinv(X_ref.T @ X_ref) @ x.T
```

the diagonal of the regression hat matrix `H = X (X^T X)^+ X^T`, generalised to
query points that need not themselves be part of the reference set. The
Moore-Penrose pseudo-inverse is used in place of a plain matrix inverse so that
collinear or constant-column reference data does not raise a numerical error.

A point is flagged by the leverage criterion if `h(x)` exceeds the conventional
warning threshold

```
h* = 3(p + 1) / n
```

where `p` is the number of features and `n` is the number of reference samples.
This threshold, and the leverage approach generally, follows Gramatica (2007) and
the OECD guidance referenced below. No centering or scaling of `X_ref` is applied
internally; standardise your features first if that is what you want, since
leverage on raw, unscaled features will be dominated by whichever feature happens
to have the largest numeric range.

### k-nearest-neighbour distance

For each query point, the mean distance to its `k` nearest reference points is
compared against a threshold derived from the reference set's own internal
spacing: specifically, for every reference point, the mean distance to its `k`
nearest other reference points (a leave-one-out calculation), and then either the
95th percentile of that distribution (the default) or its mean plus some number
of standard deviations.

A point close to the reference set in this sense has a low kNN distance; a point
in a gap, or far outside the reference set entirely, has a high one.

### Combining them

A query point is reported as being inside the applicability domain only if it
passes every criterion that is enabled. Using both criteria together, rather than
either alone, follows the recommendation of Sahigara et al. (2012): the two
catch different failure modes. Leverage is a global measure and can miss a point
that sits in a local gap within an otherwise unremarkable region; kNN distance is
local and can miss a point that is globally unusual but happens to be close to
one similarly unusual reference point. Either criterion can be disabled
individually by setting its threshold to `None`, if you only want the other one.

### References

- OECD (2007). *Guidance Document on the Validation of (Quantitative)
  Structure-Activity Relationship [(Q)SAR] Models*. OECD Series on Testing and
  Assessment No. 69.
- Gramatica, P. (2007). Principles of QSAR models validation: internal and
  external. *QSAR & Combinatorial Science*, 26(5), 694-701.
- Roy, K., Kar, S., and Ambure, P. (2015). On a simple approach for determining
  applicability domain of QSAR models. *Chemometrics and Intelligent Laboratory
  Systems*, 145, 22-29.
- Sahigara, F., Mansouri, K., Ballabio, D., Mauri, A., Consonni, V., and Todeschini,
  R. (2012). Comparison of different approaches to define the applicability
  domain of QSAR models. *Molecules*, 17(5), 4791-4810.

## API

```python
ApplicabilityDomain(
    leverage_threshold="auto",  # "auto", a float, or None to disable
    knn_k=1,                    # neighbours used for the distance criterion
    knn_threshold="auto",       # "auto", "mean_std", a float, or None to disable
    knn_percentile=95.0,        # used when knn_threshold="auto"
    knn_z=3.0,                  # used when knn_threshold="mean_std"
)
```

- `.fit(X_train)` fits the domain to a reference set and returns `self`.
- `.predict(X_query)` returns a boolean array, the combined verdict.
- `.evaluate(X_query)` returns an `ADResult` with `in_domain`, `leverage`,
  `leverage_ok`, `knn_distance`, and `knn_ok`, each a per-query array.

After fitting, `n_features_in_`, `n_samples_fit_`, `leverage_threshold_`, and
`knn_threshold_` are available as attributes, the last two being the actual
resolved numeric thresholds rather than the `"auto"` string you configured.

## Edge cases

- Non-numeric input, NaN, or infinite values raise `ValueError` immediately,
  with a specific message, rather than propagating into a silently wrong result.
- A 1D array raises `ValueError` with a hint to reshape it, rather than being
  interpreted ambiguously as one sample or one feature.
- A reference set with `n_samples <= n_features` raises `ValueError` for the
  leverage criterion specifically, since the warning-leverage threshold assumes
  real degrees of freedom; pass `leverage_threshold=None` to fit on such data
  using only the kNN criterion.
- Collinear or constant-column reference data does not raise: the
  pseudo-inverse keeps leverage well-defined, though a constant column triggers
  a `UserWarning` since it usually indicates a redundant feature.
- Calling `predict` or `evaluate` before `fit` raises `NotFittedError`.

## Development

```
git clone https://github.com/prasanta13/adcheck.git
cd adcheck
pip install -e ".[dev]"
pytest --cov=adcheck --cov-report=term-missing
```

See `CONTRIBUTING.md` for more.

## License

MIT, see `LICENSE`.

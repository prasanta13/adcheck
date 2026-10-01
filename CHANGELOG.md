# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `ApplicabilityDomain`: leverage (hat-matrix) and k-nearest-neighbour
  distance applicability domain criteria, usable separately or
  combined, with a scikit-learn-like `fit` / `predict` / `evaluate`
  API.
- Standard warning-leverage threshold h* = 3(p+1)/n, following
  Gramatica (2007).
- Configurable kNN-distance threshold: percentile of the reference
  set's own leave-one-out nearest-neighbour distances, or
  mean + z * standard deviation.
- `ADResult` dataclass exposing per-query numeric values and
  per-criterion pass/fail flags, not just the combined verdict.
- Input validation with specific error messages for non-numeric data,
  NaN or infinite values, wrong array shape, feature-count mismatch
  between fit and predict, and a reference set too small relative to
  its number of features. Collinear or constant-column reference data
  is handled via a pseudo-inverse rather than raising.
- Full pytest suite including a hand-derived numeric example.
- GitHub Actions CI running the test suite on Python 3.10 and 3.13.

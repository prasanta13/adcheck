# Contributing to adcheck

Thanks for considering a contribution. This is a small, deliberately
narrow package, so the bar for new features is scope: it should fit
the one stated goal, deciding whether a query point lies in the
applicability domain of a reference set, rather than growing into a
general outlier detection toolkit.

## Setting up a development environment

```
git clone https://github.com/prasanta13/adcheck.git
cd adcheck
pip install -e ".[dev]"
```

This installs the package in editable mode along with pytest and
pytest-cov.

## Running the tests

```
pytest --cov=adcheck --cov-report=term-missing
```

All tests should pass and coverage should stay effectively complete.
New code should come with new tests, not just an assertion that it
works, see `tests/test_leverage.py` for the project's convention of
deriving expected values explicitly in a comment rather than trusting
a second computation.

## Code style

- Functions are not prefixed with a leading underscore, whether public
  or internal. What is public is decided by `adcheck/__init__.py` and
  documented in the README, not by naming convention.
- NumPy-style docstrings on every public function, class and method.
- Type hints on function signatures where they add clarity.
- Raise `ValueError` with a specific, actionable message for invalid
  input, rather than letting NumPy raise an opaque one underneath.

## Reporting issues

Open a GitHub issue with a minimal example that reproduces the
problem, including the shapes and, if relevant, the dtype of the
arrays involved.

## Pull requests

- Keep a pull request to one change. A bug fix and a new feature in
  the same PR are harder to review and harder to revert independently.
- Update `CHANGELOG.md` under an "Unreleased" heading.
- Make sure `pytest` passes locally before opening the PR; CI will run
  it again on a couple of Python versions, but catching problems
  locally first is faster for everyone.

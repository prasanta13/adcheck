"""Custom exceptions used by adcheck."""


class NotFittedError(RuntimeError):
    """Raised when a method that requires a fitted estimator is called
    before ``fit`` has been run.
    """

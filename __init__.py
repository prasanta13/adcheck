"""adcheck: leverage and k-nearest-neighbour applicability domain checking.

Given a model's training set and new query points, decide whether each
query point lies inside the region the model was trained on. See the
README for the method and references.
"""

from .core import ApplicabilityDomain
from .exceptions import NotFittedError
from .results import ADResult

__all__ = ["ApplicabilityDomain", "ADResult", "NotFittedError"]
__version__ = "0.1.0"

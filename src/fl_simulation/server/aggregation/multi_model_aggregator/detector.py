"""Anomaly detection for model updates."""

from typing import Any, Collection, Generic, Tuple
from typing_extensions import Protocol

from ._types import T_update


# DEBUG this does not allow using simple lambdas.
class Detector(Protocol, Generic[T_update]):
    """A abnormal update detector.
    
    Given model updates, returns two collections: the first one with only "normal" updates, 
    the second - with "abnormal" updates.
    """

    # NOTE design choice: instead of having a detector, which would classify one update, we have one with access to all
    # updates. This would allow using methods, requiring all of the updates for making decision, such as Krum.
    def __call__(self, updates: Collection[T_update], *args: Any,
                 **kwds: Any) -> Tuple[Collection[T_update], Collection[T_update]]:
        """Split the provided updates in two groups: normal and abnormal.

        Args:
            updates (Collection[T_update]): model updates.

        Returns:
            Tuple[Collection[T_update], Collection[T_update]]: normal and abnormal updates respectively.
        """
        ...

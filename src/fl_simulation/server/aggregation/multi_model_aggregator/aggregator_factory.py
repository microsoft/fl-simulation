"""Aggregator factory."""

from typing import Any, Generic

import torch
import torch.nn as nn
from typing_extensions import Protocol

from ._types import T_aggregator


class AggregatorFactory(Protocol, Generic[T_aggregator]):
    """An Aggregator factory.
    
    Used to instansiate a new aggregator. Works with existing aggregators. For example:

    ```python
    a: AggregatorFactory = FedAvgAggregator.__call__

    ```
    """

    def __call__(
            self, intial_model: nn.Module, device: torch.device = torch.device("cpu"), *args: Any, **kwds: Any
    ) -> T_aggregator:
        """Instansiate a new aggregator.

        Args:
            intial_model (nn.Module): initial model.
            device (torch.device, optional): device used for computation. Defaults to torch.device("cpu").

        Returns:
            T_aggregator: Aggregator
        """
        ...

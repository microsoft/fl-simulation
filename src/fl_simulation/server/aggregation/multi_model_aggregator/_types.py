"""Type variables."""

from typing import Hashable, TypeVar

from fl_simulation.client.update import ModelUpdate
from fl_simulation.server.aggregation.aggregator import Aggregator

T_update = TypeVar("T_update", bound=ModelUpdate)
# NOTE covariant added for typechecker in AggregatorFactory to work
T_aggregator = TypeVar("T_aggregator", bound=Aggregator, covariant=True)
T_wid = TypeVar("T_wid", bound=Hashable)

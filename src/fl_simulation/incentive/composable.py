"""A composable incentive machanism."""

from typing import Collection, List

from fl_simulation.client.update import ModelUpdate
from fl_simulation.incentive.incentive_mechanism import IncentiveMechanism
from fl_simulation.server.update import AggregatedUpdate


class ComposableIm(IncentiveMechanism):
    """An incentive mechanism that is composed of many others.
    
    They all share the same underlying reward tracking object.
    """

    def __init__(self, ims: List[IncentiveMechanism]):
        """Instantiate a new `ComposableIm`.

        Args:
            ims (List[IncentiveMechanism]): The incentive mechanisms to "join".
                Their underlying rewards will be set to the same instance.
        """
        super().__init__()
        self.ims = ims
        for im in self.ims:
            im.rewards = self.rewards

    def update(self, updates: Collection[ModelUpdate], aggregated_update: AggregatedUpdate) -> None:
        for im in self.ims:
            im.update(updates, aggregated_update)

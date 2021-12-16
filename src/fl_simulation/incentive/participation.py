from typing import Collection

from fl_simulation.client.update import ModelUpdate
from fl_simulation.incentive.incentive_mechanism import IncentiveMechanism
from fl_simulation.server.update import AggregatedUpdate


class ParticipationIm(IncentiveMechanism):
    """
    Track participation in the simulation.
    A worker gets 1 point for their update being included.
    A worker gets 1 point for each example they contributed.
    """

    def update(self, updates: Collection[ModelUpdate], aggregated_update: AggregatedUpdate) -> None:
        for update in updates:
            self.reward(update.worker_id, 'num_updates', 1)
            self.reward(update.worker_id, 'num_examples', update.num_examples)

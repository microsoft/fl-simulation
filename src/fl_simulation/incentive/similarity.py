from typing import Callable, Collection

from fl_simulation.client.update import ModelUpdate
from fl_simulation.incentive.incentive_mechanism import IncentiveMechanism
from fl_simulation.server.update import AggregatedUpdate
from fl_simulation.utils.cosine_similarity import cosine_similarity_model_diff
from fl_simulation.utils.types import ModelDiff


class SimilarityIm(IncentiveMechanism):
    """
    Reward for having updates similar to the aggregated update.
    """

    def __init__(
            self,
            reward_name: str = 'similarity',
            similarity_fn: Callable[[ModelDiff, ModelDiff], float] = cosine_similarity_model_diff
    ):
        """
        Args:
            reward_name (str): The name of the key to track similarity in the rewards.
            similarity_fn (Callable[[ModelDiff, ModelDiff], float]): The function to use to compare a worker's model update to the aggregated update.
            Defaults to cosine similarity with a value in [-1,1].
        """
        super().__init__()
        self.reward_name = reward_name
        self.similarity_fn = similarity_fn

    def update(self, updates: Collection[ModelUpdate], aggregated_update: AggregatedUpdate) -> None:
        for update in updates:
            self.reward(
                    update.worker_id, self.reward_name,
                    self.similarity_fn(update.values, aggregated_update.model_diff)
            )

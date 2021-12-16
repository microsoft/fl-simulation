from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Collection, Dict, Hashable

from fl_simulation.client.update import ModelUpdate
from fl_simulation.server.update import AggregatedUpdate


class IncentiveMechanism(ABC):
    """
    Defines incentives as rewards for contributing "good" quality data
    or to track penalties for "bad" contributions.
    """

    def __init__(self):
        self.rewards = defaultdict(lambda: defaultdict(float))
        """
        The underlying rewards being tracked.
        """

    def reward(self, worker_id: Hashable, reward_type: Hashable, amount: float) -> None:
        """
        Add a reward (or penalty) for a specific worker.

        Args:
            worker_id (Hashable): The ID for the worker.
            reward_type (Hashable): The type or category of the reward.
            amount (float): The amount to add. You can use a negative number to penalize.
        """
        self.rewards[worker_id][reward_type] += amount

    def get_reward(self, worker_id: Hashable, reward_type: Hashable) -> float:
        """
        Args:
            worker_id (Hashable): The ID for the worker.
            reward_type (Hashable): The type or category of the reward.

        Returns:
            float: The reward (or penalty) for a specific worker.
        """
        return self.rewards[worker_id][reward_type]

    def get_rewards(self, worker_id: Hashable) -> Dict[Hashable, float]:
        """
        Args:
            worker_id (Hashable): The ID for the worker.

        Returns:
            Dict[Hashable, float]: The rewards (or penalties) for a specific worker.
        """
        return self.rewards[worker_id]

    @abstractmethod
    def update(self, updates: Collection[ModelUpdate], aggregated_update: AggregatedUpdate) -> None:
        """
        Compute rewards based on the updates.

        Args:
            updates (Collection[ModelUpdate]): model updates (weight differences, gradients, etc.) and, maybe, additional meta-information.
            aggregated_update (AggregatedUpdate): aggregated model updates.
        """
        raise NotImplementedError()

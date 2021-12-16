"""Federated Averaging algorithm."""

from typing import Collection

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD

from fl_simulation.client.update import FedAvgModelUpdate
from fl_simulation.server.aggregation.aggregator import Aggregator
from fl_simulation.server.update import AggregatedUpdate
from fl_simulation.utils.types import ModelDiff


class FedAvgAggregator(Aggregator):
    """Federated Averaging Aggregator. See McMahan et al.: "Communication-Efficient Learning of Deep Networks from Decentralized Data", 2017."""

    def get_opt(self) -> Optimizer:
        """Get the global optimizer for Federated Averaging, which is SGD with learning rate `1.0`.

        Returns:
            Optimizer: global optimizer, i.e. SGD with learning rate `1.0`.
        """
        return SGD(self.model.parameters(), lr=1.0)

    def aggr_fn(self, updates: Collection[FedAvgModelUpdate]) -> AggregatedUpdate:
        """Aggregate model updates into a single model update by weighted averaging.

        The averaging weights are proportional to the number of examples a given client used to compute the update.

        Args:
            updates (Iterable[FedAvgModelUpdate]): model updates (weight differences) and the number of examples the client used.

        Raises:
            ValueError: if `updates` is an empty iterator.

        Returns:
            ModelDiff: weighted average of model updates.
        """
        aggr_update = []

        total_examples = sum(m.num_examples for m in updates) or 1

        for i, _ in enumerate(self.model.parameters()):
            aggr_update.append(
                    torch.stack([m.values[i] * m.num_examples for m in updates]).sum(dim=0) / total_examples
            )

        return AggregatedUpdate(ModelDiff(aggr_update))

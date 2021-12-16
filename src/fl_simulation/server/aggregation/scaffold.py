"""Stochastic Controlled Averaging for Federated Learning (SCAFFOLD) algorithm."""

from typing import Collection, Hashable, Optional

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD

from fl_simulation.client.update import ScaffoldModelUpdate
from fl_simulation.incentive.incentive_mechanism import IncentiveMechanism
from fl_simulation.server.aggregation.aggregator import Aggregator
from fl_simulation.server.update import ScaffoldAggregatedUpdate, ScaffoldServerUpdate
from fl_simulation.utils.model import set_model_gradients
from fl_simulation.utils.types import ControlVarDiff, ModelDiff


class ScaffoldAggregator(Aggregator):
    """SCAFFOLD aggregation algorithm. See "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" by Karimireddy et al."""

    ctl_var: ControlVarDiff
    """The server's control variate."""
    total_clients: int
    """The total number of the clients in the system."""

    def __init__(
            self,
            initial_model: nn.Module,
            total_clients: int,
            device: torch.device = torch.device("cpu"),
            incentive_mechanism: Optional[IncentiveMechanism] = None,
    ) -> None:
        """Create a new `ScaffoldAggregator` instance.

        Args:
            initial_model (torch.nn.Module): initial shared model.
            total_clients (int): the total number of clients in the system.
            device (torch.device): pytorch device to use. Defaults to CPU.
            incentive_mechanism (IncentiveMechanism, optional): reward tracker.
        """
        super().__init__(initial_model, device, incentive_mechanism)
        self.ctl_var = ControlVarDiff([torch.zeros_like(p.data) for p in initial_model.parameters()])
        self.total_clients = total_clients

    def aggr_fn(self, updates: Collection[ScaffoldModelUpdate]) -> ScaffoldAggregatedUpdate:
        """Aggregate model updates into a single model update by weighted averaging.

        The averaging weights are proportional to the number of examples a given client used to compute the update.

        Args:
            updates (Collection[ScaffoldModelUpdate]): model updates (weight differences) and the number of examples the client used.

        Raises:
            ValueError: if `updates` is an empty iterator.

        Returns:
            ScaffoldAggregatedUpdate: weighted average of model updates and weighted average of control variate updates.
        """
        aggr_update = []
        ctl_var_update = []

        total_examples = sum(m.num_examples for m in updates) or 1

        for i, _ in enumerate(self.model.parameters()):
            aggr_update.append(
                    torch.stack([m.values[i] * m.num_examples for m in updates]).sum(dim=0) / total_examples
            )
            ctl_var_update.append(
                    torch.stack([m.ctl_var_update[i] * m.num_examples for m in updates]).sum(dim=0) / total_examples
            )

        return ScaffoldAggregatedUpdate(ModelDiff(aggr_update), ControlVarDiff(ctl_var_update))

    def aggregate(self, updates: Collection[ScaffoldModelUpdate]) -> None:
        """Aggregate the received updates and apply them to the shared model.

        Args:
            updates (Collection[ScaffoldModelUpdate]): model updates to be applied.
        """
        clients_in_cycle = len(updates)

        # aggregated model updates result in the pseudo-gradient, which can be used in the optimizer
        self.model.train()

        self.global_optim.zero_grad()
        aggregated_update = self.aggr_fn(updates)
        self.run_post_aggregation_hooks(updates, aggregated_update)
        set_model_gradients(self.model, aggregated_update.model_diff)
        self.global_optim.step()

        # aggregate control variate's updates
        self.apply_ctl_var_update(aggregated_update.ctl_var_diff, clients_in_cycle)

    def apply_ctl_var_update(self, update: ControlVarDiff, cl_in_cycle: int) -> None:
        """Apply control variate's update.

        Args:
            update (ControlVarDiff): the aggregated update to be applied.
            cl_in_cycle (int): the number of clients participating in the current cycle.
        """
        coeff = cl_in_cycle / self.total_clients
        # c <- c + |S|/N Δc
        for i, upd in enumerate(update):
            # use minus because Δc := c_i - c_i^+, not Δc := c_i^+ - c_i as in the paper
            # this is done for consistency with FedAvg Algorithm
            self.ctl_var[i] -= coeff * upd

    def get_opt(self, lr: float = 1.0) -> Optimizer:
        """Get the global optimizer to be used.

        Args:
            lr (float, optional): learning rate. Defaults to 1.0.

        Returns:
            Optimizer: SGD with the specified learning rate.
        """
        return SGD(self.model.parameters(), lr=lr)

    def get_server_update(self, _worker_id: Optional[Hashable] = None) -> ScaffoldServerUpdate:
        """Get the server update with the most recent version of the shared model and metadata.

        Returns:
            ScaffoldServerUpdate: the server update.
        """
        return ScaffoldServerUpdate(shared_model=self.model, ctl_var=self.ctl_var)

    def save_checkpoint(self, to: Optional[str] = None, **extra) -> str:
        return super().save_checkpoint(to=to, ctl_var=self.ctl_var, total_clients=self.total_clients, **extra)

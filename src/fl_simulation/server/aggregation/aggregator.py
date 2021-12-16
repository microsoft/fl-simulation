"""Aggregator interface."""

import os
from abc import ABC, abstractmethod
from typing import Collection, Hashable, Optional

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from fl_simulation.client.update import ModelUpdate
from fl_simulation.incentive.incentive_mechanism import IncentiveMechanism
from fl_simulation.logging.logger import get_logger
from fl_simulation.server.update import AggregatedUpdate, ServerUpdate
from fl_simulation.utils.model import set_model_gradients, set_model_params
from fl_simulation.utils.types import ModelDiff

logger = get_logger()


class Aggregator(ABC):
    """Aggregates the collected model updates and applies them to the model using the specified optimization rule."""

    model: nn.Module
    """Shared model."""
    global_optim: Optimizer
    """Global optimizer for applying model updates."""
    device: torch.device
    """PyTorch device to use."""
    incentive_mechanism: Optional[IncentiveMechanism]
    """Incentive Mechanism for tracking rewards and penalties."""

    def __init__(
            self,
            initial_model: nn.Module,
            device: torch.device = torch.device("cpu"),
            incentive_mechanism: Optional[IncentiveMechanism] = None,
    ) -> None:
        """Create a new `Aggregator` instance.

        Args:
            initial_model (torch.nn.Module): the initial shared model to be used.
            device (torch.device): PyTorch device to use. Defaults to CPU.
            incentive_mechanism (IncentiveMechanism, optional): reward tracker.
        """
        self.device = device

        # Shared model.
        self.model = initial_model

        # Shared model optimizer.
        self.global_optim = self.get_opt()

        self.incentive_mechanism = incentive_mechanism

    @abstractmethod
    def get_opt(self) -> Optimizer:
        """Get the global optimizer. A class method.

        Returns:
            Optimizer: optimizer to be used.
        """
        return NotImplemented

    def run_post_aggregation_hooks(self, updates: Collection[ModelUpdate], aggregated_update: AggregatedUpdate):
        """Run a set of actions after aggregation.

        Args:
            updates (Collection[ModelUpdate]): a set of model updates.
            aggregated_update (AggregatedUpdate): a resulting aggregated model update.
        """
        logger.debug("Running post aggregation hooks with %d updates.", len(updates))
        if self.incentive_mechanism:
            self.incentive_mechanism.update(updates, aggregated_update)

    @abstractmethod
    def aggr_fn(self, updates: Collection[ModelUpdate]) -> AggregatedUpdate:
        """Aggregate model updates into a single model update. A static method.

        Args:
            updates (Collection[ModelUpdate]): model updates (weight differences, gradients, etc.) and, maybe, additional meta-information.

        Returns:
            AggregatedUpdate: aggregated model updates.
        """
        return NotImplemented

    def aggregate(self, updates: Collection[ModelUpdate]) -> None:
        """Aggregate the received updates and apply them to the shared model.

        Args:
            updates (Iterable[ModelUpdate]): model updates to be applied.
        """
        # aggregated model updates result in the pseudo-gradient, which can be used in the optimizer
        self.model.train()

        self.model.to(self.device)

        self.global_optim.zero_grad()
        logger.info("Computing the aggregation of %d updates.", len(updates))
        aggregation_result = self.aggr_fn(updates)

        aggregation_result.model_diff = ModelDiff([grad.to(self.device) for grad in aggregation_result.model_diff])

        self.run_post_aggregation_hooks(updates, aggregation_result)

        set_model_gradients(self.model, aggregation_result.model_diff)
        self.global_optim.step()

        self.model.cpu()

    def set_shared_model(self, model: nn.Module) -> None:
        """Set the shared model to the specified one.

        Args:
            model (torch.nn.Module): new version of the shared model. Must have the same architecture as the previous one.
        """
        set_model_params(self.model, model.parameters())

    def get_server_update(self, _worker_id: Optional[Hashable] = None) -> ServerUpdate:
        """Get the server update with the most recent version of the shared model and metadata.

        Returns:
            ServerUpdate: the server update.
        """
        return ServerUpdate(shared_model=self.model)

    def save_checkpoint(self, to: Optional[str] = None, **extra) -> str:
        """Save the state of this aggregator.

        Args:
            to (Optional[str], optional): directory to save the state to. If `None`, then creates a subfolder in the current working directory, named "aggregator". Defaults to None.
            **extra (Dict, optional): other parameters to be saved.

        Returns:
            str: path to the directory containing saved state.
        """
        to = to or "aggregator"
        os.makedirs(to, exist_ok=True)

        # save the global model
        torch.save(self.model, os.path.join(to, "model.pt"))

        # save the optimizer state
        torch.save(self.global_optim.state_dict(), os.path.join(to, "optim.pickle"))

        # save other parameters
        torch.save(extra, os.path.join(to, "params.pickle"))

        return to

    def load_from_checkpoint(self, d: Optional[str] = None) -> None:
        """Load the state of the aggregator form the checkpoint.

        Args:
            d (Optional[str], optional): path to folder containing the saved state. If `None` is provided, then uses "aggregator" as the directory name. Defaults to None.
        """
        d = d or "aggregator"

        # load the local model
        model_params = nn.Module.parameters(torch.load(os.path.join(d, "model.pt")))

        set_model_params(self.model, model_params)

        # load the optimizer
        self.global_optim.load_state_dict(dict(torch.load(os.path.join(d, "optim.pickle"))))

        # load other parameters
        params = dict(torch.load(os.path.join(d, "params.pickle")))
        for attr, p in params.items():
            setattr(self, attr, p)

"""A worker interface."""

import copy
import os
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, NewType, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fl_simulation.client.update import ModelDiff, ModelUpdate
from fl_simulation.server.update import ServerUpdate
from fl_simulation.utils.model import set_model_params

WorkerEvalResult = NewType("WorkerEvalResult", Dict[str, float])
"""Results of running evaluation on the given client."""


class Worker(ABC):
    """Handles the local data and runs the computations for the training process."""

    _round_num: int = 0

    name: str
    """The name of this worker."""
    base_model: nn.Module
    """Shared model at the beginning of the cycle."""
    local_model: nn.Module
    """Locally trained model. Useful for model interpolation."""
    epochs_done: int
    """Number of completed local epochs."""
    optim: Optimizer
    """Local optimizer."""
    num_examples: int
    """Number of local training examples."""
    _loss_fn_impl: Callable[..., torch.Tensor]
    """Loss function implementation."""
    device: torch.device
    """PyTorch device to use."""
    _tensorboard_writer: Optional[SummaryWriter] = None
    """TensorBoard logger."""

    def __init__(
            self,
            base_model: nn.Module,
            loss_fn: Callable[..., torch.Tensor],
            name: Optional[str] = None,
            device: torch.device = torch.device("cpu"),
            tensorboard_writer: SummaryWriter = None,
            *opt_args,
            **opt_kwargs,
    ) -> None:
        """Create a `Worker` instance.

        Args:
            base_model (torch.nn.Module): initial local model.
            loss_fn (Callable[..., torch.Tensor]): loss function to use.
            name (Optional[str], optional): the name of this worker. If `None` is provided, uses a UUID. Defaults to `None`.
            device (torch.device, optional): pytorch device to use. Defaults to CPU.
            tensorboard_writer (Optional[SummaryWriter], optional): a logger for TensorBoard.
            opt_args: optimizer's positional arguments.
            opt_kwargs: optimizer's keyword arguments.
        """
        # NOTE Here we use a reference to the model, so that we would not need to store another copy of its parameters.
        self.base_model = base_model
        # The local model might be used by some algorithms, for example, those with model interpolation
        # `deepcopy` will not carry updates to the `base_model` when updating the local one
        self.local_model = copy.deepcopy(base_model)
        # No epochs performed
        self.epochs_done = 0
        # number of datapoints on this worker
        self.num_examples = 0

        # set up a loss function
        self.loss_fn = loss_fn
        self._tensorboard_writer = tensorboard_writer
        self.optim = self.get_opt(*opt_args, **opt_kwargs)
        self.device = device
        self.name = name or str(uuid.uuid4())

    @abstractmethod
    def get_opt(self, *opt_args, **opt_kwargs) -> Optimizer:
        """Get the local optimizer.

        Args:
            opt_args: optimizer's positional arguments.
            opt_kwargs: optimizer's keyword arguments.

        Returns:
            Optimizer: optimizer to be used.
        """
        return NotImplemented

    @abstractmethod
    def continue_training(self) -> bool:
        """Return `True` if the next epoch should be carried out, `False` otherwise.

        Returns:
            bool: whether to continue training.
        """
        return NotImplemented

    def do_optim_step(self, data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Perform a model update for a single mini-batch.

        Args:
            data (Tuple[torch.Tensor, torch.Tensor]): a single mini-batch of training data.

        Returns:
            torch.Tensor: The loss.
        """
        self.optim.zero_grad()

        loss = self.backprop(data)

        self.optim.step()

        return loss

    def backprop(self, data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Do a backpropagation step.

        Args:
            data (Tuple[torch.Tensor, torch.Tensor]): mini-batch to train on.

        Returns:
            torch.Tensor: The loss.
        """
        X, y = data

        X, y = X.to(self.device), y.to(self.device)

        pred = self.local_model(X)
        loss = self.loss_fn(pred, y)

        loss.backward()
        return loss

    @property
    def loss_fn(self):
        """Loss function to be used in local training."""
        return self._loss_fn_impl

    @loss_fn.setter
    def loss_fn(self, new_loss_fn):
        self._loss_fn_impl = new_loss_fn

    def do_cycle(self, srv_update: ServerUpdate, data_loader: DataLoader) -> ModelUpdate:
        """Perform a single round of local update (i.e. do the specified number of local epochs).

        This would return the model updates, intended to be sent to the server. Can only be used if `epoch_num` was
        set in configuration.

        Args:
            srv_update (ServerUpdate): the update received from the server.
            data_loader (torch.utils.data.DataLoader): local training data.

        Returns:
            ModelUpdate: resulting model update.
        """
        self.reset(srv_update.shared_model)

        self.local_model.to(self.device)
        self.base_model.to(self.device)

        self.local_model.train()

        self.num_examples = 0

        while self.continue_training():
            loss = None
            for data in data_loader:
                loss = self.do_optim_step(data)
                # TODO not all data is batch_size
                self.num_examples += data_loader.batch_size or 0
            if loss is not None and self._tensorboard_writer:
                self._tensorboard_writer.add_scalar(f'train/loss/{self.name}', loss, self.epochs_done)
            self.epochs_done += 1

        self.num_examples //= self.epochs_done

        self.local_model.cpu()
        self.base_model.cpu()

        return self.get_model_update()

    def do_eval_cycle(self, data: Optional[DataLoader] = None, model: Optional[nn.Module] = None) -> WorkerEvalResult:
        """Run evaluation cycle on this worker.

        Args:
            data (Optional[DataLoader], optional): data to evaluate on. If not provided, worker will evaluate on its owned data (NOT IMPLEMENTED). Defaults to None.
            model (Optional[nn.Module], optional): model to evaluate. If not provided, will use the local model. Defaults to None.

        Raises:
            NotImplementedError: if no data is supplied.

        Returns:
            WorkerEvalResult: evaluation results.
        """
        if not data:
            # TODO add evaluation on owned data.
            raise NotImplementedError("workers do not have owned data for now")

        if not model:
            model = self.local_model

        model.to(self.device)
        model.eval()

        avg_metrics = defaultdict(float)

        for i, batch in enumerate(data, 1):
            batch_metrics = self.do_eval_step(batch=batch, model=model)

            for metric, val in batch_metrics.items():
                # moving average: CMA_{n+1} = CMA_n + (x_{n+1} - CMA_n) / (n + 1)
                avg_metrics[metric] += (val - avg_metrics[metric]) / i

        if self._tensorboard_writer:
            for metric, val in avg_metrics.items():
                self._tensorboard_writer.add_scalar(f'test/avg_{metric}/{self.name}', val, self._round_num)
            self._round_num += 1

        model.cpu()

        return WorkerEvalResult(avg_metrics)

    def do_eval_step(self, batch: Any, model: nn.Module) -> Dict[str, float]:
        """Evaluate the model on the batch.

        Args:
            batch (Any): single batch of data for evaluation.
            model (nn.Module): model to be evaluated.

        Returns:
            Dict[str, float]: a dictionary with all the metrics computed, where the keys are the metrics' names, and values are the results of the computation.
        """
        return NotImplemented

    def get_model_update(self) -> ModelUpdate:
        """Calculate the model update after performed training.

        Returns:
            ModelUpdate: object, containing the model update values as well as other metainformation about the training process.
        """
        return ModelUpdate(
                values=self.calculate_model_update(self.base_model),
                num_examples=self.num_examples,
        )

    def reset_local_model(self, model: nn.Module) -> None:
        """Get the parameters of the provided model and applies them to the local model.

        Args:
            model (torch.nn.Module): a model, the parameters of which should be used in the local one.
        """
        # We cannot just remove the local model and set the new one in place of it, because the local
        # optimizer holds the reference to the parameters of the currect local model. Removing the
        # current local model object would require resetting the optimizer.
        set_model_params(self.local_model, model.parameters())

        # We also record the given model as the base model, to be able to calculate the difference later.
        self.base_model = model

    def reset(self, model: Optional[nn.Module] = None) -> None:
        """Reset the worker. May be used to prepare for a new cycle.

        Args:
            model (Optional[nn.Module]): a new version of the shared model. Defaults to None.
        """
        self.num_examples = 0
        self.epochs_done = 0

        if model:
            self.reset_local_model(model)

    def calculate_model_update(self, base_model: nn.Module) -> ModelDiff:
        """Find the difference between the local model weights and base model weights.

        Args:
            base_model (torch.nn.Module): the base model before the local update.

        Returns:
            List[torch.Tensor]: difference between local updated model's and base model's weight values.
        """
        return ModelDiff(
                [
                        p_old.data - p_new.data
                        for p_old, p_new in zip(base_model.parameters(), self.local_model.parameters())
                ]
        )

    def save_checkpoint(self, to: Optional[str] = None, **extra) -> str:
        """Save the state of this worker.

        Args:
            to (Optional[str], optional): directory to save the state to. If `None`, then creates a subfolder in the current working directory, named same as the worker. Defaults to None.
            **extra (Dict, optional): other parameters to be saved.

        Returns:
            str: path to the folder with the worker's state.
        """
        to = to or self.name
        os.makedirs(to, exist_ok=True)

        # save the local model
        torch.save(self.local_model, os.path.join(to, "model.pt"))

        # save the optimizer state
        torch.save(self.optim.state_dict(), os.path.join(to, "optim.pickle"))

        # save other parameters
        torch.save(extra, os.path.join(to, "params.pickle"))

        return to

    def load_from_checkpoint(self, d: Optional[str] = None) -> None:
        """Load the state of the worker form the checkpoint.

        Args:
            d (Optional[str], optional): path to folder containing the saved state. If `None` is provided, then uses the workers name. Defaults to None.
        """
        d = d or self.name

        # load the local model
        model_params = nn.Module.parameters(torch.load(os.path.join(d, "model.pt")))

        set_model_params(self.local_model, model_params)

        # load the optimizer
        self.optim.load_state_dict(dict(torch.load(os.path.join(d, "optim.pickle"))))

        # load other parameters
        params = dict(torch.load(os.path.join(d, "params.pickle")))
        for attr, p in params.items():
            setattr(self, attr, p)

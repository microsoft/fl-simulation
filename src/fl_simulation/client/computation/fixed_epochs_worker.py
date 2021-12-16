"""A worker performing fixed number of epochs."""

from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .worker import Worker


class FixedEpochsWorker(Worker):
    """A worker with the fixed number of epochs to be completed."""

    num_epochs: int
    """Number of epochs this worker should carry out."""

    def __init__(
            self,
            base_model: nn.Module,
            num_epochs: int,
            loss_fn: Callable[..., Any],
            name: Optional[str] = None,
            device: torch.device = torch.device("cpu"),
            tensorboard_writer: SummaryWriter = None,
            *opt_arg,
            **opt_kwarg
    ) -> None:
        """Create a new `FixedEpochsWorker` instance.

        Args:
            base_model (torch.nn.Module): initial local model.
            num_epochs (int): number of epochs to be carried out.
            loss_fn (Callable[..., Any]): a loss function to be used in training.
            name (Optional[str], optional): the name of this worker. If `None` is provided, uses a UUID. Defaults to `None`.
            device (torch.device): pytorch device to use. Defaults to CPU.
            opt_args: optimizer's positional arguments.
            opt_kwargs: optimizer's keyword arguments.
        """
        super().__init__(
                base_model,
                loss_fn,
                name=name,
                device=device,
                tensorboard_writer=tensorboard_writer,
                *opt_arg,
                **opt_kwarg
        )
        self.num_epochs = num_epochs

    def continue_training(self) -> bool:
        """Retrun `True` if the next epoch should be carried out, `False` otherwise.

        Returns:
            bool: whether to continue training.
        """
        return self.epochs_done < self.num_epochs

    def save_checkpoint(self, to: Optional[str], **extra) -> str:
        return super().save_checkpoint(to=to, num_epochs=self.num_epochs, **extra)

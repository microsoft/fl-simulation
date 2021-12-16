"""A worker for FedProx algorithm."""

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.optim import SGD, Optimizer
from torch.utils.tensorboard import SummaryWriter

from .worker import Worker


class FedProxWorker(Worker):
    """FedProx worker based on "Federated Optimization in Heterogeneous Networks", Li et al."""

    _mu_2: float
    """Proximal term coefficient mu divided by 2."""

    @property
    def mu(self) -> float:
        """Proximal term coefficient."""
        return self._mu_2 * 2

    @mu.setter
    def mu(self, value) -> None:
        self._mu_2 = value / 2

    @property
    def loss_fn(self) -> Callable[..., torch.Tensor]:
        """Get the loss function used by this worker."""
        return self._comp_loss_with_prox

    @loss_fn.setter
    def loss_fn(self, new_loss_fn):
        self._loss_fn_impl = new_loss_fn

    def __init__(
            self,
            base_model: nn.Module,
            loss_fn: Callable[..., torch.Tensor],
            mu: float = 0.01,
            name: Optional[str] = None,
            device: torch.device = torch.device("cpu"),
            tensorboard_writer: SummaryWriter = None,
            *opt_arg,
            **opt_kwarg
    ) -> None:
        """Create a new `FedProxWorker` instance.

        Args:
            base_model (torch.nn.Module): initial local model.
            loss_fn (Callable[..., torch.Tensor]): loss function to be used.
            mu (float, optional): Proximal term coefficient. Defaults to 0.01.
            name (Optional[str], optional): the name of this worker. If `None` is provided, uses a UUID. Defaults to `None`.
            device (torch.device): pytorch device to use. Defaults to CPU.
            tensorboard_writer (Optional[SummaryWriter]): A logger for TensorBoard.
            opt_arg: positional arguments for the local optimizer.
            opt_kwarg: keyword arguments for the local optimizer.
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
        self._mu_2 = mu / 2

    def _comp_loss_with_prox(self, *args, **kwargs) -> torch.Tensor:
        """Loss function with proximal term."""
        loss = self._loss_fn_impl(*args, **kwargs)
        prox_term = sum(
                (w_new - w_old).square().sum()
                for w_new, w_old in zip(self.local_model.parameters(), self.base_model.parameters())
        )

        return loss + self._mu_2 * prox_term

    def get_opt(self, *opt_arg, **opt_kwarg) -> Optimizer:
        """Get the local optimizer. Uses `SGD` optimizer. Notice, that FedProx does not make assumptions about the local solver.

        Returns:
            Optimizer: optimizer to be used.
        """
        return SGD((p for p in self.local_model.parameters() if p.requires_grad), *opt_arg, **opt_kwarg)

    def save_checkpoint(self, to: Optional[str], **extra) -> str:
        return super().save_checkpoint(to=to, _mu_2=self._mu_2, **extra)

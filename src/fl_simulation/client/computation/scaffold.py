"""A worker for Stochastic Controlled Averaging for Federated Learning algorithm."""

from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fl_simulation.server.update import ScaffoldServerUpdate
from fl_simulation.client.update import ModelUpdate, ScaffoldModelUpdate
from fl_simulation.utils.types import ControlVarDiff
from fl_simulation.utils.model import set_model_gradients

from .fixed_epochs_worker import FixedEpochsWorker


class ScaffoldWorker(FixedEpochsWorker):
    """SCAFFOLD worker based on "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" by Karimireddy et al."""

    ctl_var: List[torch.Tensor]
    """Control variate of this client. Corresponds to $c_i$ in the paper."""
    _approx_ctl_var_update: bool
    """Controls whether the control variate's update should be computed by taking the gradient
    at the server model (`False`), or re-use previously computed gradients (`True`)."""

    def __init__(
            self,
            base_model: nn.Module,
            num_epochs: int,
            loss_fn: Callable[..., Any],
            approx_ctl_var_update: bool = True,
            name: Optional[str] = None,
            device: torch.device = torch.device("cpu"),
            tensorboard_writer: SummaryWriter = None,
            *opt_arg,
            **opt_kwarg
    ) -> None:
        """Create a new `ScaffoldWorker` instance. See "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" by Karimireddy et al.

        Args:
            base_model (torch.nn.Module): initial shared model.
            num_epochs (int): number of epochs to be performed.
            loss_fn (Callable[..., Any]): loss function to be used.
            approx_ctl_var_update (bool, Optional): whether the control variate's update should be computed by taking the gradient
                at the server model (`False`), or re-use previously computed gradients (`True`). Defaults to `True`.
            name (Optional[str], optional): the name of this worker. If `None` is provided, uses a UUID. Defaults to `None`.
            device (torch.device): pytorch device to use. Defaults to CPU.
            tensorboard_writer (Optional[SummaryWriter]): A logger for TensorBoard.
            opt_arg: positional arguments for the local optimizer.
            opt_kwarg: keyword arguments for the local optimizer.
        """
        super().__init__(
                base_model,
                num_epochs,
                loss_fn,
                name=name,
                device=device,
                tensorboard_writer=tensorboard_writer,
                *opt_arg,
                **opt_kwarg
        )
        # Initialize the control variate with zeros.
        self.ctl_var = [torch.zeros_like(p.data) for p in base_model.parameters()]

        # Choose the control variate's update type.
        self._approx_ctl_var_update = approx_ctl_var_update

    def do_cycle(
            self,
            srv_update: ScaffoldServerUpdate,
            data_loader: DataLoader,
    ) -> ModelUpdate:
        """Perform a single round of local update (i.e. do the specified number of local epochs).

        This would return the model updates, intended to be sent to the server. Can only be used if `epoch_num` was
        set in configuration.

        Args:
            srv_update (ScaffoldServerUpdate): the most recent update from the server, which includes the most recent
                versions of the shared model and the server's control variate.
            data_loader (torch.utils.data.DataLoader): local training data.

        Returns:
            ModelUpdate: resulting model update.
        """
        self.reset(srv_update.shared_model)

        self.local_model.to(self.device)
        self.base_model.to(self.device)

        # c - c_i
        correction_term = ControlVarDiff(
                [c.to(self.device) - c_i.to(self.device) for c, c_i in zip(srv_update.ctl_var, self.ctl_var)]
        )

        while self.continue_training():
            loss = None
            for data in data_loader:
                # Here we split the model update in two parts, noticing that
                # weights := weights - lr*(grad + correction_term)
                # <=>
                # weights := weights - lr * grad
                # weights := weights - lr * correction_term
                # So instead of applying the corrected gradient, we first apply unmodified gradient
                # and then apply the correction.
                # FIXME won't work with stateful optimizers (if they change lr after every step)

                # Apply mini-batch gradient
                loss = self.do_optim_step(data)

                # Set the gradient to the correction term.
                set_model_gradients(self.local_model, correction_term)

                # Apply correction term.
                self.optim.step()

                self.num_examples += data_loader.batch_size or 0
            if loss is not None and self._tensorboard_writer:
                self._tensorboard_writer.add_scalar('train/loss', loss, self.epochs_done)
            self.epochs_done += 1

        self.num_examples //= self.epochs_done

        self.local_model.cpu()
        self.base_model.cpu()

        if self._approx_ctl_var_update:
            ctl_var_update = self.update_ctl_var_approx(correction_term)
        else:
            ctl_var_update = self.update_ctl_var_grad(data_loader)

        return self.get_model_update(ctl_var_update)

    def get_model_update(self, ctl_var_update: ControlVarDiff) -> ScaffoldModelUpdate:
        """Calculate the model update after performed training.

        Returns the updates, which should be applied by subtracting them, not adding as in the SCAFFOLD paper.

        Args:
            ctl_var_update (List[torch.Tensor]): the control variate's update. `c_i - c_i^+`.

        Returns:
            ScaffoldModelUpdate: object, containing the model update values as well as other metainformation about the training process.
        """
        return ScaffoldModelUpdate(
                self.calculate_model_update(self.base_model),
                ctl_var_update=ctl_var_update,
                num_examples=self.num_examples,
        )

    def update_ctl_var_grad(self, dl: DataLoader) -> ControlVarDiff:
        """Update control variate by taking gradient at the server model and return the difference between the updated\
        control variate and old one. Should be implemented analogously to `do_optim_step` method.

        Args:
            dl (DataLoader): local data loader.

        Returns:
            ControlVarDiff: difference between the updated control variate and old one.
        """
        self.local_model.to(self.device)

        # the number of dataset in batches
        l = 1

        # if we do not call `zero_grad` the gradient gets accumulated (i.e. summed)
        # see https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch#:~:text=In%20PyTorch%2C%20we%20need%20to%20set%20the%20gradients,sum%29%20the%20gradients%20on%20every%20loss.backward%20%28%29%20call.
        for l, data in enumerate(dl, start=1):
            self.backprop(data)

        self.local_model.cpu()

        # we divide the grad by the number of batches in the dataset, so that the accumulated gradient is divided by the number of
        # datapoints (almost), so that the result is an unbiased estimator
        return ControlVarDiff(
                [
                        p.grad / l if p.grad is not None else torch.zeros_like(p.data)
                        for p in self.local_model.parameters()
                ]
        )

    def update_ctl_var_approx(self, correction_term: ControlVarDiff) -> ControlVarDiff:
        """Update control variate by re-using the previously computed gradients and return difference between the\
        updated control variate and old one.

        Args:
            correction_term (ControlVarDiff): difference between server's control variate and the client's
                one. `c - c_i`.

        Returns:
            ControlVarDiff: difference between the updated control variate and old one.
        """
        # FIXME repetitive calculation.
        model_update = self.calculate_model_update(self.base_model)

        new_ctl_var = []
        i = 0

        # go over all parameter groups and compute the control variate's update in correspondence with
        # the learning rate for that parameter group
        for pg in self.optim.param_groups:
            lr = pg.get("lr", 1)
            coeff = 1 / (self.num_epochs * lr)
            n_params = len(pg["params"])

            new_ctl_var.extend(
                    coeff * update - corr.to(update) for update, corr in zip(
                            model_update[i:i + n_params],
                            correction_term[i:i + n_params],
                    )
            )
            i += n_params

        # c_i - c_i^+
        ctl_var_update = [c_old - c_new for c_old, c_new in zip(self.ctl_var, new_ctl_var)]

        self.ctl_var = new_ctl_var

        return ControlVarDiff(ctl_var_update)

    def save_checkpoint(self, to: Optional[str], **extra) -> str:
        return super().save_checkpoint(
                to=to, ctl_var=self.ctl_var, _approx_ctl_var_update=self._approx_ctl_var_update, **extra
        )

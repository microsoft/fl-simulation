"""Data types for model updates."""

from dataclasses import dataclass, field
from typing import Generic, Hashable, Optional, TypeVar

from fl_simulation.utils.types import ModelDiff, ControlVarDiff

T_wid = TypeVar("T_wid", bound=Hashable)


@dataclass
class ModelUpdate(Generic[T_wid]):
    """The class containing the model update."""

    values: ModelDiff
    """List of gradients or differences between old model weights and the locally updated ones."""

    num_examples: int
    """Number of the local examples this client has."""

    worker_id: Optional[T_wid] = field(default=None, init=False)
    """The worker that made this update."""


@dataclass
class FedAvgModelUpdate(ModelUpdate):
    """The class containing model update for Federated Averaging."""

    pass


@dataclass
class ScaffoldModelUpdate(FedAvgModelUpdate):
    """The class containing model update for SCAFFOLD."""

    ctl_var_update: ControlVarDiff
    """For SCAFFOLD. The difference between the value of the old local control variate and the updated one."""

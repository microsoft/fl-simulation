"""Representation for the server updates: new version of the shared model and additional metadata."""
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from fl_simulation.utils.types import ModelDiff, ControlVarDiff


@dataclass
class ServerUpdate:
    """The most basic server update."""

    shared_model: nn.Module
    """New version of the shared model."""


@dataclass
class ScaffoldServerUpdate(ServerUpdate):
    """Server update used in the SCAFFOLD algorithm."""

    ctl_var: List[torch.Tensor]
    """Updated control variate of the server."""


@dataclass
class AggregatedUpdate:
    """Result of aggregation of model updates."""

    model_diff: ModelDiff


@dataclass
class ScaffoldAggregatedUpdate(AggregatedUpdate):
    """Result of aggregation of model updates for the SCAFFOLD algorithm."""

    ctl_var_diff: ControlVarDiff
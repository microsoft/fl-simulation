"""Useful type decalrations."""

from typing import List, NewType

import torch

ModelDiff = NewType("ModelDiff", List[torch.Tensor])
"""The model weights update."""

ControlVarDiff = NewType("ControlVarDiff", List[torch.Tensor])
"""The control variate update."""
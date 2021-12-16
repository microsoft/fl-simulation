"""Utilities for working with models in a federated setting."""

from typing import Iterable, Sequence

import torch
import torch.nn as nn


def set_model_params(model: nn.Module, params: Iterable[nn.parameter.Parameter]) -> nn.Module:
    """Set the parameters in the specified model.

    Args:
        model (torch.nn.Module): the model to be modified.
        params (Iterable[torch.nn.parameter.Parameter]): new parameters for the model.

    Returns:
        torch.nn.Module: the updated model.
    """
    for p, p_new in zip(model.parameters(), params):
        # we need `clone`, so that `p.data` and `p_new.data` do not share the same memory
        # and `detach`, so that there is no gradient from between them.
        assert p.data.shape == p_new.data.shape, "tried to assign data with different shape"

        p.data = p_new.data.clone().detach()
    return model


def set_model_gradients(model: nn.Module, grads: Iterable[torch.Tensor]) -> nn.Module:
    """Set the gradients for the model parameters to be those of the parameters in `grads`.

    Args:
        model (torch.nn.Module): the model for which the gradients should be updated.
        grads (Iterable[torch.Tensor]): values of the gradients to be used.

    Returns:
        torch.nn.Module: model with gradients set
    """
    for p, grad in zip(model.parameters(), grads):
        assert p.data.shape == grad.shape, "gradient shape is different from the parameter data shape"

        p.grad = grad.detach().clone()
    return model

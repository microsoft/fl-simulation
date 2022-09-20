from typing import Optional

import torch


def custom_softmax(x: torch.Tensor, dim: int, offset: Optional[float] = None) -> torch.Tensor:
    """
    A custom softmax function to use when computing the gradient graph because the built-in PyTorch softmax because it doesn't work with ONNX Runtime Web when extracting the gradient graph.
    Args:
        x (torch.Tensor): The input tensor.
        dim (int): The dimension to apply the softmax to.
        offset (float, optional): An optional offset to subtract from the input tensor before applying the softmax.
        Normally, this is the maximum value in the input tensor, `(1 - mean) / std` can work well too.
    """
    # Can't use the -max trick for stability because we get an error when exporting the gradient graph.
    # Offset by the max possible value to avoid overflow.
    # result = torch.exp(x - x.max(dim=dim, keepdim=True)[0])

    if offset:
        result = torch.exp(x - offset)
    else:
        result = torch.exp(x)
    result /= result.sum(dim=dim, keepdim=True)
    return result

"""Different useful metrics."""

from typing import Sequence, Union

import numpy as np


def mrr_score(y_true: Sequence[Union[float, int]], y_score: Sequence[Union[float, int]]) -> float:
    """Compute the MRR score.

    Args:
        y_true (Sequence[Union[float, int]]): labels.
        y_score (Sequence[Union[float, int]]): predictions.

    Returns:
        float: mean reciprocal rank.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return float(np.sum(rr_score) / (np.sum(y_true) or 1))

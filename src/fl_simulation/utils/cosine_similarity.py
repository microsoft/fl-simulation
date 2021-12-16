"""Similarity and distance functions based on cosine."""

from torch import tensor
from torch.linalg import norm

from fl_simulation.client.update import ModelDiff


def cosine_similarity_model_diff(u1: ModelDiff, u2: ModelDiff) -> float:
    """Find cosine similarity of two model diffs.

    Args:
        u1 (ModelDiff): A model diff to compare.
        u2 (ModelDiff): The other model diff to compare.

    Returns:
        float: The cosine similarity of the two model diffs as if each one was flattened to a vector.
    """

    # Avoid building large new tensors.
    # Process tensors once in order to avoid returning to them later (for memory access efficiency).
    result = tensor([0.])
    u1_magnitude = tensor([0.])
    u2_magnitude = tensor([0.])
    for t1, t2 in zip(u1, u2):
        result += (t1 * t2).sum()
        u1_magnitude += norm(t1) ** 2
        u2_magnitude += norm(t2) ** 2
    if not u1_magnitude.is_nonzero() or not u2_magnitude.is_nonzero():
        return 0
    result /= (u1_magnitude.sqrt() * u2_magnitude.sqrt())
    return result.item()


def cosine_distance_model_diff(u1: ModelDiff, u2: ModelDiff) -> float:
    """Cosine distance between two model diffs.

    Args:
        u1 (ModelDiff): A model diff to compare.
        u2 (ModelDiff): The other model diff to compare.

    Returns:
        float: cosine distance between the two model diffs as if each one was flattened to a vector.
    """
    return 1.0 - cosine_similarity_model_diff(u1, u2)

"""Model assigner."""

from collections import defaultdict
from typing import Any, Callable, Collection, Dict, Generic, Mapping, Optional, Sequence, Tuple

import torch
from typing_extensions import Protocol

from fl_simulation.client.update import ModelUpdate
from fl_simulation.server.update import AggregatedUpdate
from fl_simulation.utils.cosine_similarity import cosine_distance_model_diff
from fl_simulation.utils.types import ModelDiff

from ._types import T_update

from sklearn.cluster import DBSCAN


class ModelAssigner(Protocol, Generic[T_update]):
    """Assigns users and model updates to the existing models."""

    def __call__(
            self, abnormal_updates: Collection[T_update], representatives: Dict[int, AggregatedUpdate], *args: Any,
            **kwargs: Any
    ) -> Mapping[int, Collection[T_update]]:
        """Assign abnormal updates to existing models or make a decision to split the model. Implicitly updates `_user_assignments`.

        Args:
            abnormal_updates (Collection[T_update]): abnormal updates from user with no existing model assigned.
            representatives (Dict[int, AggregatedUpdate]): updates representing each of the branch off models,
                e.g. average of all previous updates.                

        Returns:
            Mapping[int, Collection[T_update]]: mapping from model indexes to the collections of updates which should be 
                applied to the model with that index. If index does not exist, then a new model should be spawned. 
        """
        ...


class ClustorizationAlg(Protocol):
    """An interface for clustorization algorithms."""

    def fit_predict(self, X, y=None, sample_weight=None) -> Collection:
        """Cluster the samples.

        Args:
            X: samples.
            y (optional): labels. Defaults to None.
            sample_weight (optional): weight of each label. Defaults to None.

        Returns:
            Collection: predicted labels. Has the same length as there are labels. Unclustered samples have label `-1`
                assigned.
        """
        ...


class DistanceBasedModelAssigner(ModelAssigner, Generic[T_update]):
    """A model assigner making decisions based on distance between the new updates and the past ones.
    
    Either assigns the worker and their update to the model for which the average of all past updates is the closest to 
    that worker's update, or if such model cannot be found, indicates that a new model should be spawn. The closeness is
    based on the metric function provided and the threshold: if the metric returns a value smaller then the threshold, 
    the updates are deemed close.
    """

    _metric: Callable[[ModelDiff, ModelDiff], float]
    """A function used to find distance between two model updates"""
    _threshold: float
    """If metric returns result less or equal then this threshold, updates are deemed close."""
    _clutsorization_alg: ClustorizationAlg

    def __init__(
            self,
            metric: Callable[[ModelDiff, ModelDiff], float] = cosine_distance_model_diff,
            threshold: float = 0.1,
            clutsorization_alg: Optional[ClustorizationAlg] = None
    ) -> None:
        """Create a new `SimilarityBasedModelAssigner`.

        Args:
            metric (Callable[[ModelDiff, ModelDiff], float], optional): a metric function used when 
                trying to assign newly participating users to existing models. Defaults to `cosine_distance_model_diff`.
            threshold (float, optional): sensitivity of the metric. If the metric returns a result less then 
                this threshold, updates are deemed close. Defaults to `0.1`.
            clutsorization_alg (Optional[ClustorizationAlg], optional): an algorithm for clustering unassigned abnormal 
                updates. If `None`, then uses `DBSCAN` with "cosine" as `metric` and `threshold` as `eps`. Defaults to `None`.  
        """
        self._metric = metric
        self._threshold = threshold

        self._clutsorization_alg = clutsorization_alg or DBSCAN(eps=self._threshold, min_samples=3, metric="cosine")

    def __call__(
            self, abnormal_updates: Collection[T_update], representatives: Dict[int, AggregatedUpdate], *args: Any,
            **kwargs: Any
    ) -> Mapping[int, Collection[T_update]]:
        """Assign abnormal updates to existing models or indicate that is new model is required.

        Args:
            abnormal_updates (Collection[T_update]): abnormal updates.
            representatives (Dict[int, AggregatedUpdate]): representatives of the exiting models. E.g., aggregate of 
                past updates.

        Returns:
            Mapping[int, Collection[T_update]]: mapping from representative's index to the collection of updates 
                assigned to it. If there is no representative with such index, then a new model should be spawn for
                those updates. 
        """
        processed_updates = defaultdict(list)
        unprocessed_updates = []

        for u in abnormal_updates:
            if 0 in representatives:
                # if there is already a representative of the sanitized model
                # we shift the update by the update representative of the sanitized model to get and update as if it started from the
                # very initial sanitized model. Then it would be comparable with other update representatives, as they approximate an update
                # which would bring the very initial sanitized model to the given spawned one.
                shifted_update_values = ModelDiff([t1 + t2 for t1, t2 in zip(u.values, representatives[0].model_diff)])
            else:
                # otherwise use the update right away
                shifted_update_values = u.values

            # find the index of the closest representative
            model_idx, _ = min(
                    # filter out update representatives far to the processed update
                    filter(
                            lambda r: r[1] <= self._threshold,
                            # map update_representative to distance from the currently processed updated
                            map(
                                    lambda r: (r[0], self._metric(shifted_update_values, r[1].model_diff)),
                                    # filter out representative of the sanitized model: no one can be assigned to it
                                    filter(lambda r: r[0] != 0, representatives.items())
                            )
                    ),
                    # get the representative closest to the update
                    key=lambda r: r[1],
                    default=(None, None)
            )

            if model_idx is not None:
                # if we have found a model for the given update, store that update at that model's index
                processed_updates[model_idx].append(u)
            else:
                # if we have not found a model for that update, store for future processing.
                unprocessed_updates.append(u)

        clustering_predictions = []
        if len(unprocessed_updates) > 0:
            unprocessed_updates_flattened = torch.stack(
                    [torch.cat([t.reshape(-1) for t in upd.values]) for upd in unprocessed_updates]
            )
            # find clusters for unprocessed updates
            clustering_predictions = self._clutsorization_alg.fit_predict(unprocessed_updates_flattened)

        # group updates based on the assigned cluster
        clustered_updates = defaultdict(list)
        for l, u in zip(clustering_predictions, unprocessed_updates):
            clustered_updates[l].append(u)

        # pop updates without assigned cluster
        next_nonexistant_idx = -1
        noisy_updates = clustered_updates.pop(-1, None)

        if noisy_updates is not None:
            # add all noisy updates to a single cluster
            processed_updates[next_nonexistant_idx] = noisy_updates
            next_nonexistant_idx -= 1

        # add clustered abnormal users to nonexisting indexes to indicate that a new model should be spawn
        for updates in clustered_updates.values():
            processed_updates[next_nonexistant_idx] = updates
            next_nonexistant_idx -= 1

        return processed_updates

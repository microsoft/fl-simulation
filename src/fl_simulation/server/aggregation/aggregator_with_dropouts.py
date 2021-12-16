"""Aggregation with device dropout simulation."""

import random
from typing import Collection, Generic, Hashable, NoReturn, Optional, TypeVar

from fl_simulation.client.update import ModelUpdate
from fl_simulation.server.aggregation.aggregator import Aggregator
from fl_simulation.server.update import ServerUpdate

T_aggr = TypeVar("T_aggr", bound=Aggregator)


class AggregatorWithDropouts(Aggregator, Generic[T_aggr]):
    """A wrapper around aggregators to simulate device dropouts. Should only be used for simulation."""

    _aggr: T_aggr
    """The wrapped aggregator."""
    _p: float
    """Probability of device dropping out."""

    @property
    def model(self):
        return self._aggr.model

    @model.setter
    def model(self, v):
        self._aggr.set_shared_model(v)

    @property
    def global_optim(self):
        return self._aggr.global_optim

    @property
    def device(self):
        return self._aggr.device

    @device.setter
    def device(self, d):
        self._aggr.device = d

    def __init__(self, aggregator: T_aggr, dropout_probability: float) -> None:
        """Initialize `AggregatorWithDropouts`.

        Args:
            aggregator (Aggregator): an underlying aggregator.
            dropout_probability (float): the propability of a device dropping out.

        Raises:
            ValueError: if `dropout_probability` is not in [0.0, 1.0].
        """
        if dropout_probability < 0.0 or dropout_probability > 1.0:
            raise ValueError("dropout probability should be between 0.0 and 1.0")

        self._aggr = aggregator
        self._p = dropout_probability

    def get_opt(self) -> NoReturn:
        """Not used in `AggregatorWithDropouts`."""
        raise NotImplementedError("get_opt is not used in this implementation")

    def aggr_fn(self, updates: Collection[ModelUpdate]) -> NoReturn:
        """Not used in `AggregatorWithDropouts`."""
        raise NotImplementedError("aggr_fn is not used in this implementation")

    def dropout_updates(self,
                        updates: Collection[ModelUpdate],
                        per_update_prob: Optional[Collection[float]] = None) -> Collection[ModelUpdate]:
        """Dropout some of the received model updates.

        Args:
            updates (Collection[ModelUpdate]): all model updates.
            per_update_prob (Optional[Collection[float]], optional): propability of being dropped for each update. 
                If `None`, uses the probability provided on initialization for all updates. Defaults to `None`.

        Raises:
            ValueError: if `per_update_prob` is present, but does not match the size of `updates`.

        Returns:
            Collection[ModelUpdate]: remaining updates after dropping out some of the initial ones.
        """
        if per_update_prob is not None and len(per_update_prob) != len(updates):
            raise ValueError("if per_update_prob is specified it should provide probabilities for all updates")
        if per_update_prob is not None:
            updates = [u for u, p in zip(updates, per_update_prob) if random.random() >= p]
        else:
            updates = [u for u in updates if random.random() >= self._p]

        return updates

    def aggregate(self, updates: Collection[ModelUpdate], per_update_prob: Optional[Collection[float]] = None) -> None:
        """Aggregate the model updates after dropping out some of them.

        Args:
            updates (Collection[ModelUpdate]): all model updates.
            per_update_prob (Optional[Collection[float]], optional): propability of being dropped for each update. 
                If `None`, uses the probability provided on initialization for all updates. Defaults to `None`.
        """
        updates = self.dropout_updates(updates, per_update_prob=per_update_prob)

        self._aggr.aggregate(updates)

    def save_checkpoint(self, to: Optional[str] = None, **extra) -> str:
        return self._aggr.save_checkpoint(to=to, _p=self._p, **extra)

    def load_from_checkpoint(self, d: Optional[str] = None) -> None:
        self._aggr.load_from_checkpoint(d=d)
        self._p = getattr(self._aggr, "_p")
        delattr(self._aggr, "_p")

    def get_server_update(self, _worker_id: Optional[Hashable] = None) -> ServerUpdate:
        return self._aggr.get_server_update(_worker_id)

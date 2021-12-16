"""An instance of MultiModelAggregator."""

import copy
from collections import defaultdict
from itertools import chain, repeat
from pathlib import Path
from typing import Any, Collection, Dict, Generic, NoReturn, Optional, cast

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from fl_simulation.incentive.incentive_mechanism import IncentiveMechanism
from fl_simulation.logging.logger import get_logger
from fl_simulation.server.aggregation.aggregator import Aggregator
from fl_simulation.server.update import AggregatedUpdate, ServerUpdate
from fl_simulation.utils.types import ModelDiff

from ._types import T_aggregator, T_update, T_wid
from .aggregator_factory import AggregatorFactory
from .detector import Detector
from .model_assigner import ModelAssigner, DistanceBasedModelAssigner
from fl_simulation.utils.model import set_model_params

logger = get_logger()


class _PseudoGradCacher(Aggregator, Generic[T_aggregator]):
    """Proxies all the methods to the wrapped aggregator, but stores the most recent aggregated update."""

    last_update: Optional[AggregatedUpdate] = None
    last_epoch_updated: int = -1
    """Last epoch during which the aggregator got updated."""

    def __init__(self, wrapped_aggr: T_aggregator) -> None:
        """Create a `_PseudoGradCacher`.

        Args:
            wrapped_aggr (T_aggregator): wrapped aggregator
        """
        self.aggr = wrapped_aggr

    def aggr_fn(self, *args, **kwargs) -> AggregatedUpdate:
        update = self.aggr.aggr_fn(*args, **kwargs)
        self.last_update = copy.deepcopy(update)

        return update

    def get_opt(self) -> Optimizer:
        return self.aggr.get_opt()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.aggr, name)


class MultiModelAggregator(Aggregator, Generic[T_update, T_aggregator, T_wid]):
    """An Aggregator spawning a new model each time abnormal update is detected."""

    _detector: Detector[T_update]
    """Detector of abnormal updates."""
    _aggregator_factory: AggregatorFactory[_PseudoGradCacher[T_aggregator]]
    """Builds new aggregators to spawn new models."""
    # it is assumed that all the keys are >= 0
    _aggregators: Dict[int, _PseudoGradCacher[T_aggregator]]
    """Existing models."""
    _user_assignments: Dict[T_wid, int]
    """User assignments to the models. Note, that there must not be any assignment to the sanitized model."""
    _model_assigner: ModelAssigner[T_update]
    """Assigns updates and workers to existing models and requests new model splits based on updates."""
    _representatives: Dict[int, AggregatedUpdate]
    """Model diffs based on all updates to corresponding aggregator in `_aggregators`."""
    _max_branches: int
    """Maximum spawned models."""
    _number_of_rounds: int = 0
    """Number of rounds performed."""

    def __init__(
            self,
            initial_model: nn.Module,
            anomaly_detector: Detector[T_update],
            aggregator_factory: AggregatorFactory[T_aggregator],
            model_assigner: Optional[ModelAssigner[T_update]] = None,
            max_branches: int = -1,
            incentive_mechanism: Optional[IncentiveMechanism] = None,
            device: torch.device = torch.device("cpu")
    ) -> None:
        """Create a new `MultiModelAggregator`.

        Args:
            initial_model (nn.Module): initial model to work with.
            anomaly_detector (Detector[T_update]): anomaly detector, which will be used to distinguish abnormal updates
                from normal ones.
            aggregator_factory (AggregatorFactory[T_aggregator]): an aggregator factory used to spawn new aggregators 
                when there is a model split required.
            model_assigner (Optional[ModelAssigner[T_update]], optional): a callable, which assigns updates 
                and workers to existing models and requests new model split. If not provided (equals `None`), then
                `DistanceBasedModelAssigner` with default arguments is used.
            max_branches (int, optional): maximum number of spawned models, including the sanitized one. If 
                non-positive, the number of spawned models is unrestricted, eviction and merging policies, if present
                invoked on every aggregation invocation. Defaults to -1.  
            incentive_mechanism (Optional[IncentiveMechanism], optional): reward tracker.
            device (torch.device, optional): device used for computation. Defaults to `torch.device("cpu")`.
        """
        self.device = device
        self.incentive_mechanism = incentive_mechanism

        self._detector = anomaly_detector
        self._user_assignments = {}
        self._model_assigner = model_assigner or DistanceBasedModelAssigner()
        self._max_branches = max_branches

        self._representatives = {}

        # rewrite the aggregation factory so that it would return wrapped aggregator right away
        def aggr_fact(intial_model, device=torch.device("cpu"), *args, **kwds):
            aggr = aggregator_factory(intial_model, device, *args, **kwds)
            return _PseudoGradCacher(aggr)

        self._aggregator_factory = aggr_fact

        # The aggregator at index 0 holds a sanitized model and never gets evicted.
        self._aggregators = {0: self._aggregator_factory(initial_model, self.device)}

    def get_opt(self) -> NoReturn:
        """Not used in MultimodelAggregator, thus implemented only to follow the aggregator requirements."""
        raise NotImplementedError("get_opt is not implemented for MultiModelAggregator")

    def aggr_fn(self, updates: Collection[T_update]) -> NoReturn:
        """Not used in MultimodelAggregator, thus implemented only to follow the aggregator requirements."""
        raise NotImplementedError("aggr_fn is not implemented for MultiModelAggregator")

    def aggregate(self, updates: Collection[T_update]) -> None:
        """Aggregate the provided updates and apply them to the corresponding model.
        
        If the user (based on worker_id), who provided an update, already has an assigned model, their update is going
        to be aggregated together with other updates for that model and then applied to that model. If the update has
        has no model assigned and was deemed normal by the anomaly detector, then it is applied to the sanitized model.
        If the update is not assigned and was deemed abnormal by the anomaly detector, then, the function tries to
        assign it to one of the existing models based on the closeness of it with previous updates to spawned models.
        If the update cannot be assigned, a new model is spawned and the user is assigned to that model.       

        Args:
            updates (Collection[T_update]): model updates.

        Raises:
            ValueError: if there is an update without `worker_id` set.
        """
        self._number_of_rounds += 1

        normal_updates, abnormal_updates = self._detector(updates)

        # mapping from model index (corresponding to index in self._aggregators) to the list of updates
        updates_assignment = defaultdict(list)

        logger.info("Start multi-model aggregation.")
        for i, u in enumerate(chain(normal_updates, abnormal_updates)):
            if u.worker_id is None:
                raise ValueError("one of the updates does not have `worker_id` set")

            # if the user already has a model assigned, then that update is applied to the assigned model
            # normal updates with no assignment are applied to the sanitized model (at index 0)
            # abnormal updates with no assignment are stored at index -1 for future processing.

            default_idx = 0 if i < len(normal_updates) else -1
            model_idx = self._user_assignments.get(u.worker_id, default_idx)

            # if the assigned model was trimmed, remove the assignment
            if model_idx != default_idx and model_idx not in self._aggregators:
                del self._user_assignments[u.worker_id]
                model_idx = default_idx

            updates_assignment[model_idx].append(u)

        abnormal_unprocessed = updates_assignment.pop(-1, None)

        if abnormal_unprocessed is not None:
            logger.info("%d abnormal updates detected.", len(abnormal_unprocessed))

            # assign abnormal updates to the models or indicate that a new model should be spawn.
            abnormal_processed = self._model_assigner(abnormal_unprocessed, self._representatives)

            models_assigned = list(idx for idx in abnormal_processed if idx in self._aggregators)
            logger.info(
                    f"Abnormal updates processed. Models assigned: {models_assigned}. New models to spawn: {len(abnormal_processed) - len(models_assigned)}"
            )

            could_not_trim_branches = False
            for model_idx, updates in abnormal_processed.items():
                if model_idx not in self._aggregators:
                    # there is no model with such index => spawn a new one

                    # If we already could not remove some branches, no need to try again
                    if could_not_trim_branches:
                        continue

                    # number of spawned models (aggregators) is always >= 1, thus the condition is always true for
                    # max_branches <= 0.
                    if len(self._aggregators) >= self._max_branches:
                        self._try_merge()
                        self._try_evict()

                        # if we had to evict or merge some of the models but failed, drop current cluster, do not
                        # spawn a model, and indicate that we have failed to remove some branches
                        if self._max_branches > 0 and len(self._aggregators) >= self._max_branches:
                            logger.debug(f"failed to trim branches below {self._max_branches}. updates dropped.")
                            could_not_trim_branches = True
                            continue

                    model_idx = len(self._aggregators)
                    logger.info("Spawning a new model for grouped abnormal updates at index %d.", model_idx)

                    initial_model = copy.deepcopy(self._aggregators[0].model)
                    self._aggregators[model_idx] = self._aggregator_factory(initial_model, device=self.device)
                    self._aggregators[model_idx].last_epoch_updated = self._number_of_rounds

                # store abnormal updates at the assigned index
                updates_assignment[model_idx].extend(updates)
                # and add model assignment for the owners of those updates
                self._user_assignments.update(zip((cast(T_wid, u.worker_id) for u in updates), repeat(model_idx)))

        logger.info("Applying model updates to existing models.")

        # apply the updates to the corresponding models
        for model_idx, updates in updates_assignment.items():
            # when updating unsanitized models also use normal updates
            updates = updates if model_idx == 0 else updates + updates_assignment[0]
            self._aggregators[model_idx].aggregate(updates)
            self._aggregators[model_idx].last_epoch_updated = self._number_of_rounds

        self.update_representatives()

    def update_representatives(self) -> None:
        """Update the model representatives.

        Raises:
            ValueError: if the the branched models exist, but were not updated yet.
        """
        logger.info("Updating representatives.")

        for i, aggr in self._aggregators.items():
            last_update = aggr.last_update
            if last_update is None:
                raise ValueError("there was no update yet")

            if i in self._representatives:
                # sum the new update with the last representative to produce a new representative
                v = self._representatives[i].model_diff
            else:
                self._representatives[i] = last_update
                v = self._representatives[0].model_diff

            self._representatives[i].model_diff = ModelDiff([u1 + u2 for u1, u2 in zip(v, last_update.model_diff)])

    def _try_merge(self) -> None:
        # TODO add merge policy
        # models_to_merge = self.merge_policy(
        #         self._aggregators, self._representatives, self._user_assignments, self._number_of_rounds
        # )
        # for to_merge in models_to_merge:
        #     merged_idx = min(to_merge)
        #     merged_aggregators = [self._aggregators[m] for m in to_merge]

        #     merged_model_params = []
        #     for ps in zip(*(a.model.parameters() for a in merged_aggregators)):
        #         ps_data = [p.data for p in ps]
        #         # find the mean of parameters
        #         merged_model_params.append(nn.parameter.Parameter(data=torch.mean(torch.stack(ps_data, dim=0), dim=0)))

        #     # set the model at merged index to the mean of weights of merged models
        #     set_model_params(self._aggregators[merged_idx].model, merged_model_params)
        #     # mark the merged model as updated
        #     self._aggregators[merged_idx].last_epoch_updated = self._number_of_rounds

        #     # remove index of the merged model
        #     to_merge.discard(merged_idx)

        #     # delete merged aggregators
        #     for idx in to_merge:
        #         self._aggregators.pop(idx, None)

        #     # change user assignments to the index of the merged model
        #     for wid, idx in self._user_assignments.items():
        #         if idx in to_merge:
        #             self._user_assignments[wid] = merged_idx
        pass

    def _try_evict(self) -> None:
        # TODO add eviction policy
        # models_to_evict = self.eviction_policy(self._aggregators, self._user_assignments, self._number_of_rounds)
        # for idx in models_to_evict:
        #     self._aggregators.pop(idx, None)
        pass

    def set_shared_model(self, model: nn.Module, idx: int = 0) -> None:
        """Set the shared model at the provided index to the specified one.

        Args:
            model (torch.nn.Module): new version of the sanitized shared model. Must have the same architecture as the
                previous one.
            idx (int, optional): the index of the model to be updated. Defaults to `0`, which means the sanitized model
                is going to be updated.

        Raises:
            KeyError: if there is no model with such index.    
        """
        self._aggregators[idx].set_shared_model(model)

    def get_server_update(self, worker_id: Optional[T_wid] = None) -> ServerUpdate:
        """Get the server update for the worker with specified id.

        Args:
            worker_id (Optional[T_wid], optional): id of the worker for which the update should be provided. If `None` 
                or the worker has not participated in the training before, then the update with the sanitized model is
                returned. Defaults to `None`.

        Returns:
            ServerUpdate: an update containing the latest version of the model.
        """
        model_idx = 0

        if worker_id is not None:
            model_idx = self._user_assignments.get(worker_id, 0)

        srv_update = self._aggregators[model_idx].get_server_update()
        setattr(srv_update, "model_idx", model_idx)

        return srv_update

    def save_checkpoint(self, to: Optional[str] = None, **extra) -> str:
        """Save the state of this aggregator with all the underlying aggregators.

        Args:
            to (Optional[str], optional): directory to save the state to. If `None`, then creates a subfolder in the current working directory, named "aggregator". Defaults to None.
            **extra (Dict, optional): other parameters to be saved.

        Returns:
            str: path to the directory containing saved states.
        """
        base_dir = Path(to or "aggregator")
        base_dir.mkdir(exist_ok=True)

        logger.info("Saving MultiModelAggregator checkpoint to \"%s\"...", str(base_dir))

        for idx, aggr in self._aggregators.items():
            sub_dir = base_dir / str(idx)
            sub_dir.mkdir(exist_ok=True)

            aggr.save_checkpoint(str(sub_dir))

        # notice, that index 0 is going to be the first one
        torch.save(list(self._aggregators), base_dir / "model_indexes.pickle")
        torch.save(self._user_assignments, base_dir / "user_assignments.pickle")
        torch.save(self._representatives, base_dir / "representative.pickle")
        torch.save(
                {
                        "number_of_rounds": self._number_of_rounds,
                        "max_branches": self._max_branches
                }, base_dir / "state.pickle"
        )

        logger.info("MultiModelAggregator checkpoint saved to \"%s\".", str(base_dir))

        return str(base_dir)

    def load_from_checkpoint(self, d: Optional[str] = None) -> None:
        """Load the state of the aggregator and all the internal aggregators form the checkpoint.

        Args:
            d (Optional[str], optional): path to folder containing the saved states. If `None` is provided, then uses "aggregator" as the directory name. Defaults to None.
        """
        base_dir = Path(d or "aggregator")

        logger.info("Loading MultiModelAggregator checkpoint from \"%s\"...", str(base_dir))

        model_indexes = torch.load(base_dir / "model_indexes.pickle")

        for idx in model_indexes:
            # would never be true for index 0, as it is created in __init__. Thus, aggregator creation for index 0
            # never takes place in here, and insdead is created in __init__.
            if idx not in self._aggregators:
                self._aggregators[idx] = self._aggregator_factory(self._aggregators[0].model, device=self.device)

            sub_dir = base_dir / str(idx)
            self._aggregators[idx].load_from_checkpoint(str(sub_dir))

        self._user_assignments = torch.load(base_dir / "user_assignments.pickle")
        self._representatives = torch.load(base_dir / "representative.pickle")
        state = torch.load(base_dir / "state.pickle")

        self._number_of_rounds = int(state["number_of_rounds"])
        self._max_branches = int(state["max_branches"])

        logger.info("Loaded MultiModelAggregator checkpoint from \"%s\".", str(base_dir))

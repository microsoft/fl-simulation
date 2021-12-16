"""Simulation environment and setup tools."""

import logging
import os
import random as rnd
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (Callable, Dict, Generic, Hashable, List, Mapping, Optional, Tuple, Type, TypeVar, Union)

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fl_simulation.client.computation import Worker, WorkerEvalResult
from fl_simulation.logging.logger import get_logger
from fl_simulation.server.aggregation import Aggregator

T_wid = TypeVar("T_wid", bound=Hashable)
T_worker = TypeVar("T_worker", bound=Worker)
T_class = TypeVar("T_class", bound="Simulation")

logger = get_logger()


@dataclass
class EvaluationResults(Generic[T_wid], Dict[T_wid, WorkerEvalResult]):
    """Contains evaluation results for one round."""


@dataclass
class RunResults(Generic[T_wid]):
    """Contains simulation run results."""

    distributed_evaluation_results: List[EvaluationResults] = field(default_factory=list)
    """Evaluation results for each of the rounds."""

    centralized_evaluation_results: List[WorkerEvalResult] = field(default_factory=lambda: [])
    """Evaluation results for each of the rounds."""


class Simulation(Generic[T_wid, T_worker, T_class]):
    """A Federated Learning simulation."""

    workers: Dict[T_wid, T_worker]
    """Workers to perform federated optimization."""
    aggr: Aggregator
    """Aggregator to optimize the shared model."""
    _checkpoint_destination: str = "checkpoints"
    """Directory to save checkpoints to."""
    _tensorboard_writer: Optional[SummaryWriter] = None
    """TensorBoard logger."""

    def __init__(
            self,
            aggr: Aggregator,
            workers: Dict[T_wid, T_worker],
            checkpoints_dir: str = "checkpoints",
            tensorboard_writer: Optional[SummaryWriter] = None
    ) -> None:
        """Create a new Federated Learning simulation.

        Args:
            aggr (Aggregator): aggregator to be used.
            workers (Dict[Hashable, Worker]): workers to perform local training.
            checkpoints_dir (str, optional): path to directory to save checkpoints to. Defaults to "checkpoints" subdirectory in current working directory.
            tensorboard_writer (SummaryWriter, optional): a logger for TensorBoard.
            incentive_mechanism (IncentiveMechanism, optional): a tracker for rewards and penalties.
        """
        self.workers = workers
        self.aggr = aggr
        self.set_checkpoint_destination(checkpoints_dir)
        self._tensorboard_writer = tensorboard_writer

    def run_cycle(self, data: Mapping[T_wid, DataLoader]) -> None:
        """Run a single cycle of model training.

        Args:
            data (Iterable[Tuple[T_wid, DataLoader]]): iterator over worker ids and local dataloaders. The data loader will be used in
                training on the corresponding worker.
        """
        logger.info("Starting a cycle.")
        updates = []

        for w_id, dl in data.items():
            srv_update = self.aggr.get_server_update(w_id)
            update = self.workers[w_id].do_cycle(srv_update, dl)
            update.worker_id = w_id
            updates.append(update)

        self.aggr.aggregate(updates)

    def run_eval(
            self, data: Mapping[T_wid, Optional[DataLoader]], model: Optional[nn.Module] = None
    ) -> EvaluationResults:
        """Run a single round of evaluation.

        Args:
            data (Mapping[T_wid, Optional[DataLoader]]): data for evaluation. `Wid`s which are mapped to `None` are ignored.
            model (Optional[nn.Module], optional): model to evalate. If `None` supplied, users will use their local models. Defaults to None.

        Returns:
            EvaluationResults: evaluation results for non-ignored users.
        """
        logger.info("Running evaluation.")
        results = EvaluationResults()

        for wid, dl in data.items():
            if dl:
                res = self.workers[wid].do_eval_cycle(data=dl, model=model)

                results[wid] = res

        return results

    def fit(
            self,
            n_rounds: int,
            n_users_per_round: int = 0,
            train_data: Optional[Mapping[T_wid, DataLoader]] = None,
            val_data: Optional[Union[Mapping[T_wid, DataLoader], DataLoader, Tuple[Mapping[T_wid, DataLoader],
                                                                                   DataLoader]]] = None,
            evaluator: Optional[Callable[[DataLoader, nn.Module], Dict[str, float]]] = None,
            checkpoint_freq: int = 0,
    ) -> RunResults[T_wid]:
        """Run the simulation.

        Args:
            n_rounds (int): number of rounds to be performed.
            n_users_per_round (int, optional): number of users per round. If non-positive, then all users are used. Defaults to 0.
            train_data (Optional[Mapping[T_wid, DataLoader]], optional): data to train on. If `None` is supplied, no training is performed. Defaults to None.
            val_data (Optional[Union[Mapping[T_wid, DataLoader], DataLoader, 
                Tuple[Mapping[T_wid, DataLoader], DataLoader]]], optional): data to evaluate on. If 
                `Mapping[T_wid, DataLoader]` is provided, runs evaluation of each worker with data available. If only
                `Dataloader` is provided, runs evaluation on shared model. If both are provided, runs both. If `None` is
                supplied, no evaluation is performed. Defaults to None.
            evaluator (Optional[Callable[[DataLoader, nn.Module], Dict[str, float]]], optional): function which ran if
                centralized evaluation is required. If `None`, then `do_eval_cycle` of one of the workers will be used.
            checkpoint_freq (int, optional): checkpoints are saved every checkpoint_freq rounds. If non-positive, checkpoints are not saved. Defaults to 0.

        Raises:
            UserWarning: If both training and validation data are `None`.

        Returns:
            RunResults[T_wid]: results of the simulation. 
        """
        if not train_data and not val_data:
            raise UserWarning("neither training nor validation data was supplied; skipping")

        distrib_val_data = None
        central_val_data = None

        if val_data is not None:
            if isinstance(val_data, Mapping):
                distrib_val_data = val_data
            elif isinstance(val_data, DataLoader):
                central_val_data = val_data
            else:
                distrib_val_data, central_val_data = val_data

        # set evaluator to the provided one or to the do_eval_step function of one of the workers
        evaluator = evaluator or next(iter(self.workers.values())).do_eval_cycle

        participating_users = None
        if train_data is not None:
            participating_users = list(train_data.keys())
        elif distrib_val_data is not None:
            participating_users = list(distrib_val_data.keys())

        n_users_per_round = min(n_users_per_round if n_users_per_round > 0 else len(self.workers), len(self.workers))
        run_results = RunResults()

        logger.info("Starting simulation with %d users per round and %d rounds.", n_users_per_round, n_rounds)
        for i in range(n_rounds):
            logger.info("Starting round %d.", i)

            selected_cohort = rnd.sample(participating_users, k=n_users_per_round) if participating_users else []

            if train_data:
                selected_cohort_train_data = {wid: train_data[wid] for wid in selected_cohort if wid in train_data}
                self.run_cycle(selected_cohort_train_data)

            # save checkpoint
            if checkpoint_freq > 0 and i % checkpoint_freq == 0:
                self.save_checkpoint()

            if distrib_val_data:
                model = self.get_shared_model()
                model.eval()

                selected_cohort_val_data = {wid: distrib_val_data.get(wid, None) for wid in selected_cohort}
                round_eval_results = self.run_eval(selected_cohort_val_data, model=model)
                run_results.distributed_evaluation_results.append(round_eval_results)

                if logger.isEnabledFor(logging.DEBUG) or self._tensorboard_writer:
                    total_metrics = defaultdict(float)
                    for r in round_eval_results.values():
                        for metric, val in r.items():
                            total_metrics[metric] += val

                    # `or 1` so there would not be zero division
                    total = len(round_eval_results) or 1

                    for metric, val in total_metrics.items():
                        avg_metric_val = val / total
                        logger.debug("\tAvg. %s across workers: %f", metric, avg_metric_val)
                        if self._tensorboard_writer:
                            self._tensorboard_writer.add_scalar(
                                    f'test/all_workers/local/avg_{metric}', avg_metric_val, i
                            )

            if central_val_data:
                shared_model = self.get_shared_model()
                evaluation_results = evaluator(central_val_data, shared_model)
                run_results.centralized_evaluation_results.append(WorkerEvalResult(evaluation_results))

                if logger.isEnabledFor(logging.DEBUG) or self._tensorboard_writer:
                    for metric, val in evaluation_results.items():
                        logger.debug("\t%s for shared model is %f", metric, val)
                        if self._tensorboard_writer:
                            self._tensorboard_writer.add_scalar(f'test/all_workers/shared/{metric}', val, i)

        return run_results

    def get_shared_model(self) -> nn.Module:
        """Get the trained shared model.

        Returns:
            torch.nn.Module: the shared model
        """
        return self.aggr.model

    def get_local_model(self, worker_id: T_wid) -> nn.Module:
        """Get the local model of the specified worker.

        Args:
            worker_id (Hashable): id of the worker, whose model needs to be returned.

        Returns:
            torch.nn.Module: the local model of the worker with the specified id.
        """
        return self.workers[worker_id].local_model

    def set_worker_device(self, w_id: T_wid, device: torch.device) -> None:
        """Set the torch device for the specified worker.

        Args:
            w_id (Hashable): worker id.
            device (torch.device): device to use.
        """
        self.workers[w_id].device = device

    def set_aggregator_device(self, device: torch.device) -> None:
        """Set the aggregator device.

        Args:
            device (torch.device): device to use.
        """
        self.aggr.device = device

    def set_checkpoint_destination(self, d: str) -> str:
        """Set a new directory to save checkpoints to.

        Args:
            d (str): path to the directory.

        Returns:
            str: previous path.
        """
        prev_dir = self._checkpoint_destination
        self._checkpoint_destination = d

        return prev_dir

    def save_checkpoint(self, **extra) -> None:
        """Save the checkpoint, including all the worker states and the aggregator state.

        Args:
            extra (Dict, optional): additional keyword arguments to be saved.
        """
        ids_to_names = {}

        for wid, w in self.workers.items():
            ids_to_names[wid] = w.name

            # save the worker's checkpoint
            d = os.path.join(self._checkpoint_destination, w.name)
            w.save_checkpoint(to=d)
        # save aggregator state
        d = os.path.join(self._checkpoint_destination, "aggregator")
        self.aggr.save_checkpoint(to=d)
        # save mapping from worker ids to their names
        torch.save(ids_to_names, os.path.join(self._checkpoint_destination, "ids_to_names.pickle"))
        # save extra parameters
        torch.save(extra, os.path.join(self._checkpoint_destination, "params.pickle"))

    @classmethod
    def load_checkpoint(
            cls: Type[T_class], checkpoint_dir: str, workers_factory: Callable[[str], T_worker],
            aggregator_factory: Callable[[], Aggregator]
    ) -> T_class:
        """Load the simulation for the saved checkpoint.

        Args:
            checkpoint_dir (str): path to saved checkpoint.
            workers_factory (Callable[[str], Worker]): callable to create workers. When the worker is created, it's name is passed into this function. It's state is going to be loaded from the checkpoint.
            aggregator_factory (Callable[..., Aggregator]): callable to create the aggregator. It's state is going to be loaded from the checkpoint.

        Returns:
            T_class: the simulation object.
        """
        self = cls.__new__(cls)
        self._checkpoint_destination = checkpoint_dir
        self.workers = {}

        ids_to_names = torch.load(os.path.join(self._checkpoint_destination, "ids_to_names.pickle"))
        # make sure it is a dictionary
        ids_to_names = dict(ids_to_names)

        # load workers
        for wid, name in ids_to_names.items():
            worker = workers_factory(name)

            worker.load_from_checkpoint(os.path.join(self._checkpoint_destination, name))
            self.workers[wid] = worker

        # load aggregator
        aggr = aggregator_factory()
        aggr.load_from_checkpoint(os.path.join(self._checkpoint_destination, "aggregator"))
        self.aggr = aggr

        # load extra parameters
        extra = torch.load(os.path.join(self._checkpoint_destination, "params.pickle"))
        extra = dict(extra)

        for attr, val in extra.items():
            setattr(self, attr, val)

        return self

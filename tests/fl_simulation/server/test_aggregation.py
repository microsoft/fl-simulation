import pytest

import copy
import os
import torch
import torch.nn as nn

from fl_simulation.client.update import ControlVarDiff, FedAvgModelUpdate, ModelDiff, ScaffoldModelUpdate
from fl_simulation.server.aggregation import AggregatorWithDropouts, FedAvgAggregator, ScaffoldAggregator

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model():
    model = nn.Linear(3, 1)
    model.weight = torch.nn.parameter.Parameter(torch.full((1, 3), 0.0))
    model.bias = torch.nn.parameter.Parameter(torch.full((1, ), 0.0))

    return model


@pytest.fixture
def fedavg_model_updates():
    model_updates = [
            FedAvgModelUpdate(
                    values=ModelDiff([torch.tensor([[6.0, 12.0, 18.0]]),
                                      torch.tensor([24.0])]),
                    num_examples=1,
            ),
            FedAvgModelUpdate(
                    values=ModelDiff([torch.tensor([[3.0, 6.0, 9.0]]),
                                      torch.tensor([12.0])]),
                    num_examples=2,
            ),
            FedAvgModelUpdate(
                    values=ModelDiff([torch.tensor([[2.0, 4.0, 6.0]]),
                                      torch.tensor([8.0])]),
                    num_examples=3,
            ),
    ]

    return model_updates


@pytest.fixture
def scaffold_model_updates():
    model_updates = [
            ScaffoldModelUpdate(
                    values=ModelDiff([torch.tensor([[6.0, 12.0, 18.0]]),
                                      torch.tensor([24.0])]),
                    ctl_var_update=ControlVarDiff([torch.tensor([1.0, 1.0, 1.0]),
                                                   torch.tensor([1.0])]),
                    num_examples=1,
            ),
            ScaffoldModelUpdate(
                    values=ModelDiff([torch.tensor([[3.0, 6.0, 9.0]]),
                                      torch.tensor([12.0])]),
                    ctl_var_update=ControlVarDiff([torch.tensor([1.0, 1.0, 1.0]),
                                                   torch.tensor([1.0])]),
                    num_examples=2,
            ),
            ScaffoldModelUpdate(
                    values=ModelDiff([torch.tensor([[2.0, 4.0, 6.0]]),
                                      torch.tensor([8.0])]),
                    ctl_var_update=ControlVarDiff([torch.tensor([1.0, 1.0, 1.0]),
                                                   torch.tensor([1.0])]),
                    num_examples=3,
            ),
    ]

    return model_updates


def test_fedavg(model, fedavg_model_updates):
    # FedAvg uses lr=1.0
    aggr = FedAvgAggregator(model)

    aggr.aggregate(fedavg_model_updates)

    # torch.tensor call is redandunt, but it makes mypy happy
    weight = torch.tensor(aggr.model.weight.data)
    bias = torch.tensor(aggr.model.bias.data)

    weight_expected = torch.tensor([[-3.0, -6.0, -9.0]])
    bias_expected = torch.tensor([-12.0])

    assert torch.equal(weight,
                       weight_expected) and torch.equal(bias, bias_expected), "aggregation performed incorrectly"


def test_aggregator_load_checkpoint(model, scaffold_model_updates, tmp_path):
    aggr = ScaffoldAggregator(initial_model=copy.deepcopy(model), total_clients=3, device=device)

    # update the global model
    aggr.aggregate(scaffold_model_updates)

    reference_weight = torch.tensor(aggr.model.weight.data).clone().detach()
    reference_bias = torch.tensor(aggr.model.bias.data).clone().detach()
    reference_ctl_var = [t.clone().detach() for t in aggr.ctl_var]
    reference_total_clients = aggr.total_clients

    # save the checkpoint
    d = os.path.join(str(tmp_path), "aggregator")
    aggr.save_checkpoint(to=d)

    # delete that aggregator
    del aggr

    # and create a new one
    aggr = ScaffoldAggregator(initial_model=copy.deepcopy(model), total_clients=3, device=device)
    aggr.load_from_checkpoint(d=d)

    assert reference_weight.equal(torch.tensor(aggr.model.weight.data)), "global model's weight did not load correctly"
    assert reference_bias.equal(torch.tensor(aggr.model.bias.data)), "global model's bias did not load correctly"
    assert all(
            c_old.equal(c_loaded) for c_old, c_loaded in zip(reference_ctl_var, aggr.ctl_var)
    ), "control variate did not load correctly"
    assert reference_total_clients == aggr.total_clients, "the total number of clients did not load correctly"

    # update the global model once more
    aggr.aggregate(scaffold_model_updates)

    # and check whether the model got updated properly
    assert not reference_weight.equal(
            torch.tensor(aggr.model.weight.data)
    ), "global model's weight did not get updated"
    assert not reference_bias.equal(torch.tensor(aggr.model.bias.data)), "global model's bias did not get updated"
    assert any(
            not c_old.equal(c_loaded) for c_old, c_loaded in zip(reference_ctl_var, aggr.ctl_var)
    ), "control variate did not get updated"


def test_dropout_aggregator(model, fedavg_model_updates):
    # FedAvg uses lr=1.0
    aggr = AggregatorWithDropouts(FedAvgAggregator(model), 0.0)

    aggr.aggregate(fedavg_model_updates)

    # torch.tensor call is redandunt, but it makes mypy happy
    weight = torch.tensor(aggr.model.weight.data)
    bias = torch.tensor(aggr.model.bias.data)

    weight_expected = torch.tensor([[-3.0, -6.0, -9.0]])
    bias_expected = torch.tensor([-12.0])

    assert torch.equal(weight,
                       weight_expected) and torch.equal(bias, bias_expected), "aggregation performed incorrectly"

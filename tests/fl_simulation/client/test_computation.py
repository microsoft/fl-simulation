import copy
from typing import Callable, Any, Dict
import pytest

import torch
import torch.nn as nn
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from fl_simulation.client.computation import FedProxWorker, ScaffoldWorker, ScaffoldServerUpdate
from fl_simulation.server.update import ServerUpdate

from tests.fl_simulation.conftest import MnistFedAvgWorker

lr = 0.01
epoch_num = 3
batch_size_train = 64
batch_size_test = 64
num_cycles = 2
device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.mark.slow
def test_worker(mnist_hadwritten_data, mnist_model, capsys):
    train_data, test_data = mnist_hadwritten_data

    train_dl = DataLoader(train_data, batch_size=batch_size_train)
    test_dl = DataLoader(test_data, batch_size=batch_size_test)

    worker = MnistFedAvgWorker(mnist_model, num_epochs=1, loss_fn=nn.CrossEntropyLoss(), lr=0.01, device=device)

    srv_update = ServerUpdate(shared_model=mnist_model)
    accuracy = 0.0

    with torch.autograd.detect_anomaly():
        for _ in range(num_cycles):
            worker.do_cycle(srv_update, train_dl)
            srv_update = ServerUpdate(copy.deepcopy(worker.local_model))

            model = worker.local_model

            model.to(device)

            model.eval()

            test_loss = 0
            correct = 0
            total = 0

            loss_f = nn.CrossEntropyLoss(reduction="sum")

            with torch.no_grad():
                for data, target in test_dl:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += loss_f(output, target).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
                    total += data.shape[0]
            test_loss /= total
            accuracy = correct / total

            model.cpu()

            with capsys.disabled():
                print(f"Avg. loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}\n")

    assert accuracy > 0.5


@pytest.mark.slow
def test_worker_on_frozen_model(mnist_hadwritten_data, mnist_model, capsys):
    train_data, test_data = mnist_hadwritten_data

    train_dl = DataLoader(train_data, batch_size=batch_size_train)
    test_dl = DataLoader(test_data, batch_size=batch_size_test)

    initial_param_vals = []

    # freeze the first fully connected layer
    for p in mnist_model.fc1.parameters():
        p.requires_grad = False
        initial_param_vals.append(p.data.detach().clone().cpu())

    worker = MnistFedAvgWorker(mnist_model, num_epochs=1, loss_fn=nn.CrossEntropyLoss(), lr=0.01, device=device)

    srv_update = ServerUpdate(shared_model=mnist_model)
    accuracy = 0.0

    with torch.autograd.detect_anomaly():
        for _ in range(num_cycles):
            worker.do_cycle(srv_update, train_dl)
            srv_update = ServerUpdate(copy.deepcopy(worker.local_model))

            model = worker.local_model

            model.to(device)

            model.eval()

            test_loss = 0
            correct = 0
            total = 0

            loss_f = nn.CrossEntropyLoss(reduction="sum")

            with torch.no_grad():
                for data, target in test_dl:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += loss_f(output, target).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
                    total += data.shape[0]
            test_loss /= total
            accuracy = correct / total

            model.cpu()

            with capsys.disabled():
                print(f"Avg. loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}\n")

    final_param_vals = (p.data.detach().cpu() for p in worker.local_model.fc1.parameters())

    assert all((d1.equal(d2) for d1, d2 in zip(initial_param_vals, final_param_vals)))


@pytest.mark.slow
def test_worker_does_not_update_base_model(mnist_hadwritten_data, mnist_model, capsys):
    train_data, test_data = mnist_hadwritten_data

    train_dl = DataLoader(train_data, batch_size=batch_size_train)
    test_dl = DataLoader(test_data, batch_size=batch_size_test)

    worker = MnistFedAvgWorker(mnist_model, num_epochs=1, loss_fn=nn.CrossEntropyLoss(), lr=0.01, device=device)

    base_model = copy.deepcopy(mnist_model)
    srv_update = ServerUpdate(shared_model=mnist_model)
    accuracy = 0.0

    with torch.autograd.detect_anomaly():
        for _ in range(num_cycles):
            model_update = worker.do_cycle(srv_update, train_dl)

            assert any(upd.count_nonzero() for upd in model_update.values), "the model update is zero"

            assert all(
                    p_w.data.equal(p_base.data)
                    for p_w, p_base in zip(worker.base_model.parameters(), base_model.parameters())
            ), "the base model has changed"

            base_model = copy.deepcopy(worker.local_model)
            srv_update = ServerUpdate(base_model)

            model = worker.local_model

            model.to(device)

            model.eval()

            test_loss = 0
            correct = 0
            total = 0

            loss_f = nn.CrossEntropyLoss(reduction="sum")

            with torch.no_grad():
                for data, target in test_dl:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += loss_f(output, target).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
                    total += data.shape[0]
            test_loss /= total
            accuracy = correct / total

            model.cpu()

            with capsys.disabled():
                print(f"Avg. loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}\n")

    assert accuracy > 0.5


def test_fedprox_worker_updates_loss():

    class MnistFedProxWorker(FedProxWorker):

        def __init__(
                self,
                base_model: nn.Module,
                loss_fn: Callable[..., torch.Tensor],
                mu: float,
                num_epochs: int,
                device: torch.device = torch.device("cpu"),
                *opt_arg,
                **opt_kwarg,
        ) -> None:
            super(MnistFedProxWorker, self).__init__(base_model, loss_fn, mu=mu, device=device, *opt_arg, **opt_kwarg)
            self.num_epochs = num_epochs

        def continue_training(self) -> bool:
            return self.epochs_done < self.num_epochs

    def loss_f1(_1, _2):
        return torch.tensor([1.0])

    def loss_f2(_1, _2):
        return torch.tensor([2.0])

    model = nn.Linear(1, 1, bias=False)
    model.weight = torch.nn.parameter.Parameter(torch.tensor([[0.0]]))

    updated_model = nn.Linear(1, 1, bias=False)
    updated_model.weight = torch.nn.parameter.Parameter(torch.tensor([[1.0]]))

    worker = MnistFedProxWorker(base_model=model, loss_fn=loss_f1, mu=2.0, num_epochs=1, lr=0.01)
    # DO NOT DO THIS IN PRODUCTION. Worker's optimizer would still hold the parameters of the previous `local_model`.
    worker.local_model = updated_model

    # loss_f1(_, _) + 2.0/2 * norm([[1.0]] - [[0.0]])^2 == 1.0 + 1.0 == 2.0
    #    ^            ^               ^         ^
    #   == 1.0        mu         model.weight  updated_model.weight
    res = worker.loss_fn(0, 0)
    assert res == 2.0, "before any changes, computed loss is incorrect"

    # Changes the loss function without the proximal term
    worker.loss_fn = loss_f2

    # loss_f2(_, _) + 2.0/2 * norm([[1.0]] - [[0.0]])^2 == 2.0 + 1.0 == 3.0
    #    ^            ^               ^         ^
    #   == 1.0        mu         model.weight  updated_model.weight
    res = worker.loss_fn(0, 0)
    assert (res == 3.0), "after changing the underlying loss function, computed loss is incorrect"

    # Changes the loss function without the proximal term
    worker.mu = 4.0

    # loss_f2(_, _) + 4.0/2 * norm([[1.0]] - [[0.0]])^2 == 2.0 + 2.0 == 4.0
    #    ^            ^             ^         ^
    #   == 1.0        mu      model.weight  updated_model.weight
    res = worker.loss_fn(0, 0)
    assert (res == 4.0), "after changing \u03bc, computed loss is incorrect"


def test_worker_reload_from_checkpoint(mnist_hadwritten_data, mnist_model, tmp_path, capsys):

    class MnistScaffoldWorker(ScaffoldWorker):

        def get_opt(self) -> Optimizer:
            return SGD(self.local_model.parameters(), lr=0.01)

        def do_eval_step(self, batch: Any, model: nn.Module) -> Dict[str, float]:
            data, target = batch
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = nn.functional.cross_entropy(output, target, reduction="sum").item()
            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).sum()
            total = data.shape[0]

            return {"loss": test_loss, "accuracy": correct / total}

    train_data, test_data = mnist_hadwritten_data

    train_dl = DataLoader(Subset(train_data, range(1000)), batch_size=batch_size_train)
    test_dl = DataLoader(Subset(test_data, range(1000)), batch_size=batch_size_test)

    initial_model = copy.deepcopy(mnist_model)

    worker = MnistScaffoldWorker(initial_model, num_epochs=1, loss_fn=nn.CrossEntropyLoss(), device=device)

    srv_update = ScaffoldServerUpdate(
            shared_model=initial_model, ctl_var=[torch.zeros_like(p.data) for p in initial_model.parameters()]
    )

    with torch.autograd.detect_anomaly():
        # Do one optimization cycle
        worker.do_cycle(srv_update, train_dl)

        res = worker.do_eval_cycle(test_dl)

        with capsys.disabled():
            print(f"""Avg. loss: {res["loss"]:.4f}, Accuracy: {res["accuracy"]:.2f}\n""")

        reference_model = copy.deepcopy(worker.local_model)
        reference_ctl_var = copy.deepcopy(worker.ctl_var)

        srv_update = ScaffoldServerUpdate(
                shared_model=copy.deepcopy(worker.local_model), ctl_var=copy.deepcopy(worker.ctl_var)
        )

        # save checkpoint
        worker.save_checkpoint(to=str(tmp_path))

        # and remove the worker
        del worker

        # create a new worker. notice, that the model we use in init is untouched
        worker = MnistScaffoldWorker(
                copy.deepcopy(mnist_model), num_epochs=1, loss_fn=nn.CrossEntropyLoss(), device=device
        )

        # load from the checkpoint
        worker.load_from_checkpoint(str(tmp_path))

        # make sure we loaded the model and got the updated weights
        assert all(
                p_old.data.equal(p_loaded.data)
                for p_old, p_loaded in zip(reference_model.parameters(), worker.local_model.parameters())
        ), "the loaded model does not match the reference one"

        # make sure we loaded the control variate correctly
        assert all(
                c_old.equal(c_loaded) for c_old, c_loaded in zip(reference_ctl_var, worker.ctl_var)
        ), "loaded control variate does not match the reference one"

        # run one more cycle to make sure everything works
        worker.do_cycle(srv_update, train_dl)

        res = worker.do_eval_cycle(test_dl)

        # make sure we the local model got updated
        assert any(
                not p_old.data.equal(p_loaded.data)
                for p_old, p_loaded in zip(reference_model.parameters(), worker.local_model.parameters())
        ), "the local model did not get updated after second cycle"

        # make sure the control variate got updated as well
        assert any(
                not c_old.equal(c_loaded) for c_old, c_loaded in zip(reference_ctl_var, worker.ctl_var)
        ), "loaded control variate did not get updated after the second cycle"

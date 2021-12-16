from typing import Callable, Dict, Tuple

import pytest
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Subset
from torchvision.datasets.mnist import MNIST

from fl_simulation.client.computation import FedProxWorker, ScaffoldWorker
from fl_simulation.server.aggregation import Aggregator, FedProxAggregator, ScaffoldAggregator
from fl_simulation.simulation.instance import Simulation
from tests.fl_simulation.conftest import MnistFedAvgAggregator, MnistFedAvgWorker

num_clients = 3
batch_size_train = 64
batch_size_test = 64
device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MockAggregator(Aggregator):
    """Mock of the Aggregator. Just unpacks the update from a single worker."""

    def get_opt(self):
        return SGD(self.model.parameters(), lr=1.0)

    def aggr_fn(self, updates):
        return updates[0].values


@pytest.fixture
def mnist_mock_sim(mnist_model, tensorboard_writer):
    aggr = MockAggregator(mnist_model)

    workers = {
            i: MnistFedAvgWorker(
                    mnist_model,
                    num_epochs=1,
                    loss_fn=nn.CrossEntropyLoss(),
                    device=device,
                    tensorboard_writer=tensorboard_writer,
                    lr=0.01,
            )
            for i in range(num_clients)
    }

    sim = Simulation(aggr, workers)

    return sim


@pytest.fixture
def mnist_fedavg_simulation(mnist_model):
    aggr = MnistFedAvgAggregator(mnist_model)

    workers = {
            i: MnistFedAvgWorker(
                    mnist_model,
                    num_epochs=1,
                    loss_fn=nn.CrossEntropyLoss(),
                    device=device,
                    lr=0.01,
            )
            for i in range(num_clients)
    }

    sim = Simulation(aggr, workers)

    return sim


@pytest.fixture
def mnist_fedprox_simulation(mnist_model: nn.Module) -> Simulation:

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

    aggr = FedProxAggregator(mnist_model)

    workers = {
            i: MnistFedProxWorker(
                    mnist_model,
                    loss_fn=nn.CrossEntropyLoss(),
                    mu=0.01,
                    num_epochs=1,
                    device=device,
                    lr=0.01,
            )
            for i in range(num_clients)
    }

    return Simulation(aggr, workers)


@pytest.fixture
def mnist_scaffold_simulation(mnist_model: nn.Module) -> Simulation:

    class MnistScaffoldWorker(ScaffoldWorker):

        def get_opt(self) -> Optimizer:
            return SGD(self.local_model.parameters(), lr=0.01)

    aggr = ScaffoldAggregator(mnist_model, total_clients=num_clients, device=device)

    workers = {
            i: MnistScaffoldWorker(mnist_model, loss_fn=nn.CrossEntropyLoss(), device=device, num_epochs=1)
            for i in range(num_clients)
    }

    return Simulation(aggr, workers)


@pytest.fixture
def mnist_hadwritten_data_federated(
        mnist_hadwritten_data: Tuple[MNIST, MNIST]
) -> Tuple[Dict[int, DataLoader], DataLoader]:
    train_ds, test_ds = mnist_hadwritten_data

    l = len(train_ds) // num_clients

    train_dls = {
            i: DataLoader(
                    Subset(
                            train_ds,
                            list(range(i * l, (i + 1) * l)),
                    ),
                    batch_size=batch_size_train,
            )
            for i in range(num_clients)
    }

    test_dl = DataLoader(test_ds, batch_size=batch_size_test)

    return train_dls, test_dl


@pytest.mark.slow
def test_mock(mnist_hadwritten_data_federated, mnist_mock_sim: Simulation, capsys, tensorboard_writer):
    num_cycles = 2

    train_dls, test_dl = mnist_hadwritten_data_federated

    sim = mnist_mock_sim

    accuracy = 0.0
    with capsys.disabled():
        with torch.autograd.detect_anomaly():
            for i in range(num_cycles):
                sim.run_cycle(train_dls)

                model = sim.get_shared_model()

                test_loss = 0
                correct = 0
                total = 0

                loss_f = nn.CrossEntropyLoss(reduction="sum")

                model.to(device)
                model.eval()

                with torch.no_grad():
                    for data, target in test_dl:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        test_loss += loss_f(output, target).item()
                        pred = output.data.max(1, keepdim=True)[1]
                        correct += pred.eq(target.data.view_as(pred)).sum()
                        total += pred.shape[0]

                test_loss /= total
                accuracy = correct / total

                model.cpu()

                print(f"\nCYCLE {i}: Avg. loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}\n")
                tensorboard_writer.add_scalar('test/loss', test_loss, i, new_style=True)
                tensorboard_writer.add_scalar('test/accuracy', accuracy, i, new_style=True)

    tensorboard_writer.flush()
    assert accuracy > 0.5


@pytest.mark.slow
def test_fedavg(mnist_hadwritten_data_federated, mnist_fedavg_simulation: Simulation, capsys, tensorboard_writer):
    num_cycles = 10

    train_dls, test_dl = mnist_hadwritten_data_federated

    sim = mnist_fedavg_simulation

    accuracy = 0.0
    with capsys.disabled():
        with torch.autograd.detect_anomaly():
            for i in range(num_cycles):
                sim.run_cycle(train_dls)

                model = sim.get_shared_model()

                test_loss = 0
                correct = 0
                total = 0

                loss_f = nn.CrossEntropyLoss(reduction="sum")

                model.to(device)
                model.eval()

                with torch.no_grad():
                    for data, target in test_dl:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        test_loss += loss_f(output, target).item()
                        pred = output.data.max(1, keepdim=True)[1]
                        correct += pred.eq(target.data.view_as(pred)).sum()
                        total += pred.shape[0]

                test_loss /= total
                accuracy = correct / total

                model.cpu()
                print(f"\nCYCLE {i}: Avg. loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}\n")
                tensorboard_writer.add_scalar('test/loss', test_loss, i, new_style=True)
                tensorboard_writer.add_scalar('test/accuracy', accuracy, i, new_style=True)

    assert accuracy > 0.5


@pytest.mark.slow
def test_fedprox(mnist_hadwritten_data_federated, mnist_fedprox_simulation: Simulation, capsys):
    num_cycles = 2

    train_dls, test_dl = mnist_hadwritten_data_federated

    sim = mnist_fedprox_simulation

    accuracy = 0.0
    with capsys.disabled():
        with torch.autograd.detect_anomaly():
            for i in range(num_cycles):
                sim.run_cycle(train_dls)

                model = sim.get_shared_model()

                test_loss = 0
                correct = 0
                total = 0

                loss_f = nn.CrossEntropyLoss(reduction="sum")

                model.to(device)
                model.eval()

                with torch.no_grad():
                    for data, target in test_dl:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        test_loss += loss_f(output, target).item()
                        pred = output.data.max(1, keepdim=True)[1]
                        correct += pred.eq(target.data.view_as(pred)).sum()
                        total += pred.shape[0]

                test_loss /= total
                accuracy = correct / total

                model.cpu()
                print(f"\nCYCLE {i}: Avg. loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}\n")

    assert accuracy > 0.5


@pytest.mark.slow
def test_scaffold(mnist_hadwritten_data_federated, mnist_scaffold_simulation: Simulation, capsys):
    num_cycles = 1

    train_dls, test_dl = mnist_hadwritten_data_federated

    sim = mnist_scaffold_simulation

    accuracy = 0.0
    with capsys.disabled():
        with torch.autograd.detect_anomaly():
            for i in range(num_cycles):
                sim.run_cycle(train_dls)

                model = sim.get_shared_model()

                test_loss = 0
                correct = 0
                total = 0

                loss_f = nn.CrossEntropyLoss(reduction="sum")

                model.to(device)
                model.eval()

                with torch.no_grad():
                    for data, target in test_dl:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        test_loss += loss_f(output, target).item()
                        pred = output.data.max(1, keepdim=True)[1]
                        correct += pred.eq(target.data.view_as(pred)).sum()
                        total += pred.shape[0]

                test_loss /= total
                accuracy = correct / total

                model.cpu()
                print(f"\nCYCLE {i}: Avg. loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}\n")

    assert accuracy > 0.5


@pytest.mark.slow
def test_scaffold_with_grad_ctl_var_updates(
        mnist_hadwritten_data_federated,
        mnist_scaffold_simulation: Simulation,
        capsys,
):
    num_cycles = 1

    train_dls, test_dl = mnist_hadwritten_data_federated

    sim = mnist_scaffold_simulation
    for worker in sim.workers.values():
        worker._approx_ctl_var_update = False

    accuracy = 0.0
    with capsys.disabled():
        with torch.autograd.detect_anomaly():
            for i in range(num_cycles):
                sim.run_cycle(train_dls)

                model = sim.get_shared_model()

                test_loss = 0
                correct = 0
                total = 0

                loss_f = nn.CrossEntropyLoss(reduction="sum")

                model.to(device)
                model.eval()

                with torch.no_grad():
                    for data, target in test_dl:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        test_loss += loss_f(output, target).item()
                        pred = output.data.max(1, keepdim=True)[1]
                        correct += pred.eq(target.data.view_as(pred)).sum()
                        total += pred.shape[0]

                test_loss /= total
                accuracy = correct / total

                model.cpu()
                print(f"\nCYCLE {i}: Avg. loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}\n")

    assert accuracy > 0.5

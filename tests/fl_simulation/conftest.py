import pytest

import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Subset
import torchvision
from torch.utils.tensorboard import SummaryWriter

from fl_simulation.client.computation import FedAvgWorker
from fl_simulation.server.aggregation import FedAvgAggregator


class MnistNet(nn.Module):

    def __init__(self):
        super(MnistNet, self).__init__()
        hidden_1 = 512
        hidden_2 = 512
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc3(x)
        return x


@pytest.fixture
def mnist_hadwritten_data():
    path = "data"

    train_data = torchvision.datasets.MNIST(
            path,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                    [
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307, ), (0.3081, )),
                    ]
            ),
    )

    test_data = torchvision.datasets.MNIST(
            path,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                    [
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307, ), (0.3081, )),
                    ]
            ),
    )

    train_data = Subset(train_data, list(range(min(10000, len(train_data)))))
    test_data = Subset(test_data, list(range(min(10000, len(test_data)))))

    return train_data, test_data


@pytest.fixture
def mnist_model():
    return MnistNet()


@pytest.fixture
def tensorboard_writer():
    return SummaryWriter(os.path.join('tensorboard_runs', 'tests'))


class MnistFedAvgWorker(FedAvgWorker):
    pass


class MnistFedAvgAggregator(FedAvgAggregator):
    pass

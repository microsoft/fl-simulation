import unittest

import torch
import torch.nn.functional as F
from fl_simulation.utils.softmax import custom_softmax
from torch import tensor


class TestSoftmax(unittest.TestCase):
    def test_custom_softmax(self):
        x = tensor([[1., 2, 3]])
        assert custom_softmax(x, dim=1).tolist() == F.softmax(x, dim=1).tolist()
        x = torch.randn((2, 3))
        assert custom_softmax(x, dim=-1).tolist() == F.softmax(x, dim=-1).tolist()

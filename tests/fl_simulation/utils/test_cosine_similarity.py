import unittest
from math import sqrt

from torch import tensor

from fl_simulation.client.update import ModelDiff
from fl_simulation.utils.cosine_similarity import cosine_similarity_model_diff


class TestCosineSimilarity(unittest.TestCase):

    def test_cosine_similarity(self):
        u1 = ModelDiff([tensor([1.])])
        u2 = ModelDiff([tensor([1.])])
        assert cosine_similarity_model_diff(u1, u2) == 1.0

        u1 = ModelDiff([tensor([1., 1]), tensor([[1., -1], [-2, 3]])])
        u2 = ModelDiff([tensor([1., -1]), tensor([[-1., 1], [-2, 5]])])
        self.assertAlmostEqual(
                cosine_similarity_model_diff(u1, u2),
                (0 + -2 + (4 + 15)) / (sqrt(2 + 2 + (4 + 9)) * sqrt(2 + 2 + (4 + 25))), 5
        )

    def test_cosine_similarity_0(self):
        # Shouldn't happen and not really defined but it's good to test just in case.
        u1 = ModelDiff([tensor([0.])])
        u2 = ModelDiff([tensor([0.])])
        assert cosine_similarity_model_diff(u1, u2) == 0

        u1 = ModelDiff([tensor([0.])])
        u2 = ModelDiff([tensor([1.])])
        assert cosine_similarity_model_diff(u1, u2) == 0

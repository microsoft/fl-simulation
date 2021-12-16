import unittest
from math import sqrt

from torch import tensor

from fl_simulation.client.update import ModelUpdate
from fl_simulation.incentive import ComposableIm, ParticipationIm
from fl_simulation.incentive.similarity import SimilarityIm
from fl_simulation.server.update import AggregatedUpdate
from fl_simulation.utils.types import ModelDiff


class TestComposable(unittest.TestCase):

    def test_composable(self):
        im = ComposableIm([ParticipationIm(), SimilarityIm()])
        updates = [
                ModelUpdate(ModelDiff([tensor([1., 1])]), 2),
                ModelUpdate(ModelDiff([tensor([1., 3])]), 3),
        ]
        for i, update in enumerate(updates):
            update.worker_id = i
        aggregated_update = AggregatedUpdate(ModelDiff([tensor([1., 2])]))
        im.update(updates, aggregated_update)

        for update in updates:
            assert im.get_reward(update.worker_id, 'num_examples') == update.num_examples
            assert im.get_reward(update.worker_id, 'num_updates') == 1

        self.assertAlmostEqual((1 + 2) / (sqrt(2) * sqrt(5)), im.get_reward(0, 'similarity'))
        self.assertAlmostEqual((1 + 6) / (sqrt(10) * sqrt(5)), im.get_reward(1, 'similarity'))

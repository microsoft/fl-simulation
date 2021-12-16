from torch import tensor

from fl_simulation.client.update import ModelUpdate
from fl_simulation.incentive import ParticipationIm
from fl_simulation.server.update import AggregatedUpdate
from fl_simulation.utils.types import ModelDiff


def test_participation():
    im = ParticipationIm()
    updates = [
            ModelUpdate(ModelDiff([tensor([1, 2])]), 2),
            ModelUpdate(ModelDiff([tensor([1, 2])]), 3),
    ]
    for i, update in enumerate(updates):
        update.worker_id = i
    aggregated_update = AggregatedUpdate(ModelDiff([tensor([1, 2])]))
    im.update(updates, aggregated_update)

    for update in updates:
        assert im.get_reward(update.worker_id, 'num_examples') == update.num_examples
        assert im.get_reward(update.worker_id, 'num_updates') == 1

    expected = {0: {'num_examples': 2, 'num_updates': 1}, 1: {'num_examples': 3, 'num_updates': 1}}
    assert im.rewards == expected

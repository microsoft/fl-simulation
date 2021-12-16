from fl_simulation.client.update import ModelUpdate
from fl_simulation.server.aggregation import DistanceBasedModelAssigner
import pytest
import torch

from fl_simulation.server.update import AggregatedUpdate
from fl_simulation.utils.types import ModelDiff


@pytest.fixture
def model_assigner():
    return DistanceBasedModelAssigner()


@pytest.fixture
def representatives():
    return {
            1: AggregatedUpdate(ModelDiff([torch.tensor([0.0]), torch.tensor([1.0])])),
            2: AggregatedUpdate(ModelDiff([torch.tensor([1.0]), torch.tensor([0.0])])),
            3: AggregatedUpdate(ModelDiff([torch.tensor([-1.0]), torch.tensor([0.0])]))
    }


@pytest.fixture
def assignable_updates():
    """Return updates assignable to the representatives.

    Returns:
        List[ModelUpdate]: updates, which should be assigned to representatives 1, 2, and 3 respectively. 
    """
    return [
            ModelUpdate(ModelDiff([torch.tensor([0.0]), torch.tensor([1.0001])]), 1),
            ModelUpdate(ModelDiff([torch.tensor([1.0001]), torch.tensor([0.0])]), 1),
            ModelUpdate(ModelDiff([torch.tensor([-1.0001]), torch.tensor([0.0])]), 1)
    ]


@pytest.fixture
def cluster_updates():
    return [
            ModelUpdate(ModelDiff([torch.tensor([0.0]), torch.tensor([1.0])]), 1),
            ModelUpdate(ModelDiff([torch.tensor([0.0]), torch.tensor([1.001])]), 1),
            ModelUpdate(ModelDiff([torch.tensor([0.0]), torch.tensor([1.002])]), 1),
            ModelUpdate(ModelDiff([torch.tensor([1.0]), torch.tensor([0.0])]), 1),
            ModelUpdate(ModelDiff([torch.tensor([1.001]), torch.tensor([0.0])]), 1),
            ModelUpdate(ModelDiff([torch.tensor([1.002]), torch.tensor([0.0])]), 1),
    ]


def test_updates_assigned_to_models(model_assigner, representatives, assignable_updates):
    assignments = model_assigner(assignable_updates, representatives)
    # for each update in `assignable_updates` indicates to which representative it should be assigned
    expected_assignments = [1, 2, 3]

    for u, expected in zip(assignable_updates, expected_assignments):
        assert expected in assignments, f"update {u} has not been assigned to expected representative {expected}"
        assert any(
                all(t1.eq(t2) for t1, t2 in zip(u.values, upd.values)) for upd in assignments[expected]
        ), f"update {u} has not been assigned to expected representative {expected}"


def test_updates_clustered(model_assigner, cluster_updates):

    def clusters_equal(c1, c2):
        if len(c1) != len(c2):
            return False

        all_found = True

        for elem1 in c1:
            all_found = all_found and any(all(t1.eq(t2) for t1, t2 in zip(elem1.values, elem2.values)) for elem2 in c2)

        for elem1 in c2:
            all_found = all_found and any(all(t1.eq(t2) for t1, t2 in zip(elem1.values, elem2.values)) for elem2 in c1)

        return all_found

    assignments = model_assigner(cluster_updates, {})
    expected_cluster1 = cluster_updates[:3]
    expected_cluster2 = cluster_updates[3:]

    assert len(assignments) == 2, f"expected 2 clusters, got {len(assignments)}"

    _, cl1 = assignments.popitem()
    _, cl2 = assignments.popitem()

    assert clusters_equal(expected_cluster1, cl1) or clusters_equal(
            expected_cluster1, cl2
    ), f"clusters where not formed correctly: expected {expected_cluster1} to be preset, got {cl1}, {cl2}"

    assert clusters_equal(expected_cluster2, cl1) or clusters_equal(
            expected_cluster2, cl2
    ), f"clusters where not formed correctly: expected {expected_cluster2} to be preset, got {cl1}, {cl2}"

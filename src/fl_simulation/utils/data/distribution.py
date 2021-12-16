"""Utilities for data distribution across clients in the Federated Setting."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset
from typing import Callable, Dict, Mapping, Sized, cast


def cosine_similarity_criterion(t1: torch.Tensor, t2: torch.Tensor) -> float:
    """Calculate cosine similarity of two tensors (e.g., dataset encodings) and return the numerical representation of that similarity.

    Args:
        t1 (torch.Tensor): the first tensor. Must be of one-dimensional.
        t2 (torch.Tensor): the second tensor. Must be of one-dimensional.

    Returns:
        float: numerical representation of the similarity of two tensors. Lies in [0, 1].
    """
    return float(F.cosine_similarity(t1, t2, dim=0))


def merge_user_datasets(
        user_dss: Mapping[str, Dataset],
        ds_mapper: Callable[[Dataset], torch.Tensor],
        similarity_criterion: Callable[[torch.Tensor, torch.Tensor], float] = cosine_similarity_criterion,
        min_examples: int = -1,
) -> Dict[str, Dataset]:
    """Merge the users' datasets so that each has sufficient data.

    Args:
        user_dss (Mapping[str, Dataset & Sized]): user datasets. Mappes users' ids to the corresponding dataset.
        ds_mapper (Callable[[Dataset], torch.Tensor]): function mapping the user dataset into one-dimensional tensor (i.e. function calculating representation).
        similarity_criterion (Callable[[torch.Tensor, torch.Tensor], float], optional): function for finding the similarity of two datasets based on their tensor representation. Defaults to cosine_similarity_criterion.
        min_examples (int, optional): minimal number of examples each resulting dataset should have. Defaults to -1.

    Returns:
        Dict[str, Dataset]: if the merge was successfull, then returns the merged datasets. Otherwise, returns an empty dictionary.
    """
    # pool of users available for merging
    uid_available = set(user_dss.keys())
    dss_lengths = {uid: len(cast(Sized, ds)) for uid, ds in user_dss.items()}

    uid_insuf = [uid for uid, l in dss_lengths.items() if l < min_examples]

    dss_encodings = {uid: ds_mapper(ds) for uid, ds in user_dss.items()}
    new_dss = {}

    # For each user with insufficient data
    for uid in uid_insuf:
        # continue if this uid is no longer available for merging
        if uid not in uid_available:
            continue
        # otherwise mark it as unavailable
        uid_available.remove(uid)

        # find pairwise similarity with other users available for merging
        ds_enc = dss_encodings[uid]
        pairwise_sim = {
                uid_other: similarity_criterion(ds_enc, dss_encodings[uid_other])
                for uid_other in uid_available
        }

        # sort by other dataset lengths
        pairwise_sim_sorted_len = sorted(pairwise_sim.items(), key=lambda p: dss_lengths[p[0]])
        # sort by similarity
        pairwise_sim_sorted = sorted(pairwise_sim_sorted_len, key=lambda p: p[1])

        # choose the other users to merge with the given one
        users_to_merge = [uid]
        num_examples = dss_lengths[uid]
        i = 0

        # while we still need more examples
        while num_examples < min_examples:
            # if we exceeded the available users, return failure marker
            if i >= len(pairwise_sim_sorted):
                return {}
            # record the next most similar user with least data to be merged with the current one
            uid_other, _ = pairwise_sim_sorted[i]
            users_to_merge.append(uid_other)
            # make the merged user unavailable
            uid_available.remove(uid_other)
            # update the number of the examples in the merged dataset
            num_examples += dss_lengths[uid_other]
        # actually merge the datasets
        new_dss[uid] = ConcatDataset([user_dss[uid] for uid in users_to_merge])

    # add untouched users to the new datasets
    for uid in uid_available:
        new_dss[uid] = user_dss[uid]

    return new_dss

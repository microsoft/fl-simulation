"""Federated datasets."""

import enum
import os
import random as rnd
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import (Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple, Union)

import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer

from fl_simulation.utils.file import extract_zip
from fl_simulation.utils.net import download_url


class MINDSize(enum.Enum):
    """The MIND version to use."""

    small = "small"
    large = "large"


class MINDMode(enum.Enum):
    """Whether the dataset is for training, for validation, or for testing (competition)."""

    train = "train"
    val = "dev"
    test = "test"


class MIND(Dataset):
    """Microsoft News Dataset wrapper for PyTorch."""

    @dataclass(repr=True, eq=True)
    class _BehaviorEntry:
        """User behaviour entry based on an impression.

        Args:
            uid (str): user identifier.
            impr_id (int): impression identifier.
            cand_news_id (int): identifier of the candidate article.
            label (int): 0 for non-clicked candidates, 1 for clicked candidates.
        """

        uid: str
        impr_id: int
        cand_news_id: int
        label: int

    @dataclass(repr=True)
    class _HistoryEntry:
        """The history entry.
        
        Args:
            hist_enc (List[int]): a list of news ids clicked by the user. The most recent ones are on the right, padding is on the left.
            hist_mask (torch.Tensor): a mask indicating the padded positions (articles). Has the same size as `hist_enc` and is `True` in the positions, which are padding.  
        """

        hist_enc: List[int]
        hist_mask: torch.Tensor

    vocab: Dict[str, int]
    """Mapping from tokens to their ids."""
    nid2idx: Dict[str, int]
    """Mapping from the raw news ids to the corresponding index in the titles table."""
    behaviors: List[_BehaviorEntry]
    """User behaviors."""
    histories: Dict[int, _HistoryEntry]
    """User histories."""

    max_title_len: int
    """Maximum length of the news article title in tokens."""
    max_hist_len: int
    """Maximum length of the user's history in news articles."""

    titles_enc: torch.Tensor
    """News articles' titles encoded in correspondence with `vocab` ordered so that title for article N could be found at index `nid2idx[N]`."""

    def __init__(
            self,
            data_dir: Union[str, bytes, "os.PathLike[str]"] = "mind",
            mind_size: MINDSize = MINDSize.small,
            mind_mode: MINDMode = MINDMode.train,
            npratio: int = -1,
            max_title_len: int = 20,
            max_hist_len: int = 100,
            max_word_freq: float = 0.8,
            min_word_occur: int = 5,
            max_vocab_len: Optional[int] = None,
            pad_tok: str = "<PAD>",
            vocab: Optional[Dict[str, int]] = None,
            impr_with_labels: bool = True,
            base_url: str = "https://mind201910small.blob.core.windows.net/release",
            download_verbose: bool = False,
            force_download: bool = False,
            download_progress_updater: Optional[Callable[[int, int, int], None]] = None
    ) -> None:
        """Load MIND from the specified files into the PyTorch Dataset.

        Args:
            data_dir (Union[str, bytes, "PathLike[str]"], optional): directory with the MIND dataset. If there is no data at that location, the data will be downloaded there. Defaults to "mind".
            mind_size (MindSize, optional): which size of the dataset to use.
            mind_mode (MindMode, optional): the mode for the dataset (train, validation, test).
            npratio (int, optional): negative and positive ratio for negative sampling. For each positive sample, `npratio` negative samples will be saved. -1 means no negative sampling. Defaults to -1.
            max_title_len (int, optional): maximum title length in tokens. Deafaults to 20, which should be sufficient in most cases, as seen on fig. 2(a) in Wu et al. "MIND: A Large-scale Dataset for News Recommendation" 
            max_hist_len (int, optional): maximum history length in entries. Defaults to 100, which would be sufficient for almost all points in MIND.
            max_word_freq (float, optional): maximum word frequency for the word to be included in the vocabulary. Defaults to 0.8.
            min_word_occur (int, optional): minimal word occurrence for it to be included in the vocabulary. Defaults to 5.
            max_vocab_len (Optional[int], optional): maximum number of tokens in the vocabulary. If set to None (or non-positive int), no limit is set. Defaults to None.
            pad_tok (str, optional): token used for padding. Defaults to "<PAD>".
            vocab (Optional[Dict[str, int]], optional): predefined vocabulary. If None provided, will be build based on the news articles' titles. Out of vocabulary token should be placed at index 0, or be called <OOV>. Defaults to None.
            impr_with_labels (bool, optional): whether there are labels available for the impressions. Should be set to false for the test dataset. Defaults to `True`.
            base_url (str, optional): base usr for the MIND.
            download_verbose (bool, optional): whether to print download reporting messages to the console. Defaults to False. 
            force_download (bool, optional): whether to replace the destination dataset files if they already exist. Defaults to False.
            download_progress_updater (Optional[Callable[[int, int, int], None]], optional): download progress report callback. Defaults to None.

        Raises:
            ValueError: if the small the `mind_size` and the test `mind_mode` are combined -- there is no test set for the small version. 
        """
        super().__init__()

        if mind_mode == MINDMode.test and mind_size == MINDSize.small:
            raise ValueError("there is no test set for the small version of MIND")

        self.nid2idx = {}

        self.behaviors = []
        self.histories = {}

        self.max_title_len = max_title_len
        self.max_hist_len = max_hist_len

        self.tokenizer = get_tokenizer("basic_english")

        os.makedirs(data_dir, exist_ok=True)

        arch_name = f"MIND{mind_size.value}_{mind_mode.value}.zip"
        url = f"{base_url}/{arch_name}"
        destination_filename = os.path.join(str(data_dir), arch_name)

        download_url(
                url,
                destination_filename,
                verbose=download_verbose,
                force_download=force_download,
                progress_updater=download_progress_updater,
        )
        extract_zip(destination_filename, str(data_dir))

        news_file = os.path.join(str(data_dir), "news.tsv")
        behaviors_file = os.path.join(str(data_dir), "behaviors.tsv")

        self._load_news(news_file, max_word_freq, min_word_occur, pad_tok, vocab, max_vocab_len)
        self._load_behaviors(behaviors_file, impr_with_labels, npratio)

    @staticmethod
    def _make_raw(
            vocab: Dict[str, int], nid2idx: Dict[str, int], behaviors: List[_BehaviorEntry],
            histories: Dict[int, _HistoryEntry], max_title_len: int, max_hist_len: int, titles_enc: torch.Tensor
    ):
        self = MIND.__new__(MIND)

        self.vocab = vocab
        self.nid2idx = nid2idx
        self.behaviors = behaviors
        self.histories = histories
        self.max_title_len = max_title_len
        self.max_hist_len = max_hist_len

        self.titles_enc = titles_enc

        return self

    def _load_news(
            self, news_file: Union[str, bytes, "os.PathLike[str]"], max_word_freq: float, min_word_occur: int,
            pad_tok: str, vocab: Optional[Dict[str, int]], max_vocab_len: Optional[int]
    ) -> None:
        """Load the datapoints from the news file.

        Args:
            news_file (Union[str, bytes, "PathLike[str]"]): path to the news file.
            max_word_freq (float): maximum word frequency for the word to be included in the vocabulary. 
            min_word_occur (int): minimal word occurrence for it to be included in the vocabulary. 
            pad_tok (str): token used for padding.
            vocab (Optional[Dict[str, int]], optional): predefined vocabulary. If None provided, will be build based on the news articles' titles. Out of vocabulary token should be placed at index 0, or be called <OOV>. Defaults to None.
            max_vocab_len (Optional[int], optional): maximum number of tokens in the vocabulary. If set to None (or non-positive int), no limit is set. Defaults to None.
        """
        # Skip repetitive work
        if self.nid2idx:
            return

        # add id for an dummy article
        self.nid2idx["<UNK>"] = 0

        # the title at index 0 is of the dummy article
        titles_tok = [[]]

        with open(news_file, encoding='utf8') as news:
            for dp in news:
                nid, cat, subcat, title, abstract, url, title_entities, abstract_entities = dp.strip("\n").split("\t")

                if not nid in self.nid2idx:
                    # notice that new news ids start from 1, 0 reserved for the dummy article
                    self.nid2idx[nid] = len(self.nid2idx)
                    titles_tok.append(self.tokenizer(title))

        if not vocab:  # no predefined vocabulary
            self._build_vocab(titles_tok, min_word_occur, max_word_freq, pad_tok)
        else:
            # make sure the vocabulary in sorted
            self.vocab = dict(sorted(vocab.items(), key=lambda pair: pair[1]))

        # trim the vocabulary to max_vocab_len
        if max_vocab_len is not None and max_vocab_len > 0:
            self.vocab = dict(list(self.vocab.items())[:max_vocab_len])

        # this will store the encoded titles of the news articles. Tensor #titles (index 0 for dummy article) x max_title_len
        self.titles_enc = torch.full(
                (len(titles_tok), self.max_title_len), fill_value=self.vocab.get(pad_tok, 0), dtype=torch.int64
        )

        for nid, title in enumerate(titles_tok):
            for tid, tok in enumerate(title[:self.max_title_len]):
                self.titles_enc[nid, tid] = self.vocab.get(tok) or self.vocab.get("<OOV>", 0)

    def _load_behaviors(
            self, behaviors_file: Union[str, bytes, "os.PathLike[str]"], impr_with_labels: bool, npratio: int
    ) -> None:
        """Load user behaviors from the behaviors file.

        Args:
            behaviors_file (Union[str, bytes, "PathLike[str]"): path to the behaviors file.
            impr_with_labels (bool, optional): whether there are labels available for the impressions. Should be set to false for the test dataset. Defaults to `True`.
            npratio (int, optional): negative and positive ratio for negative sampling. For each positive sample, `npratio` negative samples will be saved. -1 means no negative sampling. Defaults to -1.
        """
        # Skip repetitive work
        if self.behaviors:
            return

        with open(behaviors_file, encoding="utf8") as b:
            for dp in b:
                impr_id, uid, time, hist, impr = dp.strip("\n").split("\t")

                impr_id = int(impr_id)

                # drop NaNs
                if not (impr and hist):
                    continue

                hist = hist.split()
                hist_enc = [self.nid2idx.get(h, 0) for h in hist[len(hist) - self.max_hist_len:]]
                hist_len = len(hist_enc)

                # history represented as [0 ... 0 (padding), news ids]
                hist_enc_padded = [0] * (self.max_hist_len - hist_len) + hist_enc
                # mask with True at padding positions and False at the other ones.
                hist_mask = torch.tensor(
                        [True] * (self.max_hist_len - hist_len) + [False] * hist_len, dtype=torch.bool
                )

                # add this history to the table
                self.histories[impr_id] = MIND._HistoryEntry(hist_enc=hist_enc_padded, hist_mask=hist_mask)

                impressions = [i.split("-") for i in impr.split()]

                if impr_with_labels and npratio >= 0:
                    positive = list(filter(lambda p: p[1] == "1", impressions))
                    positive = rnd.sample(positive, k=len(positive))  # shuffle
                    negative = list(filter(lambda p: p[1] == "0", impressions))
                    negative = rnd.sample(negative, k=len(negative))  # shuffle

                    negative_to_take = min(len(negative), len(positive) * npratio)
                    positive_to_take = negative_to_take // npratio

                    pos_and_neg = positive[:positive_to_take] + negative[:negative_to_take]

                    impressions = rnd.sample(pos_and_neg, k=len(pos_and_neg))  # shuffle

                for i in impressions:
                    cand = self.nid2idx.get(i[0], 0)
                    label = int(i[1]) if impr_with_labels else 0

                    self.behaviors.append(
                            MIND._BehaviorEntry(uid=uid, impr_id=impr_id, cand_news_id=cand, label=label)
                    )

    def _build_vocab(
            self,
            titles_toks: List[List[str]],
            min_occur: int = 5,
            max_frequency: float = 0.8,
            pad_tok: Optional[str] = "<PAD>"
    ) -> None:
        """Build vocabulary of tokens in `titles_toks`.

        Args:
            titles_toks (List[List[str]]): totkens in the titles.
            min_occur (int, optional): minimal word occurrence for it to be included in the vocabulary. Defaults to 5.
            max_frequency (float, optional): maximum word frequency for the word to be included in the vocabulary. Defaults to 0.8.
            pad_tok (Optional[str], optional): token used for padding. Defaults to "<PAD>".
        """
        token_counts = Counter()

        n_titles = len(titles_toks)

        for title in titles_toks:
            unique_tokens = set(title)

            for tok in unique_tokens:
                token_counts[tok] += 1

        word_cnt = [(w, c) for w, c in token_counts.items() if c >= min_occur and c / n_titles <= max_frequency]

        word_cnt = sorted(word_cnt, key=lambda x: x[1], reverse=True)

        if pad_tok:
            word_cnt = [(pad_tok, 0), ("<OOV>", 0)] + word_cnt

        self.vocab = {w: i for i, (w, _) in enumerate(word_cnt)}

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Get the behavior datapoint at `index`.

        Args:
            index (int): index of the datapoint in the dataset.
        
        Returns:
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, int]: impression id, encoded history titles ([max_hist_size, max_title_len]),
                history mask of the same size (indicate the position of padding entries), encoded candidates' title ([max_title_len]),
                and label for that candidate. 
        """
        b = self.behaviors[index]
        h = self.histories[b.impr_id]

        cand_title = self.titles_enc[b.cand_news_id]
        history_titles = self.titles_enc[h.hist_enc]

        return b.impr_id, history_titles.contiguous(), h.hist_mask, cand_title, b.label

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.

        Returns:
            int: number of behaviors in the dataset.
        """
        return len(self.behaviors)

    def into_federated(self) -> "MINDFed":
        """Instantiate a federated version of the dataset from the current centralized version."""
        return MINDFed.init_from(self)


class MINDFed:
    """A federated version of the MIND. Each user holds only their own datapoints."""

    _per_user_ds: Dict[str, MIND]
    """Mapping from raw user id to the dataset of their datapoints only."""

    @staticmethod
    def init_from(mind: MIND) -> "MINDFed":
        """Initialize MINDFed from an instance of MIND.
        
        Returns:
            MINDFed: a new instance. 
        """
        return MINDFed(
                mind.behaviors, mind.histories, mind.titles_enc, mind.vocab, mind.nid2idx, mind.max_title_len,
                mind.max_hist_len
        )

    def __init__(
            self, behaviors: List[MIND._BehaviorEntry], histories: Dict[int, MIND._HistoryEntry],
            titles_enc: torch.Tensor, vocab: Dict[str, int], nid2idx: Dict[str,
                                                                           int], max_title_len: int, max_hist_len: int
    ) -> None:
        """Initialize the MIND dataset from raw parts.

        Args:
            behaviors (List[MIND._BehaviorEntry]): prediction candidates.
            titles_enc (torch.Tensor): encoded news titles.
            vocab (Dict[str, int]): title words vocabulary.
            nid2idx (Dict[str, int]): mapping from raw news ids to indexes. 
            max_title_len (int): maximum title length.
            max_hist_len (int): maximum history length.
        """
        self._per_user_ds = defaultdict(
                lambda: (
                        MIND._make_raw(
                                vocab=vocab,
                                nid2idx=nid2idx,
                                behaviors=[
                                ],  # we are going to fill in the behaviours later, with only behaviours of one client
                                histories=histories,
                                max_title_len=max_title_len,
                                max_hist_len=max_hist_len,
                                titles_enc=titles_enc
                        )
                )
        )

        for behavior in behaviors:
            uid = behavior.uid

            self._per_user_ds[uid].behaviors.append(behavior)
            # at this point `_per_user_ds` contains MIND DataSets per each user

        self._per_user_ds = dict(self._per_user_ds)

    def get_user_ids(self) -> Set[str]:
        """Get all ids of the users with the data.

        Returns:
            Set[str]: all user ids with data.
        """
        return set(self._per_user_ds)

    def __getitem__(self, uid: str) -> MIND:
        """Get the dataset for user with `uid`.

        Args:
            uid (str): raw user id.

        Raises:
            KeyError: there is no with the given identifier. 

        Returns:
            MIND: dataset for user with `uid`.
        """
        if not uid in self._per_user_ds:
            raise KeyError(f"no user for user id {uid}")

        return self._per_user_ds[uid]

    def sample(self, k: int) -> Generator[Tuple[str, MIND], None, None]:
        """Sample `k` users and yield their datasets. If the number of users is insufficient, all the users are returned.

        Args:
            k (int): number of users to be sampled.

        Returns:
            Generator[Tuple[str, MIND], None, None]: uids and datasets of the sampled users.
        """
        # NOTE we might have just skipped sampling and return all the users from `_per_user_ds`, but sampling also introduces shuffling,
        # thus there is still additional functionality even if `k >= len(_per_user_ds)`
        sampled_uids = rnd.sample(list(self._per_user_ds), k=min(k, len(self._per_user_ds)))

        for uid in sampled_uids:
            yield uid, self._per_user_ds[uid]

    def remove_if(self, predicate: Callable[[str, MIND], bool]) -> None:
        """Remove users and corresponding datasets for which the predicate returns `True`.

        Args:
            predicate (Callable[[str, MIND], bool]): predicate indicating which users to remove based on their uids and datasets.
        """
        users_to_remove = [uid for uid, ds in self._per_user_ds.items() if predicate(uid, ds)]

        for uid in users_to_remove:
            del self._per_user_ds[uid]

    def remove_insuff_len(self, n: int = 64) -> None:
        """Remove users with insufficient data.

        Args:
            n (int, optional): minimal number of behaviors recorded for user to be included. Defaults to 32.
        """
        self.remove_if(lambda _, ds: len(ds) < n)

    def remove_users(self, uids: Iterable[str]) -> None:
        """Remove users whose uids are in `uids`.

        Args:
            uids (Iterable[str]):  users to remove.
        """
        uids = set(uids)

        # Doing this directly would be faster that going over all the element of the _per_user_ds
        for uid in uids:
            del self._per_user_ds[uid]

    def remove_all_except(self, uids: Iterable[str]) -> None:
        """Remove all users, except those with uids in `uids`.

        Args:
            uids (Iterable[str]): uids of the users to save.
        """
        return self.remove_users(set(self._per_user_ds) - set(uids))

    def reduce_users_to(self, n: int) -> None:
        """Reduce the number of users in the system down to `n`. The users are selected randomly. To select users deterministically use `remove_all_except` or `remove_users`.

        Args:
            n (int): a number of users to be left in the system.
        """
        if n >= len(self._per_user_ds):
            return None

        saved_users = rnd.sample(set(self._per_user_ds), k=n)

        return self.remove_all_except(saved_users)

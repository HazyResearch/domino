import warnings
from itertools import combinations
from typing import Collection, Dict, Iterable, List, Mapping, Sequence, Set, Union

import meerkat as mk
import nltk
import numpy as np
import pandas as pd
import terra
from meerkat.contrib.gqa import read_gqa_dps
from nltk.corpus import wordnet as wn
from torchvision import transforms
from tqdm import tqdm

from domino.data.gqa import ATTRIBUTE_GROUPS, DATASET_DIR, split_gqa
from domino.slices.abstract import AbstractSliceBuilder

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


def _get_hypernyms(data_dp: mk.DataPanel):
    synsets = set(data_dp["synset"].unique())
    hypernyms = []

    for synset in synsets:
        synset = wn.synset(synset)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for hypernym in synset.closure(lambda s: s.hypernyms()):
                hypernyms.append(
                    {
                        "synset": synset.name(),
                        "hypernym": hypernym.name(),
                    }
                )

    # run through the hypernyms to get their hypernyms
    df = pd.DataFrame(hypernyms)
    for hypernym in df["hypernym"].unique():
        synset = wn.synset(hypernym)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for hypernym in synset.closure(lambda s: s.hypernyms()):
                hypernyms.append(
                    {
                        "synset": synset.name(),
                        "hypernym": hypernym.name(),
                    }
                )
    return pd.DataFrame(hypernyms)


class ImageNetSliceBuilder(AbstractSliceBuilder):
    def _prepare_dp(
        self,
        data_dp: mk.DataPanel,
        target_synset: str,
        slice_synsets: Union[str, Iterable[str]],
    ):
        data_dp = data_dp.view()
        hypernyms = _get_hypernyms(data_dp=data_dp)
        target_synsets = hypernyms["synset"][hypernyms["hypernym"] == target_synset]

        data_dp["target"] = data_dp["synset"].isin(target_synsets).astype(int).values

        if isinstance(slice_synsets, str):
            slice_synsets = [slice_synsets]

        n_slices = len(slice_synsets)
        slices = np.zeros((len(data_dp), n_slices))
        for slice_idx, slice_synset in enumerate(slice_synsets):
            slices[:, slice_idx] = (
                data_dp["synset"].isin(
                    [slice_synset]
                    + list(hypernyms["synset"][hypernyms["hypernym"] == slice_synset])
                    # include all hyponyms of the slice_synset when filtering datapanel
                )
            ).astype(int)
        data_dp["slices"] = slices
        data_dp["input"] = data_dp["image"]
        data_dp["id"] = data_dp["image_id"]

        return data_dp

    def build_rare_setting(
        self,
        data_dp: mk.DataPanel,
        target_synset: str,
        slice_synsets: Union[str, Iterable[str]],
        slice_frac: float,
        target_frac: float,
        n: int,
        **kwargs,
    ):
        data_dp = self._prepare_dp(
            data_dp, target_synset=target_synset, slice_synsets=slice_synsets
        )
        n_slices = len(slice_synsets)
        n_pos = int(n * target_frac)
        dp = data_dp.lz[
            np.random.permutation(
                np.concatenate(
                    (
                        *(
                            np.random.choice(
                                np.where(data_dp["slices"][:, slice_idx] == 1)[0],
                                int(slice_frac * n_pos),
                                replace=False,
                            )
                            for slice_idx in range(n_slices)
                        ),
                        np.random.choice(
                            np.where(
                                (data_dp["target"] == 1)
                                & (data_dp["slices"].sum(axis=1) == 0)
                            )[0],
                            int((1 - n_slices * slice_frac) * n_pos),
                            replace=False,
                        ),
                        np.random.choice(
                            np.where(data_dp["target"] == 0)[0],
                            n - n_pos,
                            replace=False,
                        ),
                    )
                )
            )
        ]
        return dp

    def build_correlation_setting(self):
        raise NotImplementedError

    def build_noisy_label_setting(
        self,
        data_dp: mk.DataPanel,
        target_synset: str,
        slice_synsets: Union[str, Iterable[str]],
        error_rate: float,
        target_frac: float,
        n: int,
        **kwargs,
    ):
        data_dp = self._prepare_dp(
            data_dp, target_synset=target_synset, slice_synsets=slice_synsets
        )
        n_pos = int(n * target_frac)

        pos_idxs = np.random.choice(
            np.where(data_dp["target"] == 1)[0],
            n_pos,
            replace=False,
        )
        neg_idxs = np.random.choice(
            np.where(data_dp["target"] == 0)[0],
            n - n_pos,
            replace=False,
        )
        dp = data_dp.lz[np.random.permutation(np.concatenate((pos_idxs, neg_idxs)))]

        # flip the labels in the slice with probability equal to `error_rate`
        flip = (np.random.rand(len(dp)) < error_rate) * dp["slices"].any(axis=1)
        dp["target"][flip] = np.abs(1 - dp["target"][flip])

        return dp

    def buid_noisy_feature_setting(self):
        raise NotImplementedError

    def _get_candidate_settings(
        self,
        data_dp: mk.DataPanel,
        hypernyms: mk.DataPanel,
        words_dp: mk.DataPanel,
        target_synsets: Iterable[str],
        num_slices: int,
    ):
        data_dp = data_dp.view()
        words = set(words_dp["word"])
        if target_synsets is None:
            target_synsets = DEFAULT_TARGET_SYNSETS

        for target_synset in tqdm(target_synsets):
            target_synsets = list(
                hypernyms[hypernyms["hypernym"] == target_synset]["synset"].unique()
            )
            targets = data_dp["synset"].isin(target_synsets)

            # only use synsets that are in our set of explanation words for slice, so
            # that we can compute mrr on natural languge explanations
            candidate_slice_synsets = _filter_synsets(target_synsets, words)
            # double list length so we can wrap around
            candidate_slice_synsets = candidate_slice_synsets * 2
            for start_idx in range(0, len(candidate_slice_synsets), num_slices):
                # TODO: ensure no overlapping
                slice_synsets = candidate_slice_synsets[
                    start_idx : start_idx + num_slices
                ]

                yield slice_synsets, target_synset, targets

    def collect_rare_settings(
        self,
        data_dp: mk.DataPanel,
        words_dp: mk.DataPanel,
        target_synsets: Iterable[str] = None,
        min_slice_frac: float = 0.001,
        max_slice_frac: float = 0.001,
        num_frac: int = 1,
        num_slices: int = 3,
        n: int = 100_000,
    ) -> mk.DataPanel:
        settings = []
        hypernyms = _get_hypernyms(data_dp=data_dp)

        for slice_synsets, target_synset, targets in self._get_candidate_settings(
            data_dp=data_dp,
            words_dp=words_dp,
            hypernyms=hypernyms,
            target_synsets=target_synsets,
            num_slices=num_slices,
        ):
            in_slice = data_dp["synset"].isin(
                slice_synsets
                + list(hypernyms["synset"][hypernyms["hypernym"].isin(slice_synsets)])
                # include all hyponyms of the slice_synsets when filtering datapanel
            )
            out_slice = (in_slice == 0) & (targets == 1)

            # get the maximum class balance (up to 0.5) for which the n is possible
            target_frac = min(
                0.5,
                in_slice.sum() / int(max_slice_frac * n),
                out_slice.sum() / int((1 - min_slice_frac) * n),
                targets.sum() / n,
            )
            settings.extend(
                [
                    {
                        "dataset": "imagenet",
                        "slice_category": "rare",
                        "alpha": slice_frac,
                        "target_name": target_synset,
                        "slice_names": slice_synsets,
                        "build_setting_kwargs": {
                            "target_frac": target_frac,
                            "slice_frac": slice_frac,
                            "n": n,
                            "slice_synsets": slice_synsets,
                            "target_synset": target_synset,
                        },
                    }
                    for slice_frac in np.geomspace(
                        min_slice_frac, max_slice_frac, num_frac
                    )
                ]
            )
        return mk.DataPanel(settings)

    def collect_noisy_label_settings(
        self,
        data_dp: mk.DataPanel,
        words_dp: mk.DataPanel,
        target_synsets: Iterable[str] = None,
        min_error_rate: float = 0.1,
        max_error_rate: float = 0.5,
        num_samples: int = 1,
        num_slices: int = 3,
        n: int = 100_000,
    ) -> mk.DataPanel:
        settings = []
        hypernyms = _get_hypernyms(data_dp=data_dp)

        for slice_synsets, target_synset, targets in self._get_candidate_settings(
            data_dp=data_dp,
            words_dp=words_dp,
            hypernyms=hypernyms,
            target_synsets=target_synsets,
            num_slices=num_slices,
        ):
            # get the maximum class balance (up to 0.5) for which the n is possible
            target_frac = min(
                0.5,
                targets.sum() / n,
            )
            settings.extend(
                [
                    {
                        "dataset": "imagenet",
                        "slice_category": "noisy_label",
                        "alpha": error_rate,
                        "target_name": target_synset,
                        "slice_names": slice_synsets,
                        "build_setting_kwargs": {
                            "target_frac": target_frac,
                            "error_rate": error_rate,
                            "n": n,
                            "slice_synsets": slice_synsets,
                            "target_synset": target_synset,
                        },
                    }
                    for error_rate in np.linspace(
                        min_error_rate, max_error_rate, num_samples
                    )
                ]
            )
        return mk.DataPanel(settings)


DEFAULT_TARGET_SYNSETS = [
    "vehicle.n.01",
    "clothing.n.01",
    "food.n.01",
    "musical_instrument.n.01",
    "bird.n.01",
    "vegetable.n.01",
    "fruit.n.01",
    "fish.n.01",
    "car.n.01",
    "ball.n.01",
    "building.n.01",
    "dog.n.01",
    "cat.n.01",
    "big_cat.n.01",
    "timepiece.n.01",
]


def _filter_synsets(synsets: Iterable[str], words: Set[str]):
    filtered_synsets = []
    for synset in synsets:
        synset = wn.synset(synset)
        if len(set(map(lambda x: x.lower(), synset.lemma_names())) & words) > 0:
            filtered_synsets.append(synset.name())

    return filtered_synsets

from typing import Iterable, Union

import meerkat as mk
import numpy as np
from tqdm import tqdm

from .abstract import AbstractSliceBuilder
from .utils import CorrelationImpossibleError, induce_correlation

# this is a subset of the full attribute set
ATTRIBUTES = [
    "bald",
    "bangs",
    "black_hair",
    "blond_hair",
    "blurry",
    "brown_hair",
    "eyeglasses",
    "goatee",
    "gray_hair",
    "male",
    "mustache",
    "no_beard",
    "smiling",
    "wearing_earrings",
    "wearing_hat",
    "wearing_lipstick",
    "wearing_necklace",
    "wearing_necktie",
    "young",
]


class CelebASliceBuilder(AbstractSliceBuilder):
    def build_correlation_setting(
        self,
        data_dp: mk.DataPanel,
        target: str,
        correlate: str,
        corr: float,
        n: int,
        **kwargs,
    ):
        indices = induce_correlation(
            dp=data_dp,
            corr=corr,
            attr_a=target,
            attr_b=correlate,
            match_mu=True,
            n=n,
        )
        dp = data_dp.lz[indices]

        dp["slices"] = np.array(
            [
                (dp[target] == 0) & (dp[correlate] == 1),
                (dp[target] == 1) & (dp[correlate] == 0),
            ]
        ).T
        dp["target"] = dp[target].values
        dp["correlate"] = dp[correlate].values
        dp["input"] = dp["image"]
        dp["id"] = dp["image_id"]
        return dp

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
        raise NotImplementedError

    def build_noisy_label_setting(self):
        raise NotImplementedError

    def buid_noisy_feature_setting(self):
        raise NotImplementedError

    def collect_correlation_settings(
        self,
        data_dp: mk.DataPanel,
        min_corr: float = 0.0,
        max_corr: float = 0.8,
        num_corr: int = 5,
        n: int = 20_000,
    ):

        # attribute -> correlate, object -> target
        settings = []
        for target in tqdm(ATTRIBUTES):
            for correlate in ATTRIBUTES:
                if target == correlate:
                    continue

                try:
                    for corr in [
                        min_corr,
                        max_corr,
                    ]:
                        _ = induce_correlation(
                            dp=data_dp,
                            corr=corr,
                            attr_a=target,
                            attr_b=correlate,
                            match_mu=False,
                            n=n,
                        )

                    settings.extend(
                        [
                            {
                                "dataset": "celeba",
                                "slice_category": "correlation",
                                "alpha": corr,
                                "target_name": target,
                                "slice_names": [
                                    f"{target}=0_{correlate}=1",
                                    f"{target}=1_{correlate}=0",
                                ],
                                "build_setting_kwargs": {
                                    "n": n,
                                    "correlate": correlate,
                                    "target": target,
                                    "corr": corr,
                                },
                            }
                            for corr in np.linspace(min_corr, max_corr, num_corr)
                        ]
                    )
                except CorrelationImpossibleError:
                    pass
        return mk.DataPanel(settings)

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
    ):
        data_dp = data_dp.view()
        hypernyms = _get_hypernyms(data_dp=data_dp)
        words = set(words_dp["word"])
        if target_synsets is None:
            target_synsets = DEFAULT_TARGET_SYNSETS

        settings = []
        for target_synset in tqdm(target_synsets):
            target_synsets = list(
                hypernyms[hypernyms["hypernym"] == target_synset]["synset"].unique()
            )
            targets = data_dp["synset"].isin(target_synsets)

            # only use synsets that are in our set of explanation words for slice, so that
            # we can compute mrr on natural languge explanations
            candidate_slice_synsets = _filter_synsets(target_synsets, words)
            # double list length so we can wrap around
            candidate_slice_synsets = candidate_slice_synsets * 2
            for start_idx in range(0, len(candidate_slice_synsets), num_slices):
                # TODO: ensure no overlapping
                slice_synsets = candidate_slice_synsets[
                    start_idx : start_idx + num_slices
                ]

                in_slice = data_dp["synset"].isin(
                    slice_synsets
                    + list(
                        hypernyms["synset"][hypernyms["hypernym"].isin(slice_synsets)]
                    )
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
                            "target_frac": target_frac,
                            "slice_frac": slice_frac,
                            "n": n,
                            "slice_synsets": slice_synsets,
                            "target_synset": target_synset,
                        }
                        for slice_frac in np.geomspace(
                            min_slice_frac, max_slice_frac, num_frac
                        )
                    ]
                )
        return mk.DataPanel(settings)

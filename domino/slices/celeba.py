from typing import Iterable, List, Union

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
        target_attrs: List[str],
        slice_attrs: List[str],
        slice_frac: float,
        target_frac: float,
        n: int,
        **kwargs,
    ):
        targets = np.any(
            np.array([data_dp[attr].data for attr in target_attrs]), axis=0
        ).T

        data_dp["slices"] = np.array([data_dp[attr].data for attr in slice_attrs]).T

        data_dp["target"] = targets
        data_dp["input"] = data_dp["image"]
        data_dp["id"] = data_dp["image_id"]

        n_slices = len(slice_attrs)
        n_pos = int(n * target_frac)
        dp = data_dp.lz[
            np.random.permutation(
                np.concatenate(
                    (
                        *(
                            np.random.choice(
                                np.where(
                                    (data_dp["slices"][:, slice_idx] == 1)
                                    & (data_dp["slices"].sum(axis=1) == 1)
                                    # ensure no other slices are 1
                                )[0],
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

        return mk.DataPanel(settings)

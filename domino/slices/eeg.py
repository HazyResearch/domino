from typing import Dict, List, Mapping, Sequence

import meerkat as mk
import numpy as np
import terra
from meerkat.contrib.eeg import build_stanford_eeg_dp
from torchvision import transforms
from tqdm import tqdm

from domino.slices.abstract import AbstractSliceBuilder

from .utils import CorrelationImpossibleError, induce_correlation


class EegSliceBuilder(AbstractSliceBuilder):
    def build_correlation_setting(
        self,
        data_dp: int,
        correlate: str,
        corr: float,
        n: int,
        correlate_threshold: float = None,
        **kwargs,
    ) -> mk.DataPanel:

        if correlate_threshold:
            data_dp[f"binarized_{correlate}"] = (
                data_dp[correlate].data > correlate_threshold
            ).astype(int)
            correlate = f"binarized_{correlate}"

        indices = induce_correlation(
            dp=data_dp,
            corr=corr,
            attr_a="target",
            attr_b=correlate,
            match_mu=True,
            n=n,
        )

        dp = data_dp.lz[indices]

        # define the "slices" column
        # for a spurious correlation, in-slice is where the correlate != target
        # slice_0 = dp[correlate] != dp["target"]
        # dp["slices"] = slice_0.reshape(-1, 1)
        dp["slices"] = np.array(
            [
                (dp["target"] == 0) & (dp[correlate] == 1),
                (dp["target"] == 1) & (dp[correlate] == 0),
            ]
        ).T

        return dp

    def build_rare_setting(
        self,
        data_dp: mk.DataPanel,
        attribute: str,
        attribute_thresh: float,
        slice_frac: float,
        n: int,
        **kwargs,
    ):

        data_dp["slices"] = np.array([data_dp[attribute].data < attribute_thresh]).T

        dp = data_dp.lz[
            np.random.permutation(
                np.concatenate(
                    (
                        np.random.choice(
                            np.where(data_dp["slices"].sum(axis=1) == 1)[0],
                            int(slice_frac * n),
                            replace=False,
                        ),
                        np.random.choice(
                            np.where(data_dp["slices"].sum(axis=1) == 0)[0],
                            int((1 - slice_frac) * n),
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

    def collect_rare_settings(
        self,
        data_dp: mk.DataPanel,
        attributes: List[str] = ["age"],
        attribute_thresholds: List[float] = [1],
        min_slice_frac: float = 0.001,
        max_slice_frac: float = 0.001,
        num_frac: int = 1,
        n: int = 8000,
    ):
        settings = []
        for attribute in attributes:
            for attribute_threshold in attribute_thresholds:
                settings.extend(
                    [
                        {
                            "dataset": "eeg",
                            "slice_category": "rare",
                            "alpha": slice_frac,
                            "target_name": "sz",
                            "slice_names": [f"{attribute}<{attribute_threshold}"],
                            "build_setting_kwargs": {
                                "attribute": attribute,
                                "attribute_thresh": attribute_threshold,
                                "slice_frac": slice_frac,
                                "n": n,
                            },
                        }
                        for slice_frac in np.geomspace(
                            min_slice_frac, max_slice_frac, num_frac
                        )
                    ]
                )

        return mk.DataPanel(settings)

    def collect_correlation_settings(
        self,
        data_dp: mk.DataPanel,
        min_corr: float = 0.0,
        max_corr: float = 0.8,
        num_corr: int = 5,
        correlate_thresholds: List[float] = [1],
        correlate_list: List[str] = ["age"],
        n: int = 8000,
    ) -> mk.DataPanel:

        settings = []
        for correlate in correlate_list:
            for correlate_threshold in correlate_thresholds:
                try:
                    for corr in [
                        min_corr,
                        max_corr,
                    ]:
                        if correlate_threshold:
                            data_dp[f"binarized_{correlate}"] = (
                                data_dp[correlate].data > correlate_threshold
                            ).astype(int)
                            correlate_ = f"binarized_{correlate}"
                        _ = induce_correlation(
                            dp=data_dp,
                            corr=corr,
                            attr_a="target",
                            attr_b=correlate_,
                            match_mu=True,
                            n=n,
                        )

                    settings.extend(
                        [
                            {
                                "dataset": "eeg",
                                "slice_category": "correlation",
                                "alpha": corr,
                                "target_name": "sz",
                                "slice_names": [
                                    f"sz=0_{correlate}>{correlate_threshold}",
                                    f"sz=1_{correlate}<{correlate_threshold}",
                                ],
                                "build_setting_kwargs": {
                                    "correlate": correlate,
                                    "corr": corr,
                                    "correlate_threshold": correlate_threshold,
                                    "n": n,
                                },
                            }
                            for corr in np.linspace(min_corr, max_corr, num_corr)
                        ]
                    )

                except CorrelationImpossibleError:
                    pass

        return mk.DataPanel(settings)

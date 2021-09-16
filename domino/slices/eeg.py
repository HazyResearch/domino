from typing import Dict, List, Mapping, Sequence

import meerkat as mk
import numpy as np
import terra
from meerkat.contrib.eeg import build_stanford_eeg_dp
from torchvision import transforms
from tqdm import tqdm

from domino.slices.abstract import AbstractSliceBuilder

from .utils import induce_correlation


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

        return dp

    def build_rare_setting(self):
        raise NotImplementedError

    def build_noisy_label_setting(self):
        raise NotImplementedError

    def buid_noisy_feature_setting(self):
        raise NotImplementedError

    def collect_correlation_settings(
        self,
        correlate_list: List[str],
        corr_list: List[float],
        correlate_thresholds: List[float] = None,
        n: int = 2500,
    ) -> mk.DataPanel:

        settings = []
        for ndx, correlate in enumerate(correlate_list):
            for corr in corr_list:
                settings.append(
                    {
                        "dataset": "eeg",
                        "slice_category": "correlation",
                        "correlate": correlate,
                        "corr": corr,
                        "correlate_threshold": correlate_thresholds[ndx],
                        "n": n,
                    }
                )

        return mk.DataPanel(settings)

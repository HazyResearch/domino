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
            [((dp[correlate] == 0) * dp["target"]), (dp[correlate] * dp["target"])]
        ).T

        return dp

    def build_rare_setting(self):
        raise NotImplementedError

    def build_noisy_label_setting(self):
        raise NotImplementedError

    def buid_noisy_feature_setting(self):
        raise NotImplementedError

    def collect_correlation_settings(
        self,
        data_dp: mk.DataPanel,
        min_corr: float = 0.0,
        max_corr: float = 0.9,
        num_corr: int = 5,
        correlate_threshold: float = 1,
        correlate_list: List[str] = ["age"],
        n: int = 8000,
    ) -> mk.DataPanel:

        settings = []
        for correlate in correlate_list:

            try:
                for corr in [
                    min_corr,
                    max_corr,
                ]:
                    if correlate_threshold:
                        data_dp[f"binarized_{correlate}"] = (
                            data_dp[correlate].data > correlate_threshold
                        ).astype(int)
                        correlate = f"binarized_{correlate}"
                    _ = induce_correlation(
                        dp=data_dp,
                        corr=corr,
                        attr_a="target",
                        attr_b=correlate,
                        match_mu=True,
                        n=n,
                    )

                settings.extend(
                    [
                        {
                            "dataset": "eeg",
                            "slice_category": "correlation",
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

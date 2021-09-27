from typing import Dict, List, Mapping, Sequence

import meerkat as mk
import meerkat.contrib.mimic.gcs
import numpy as np
import terra
from torchvision import transforms
from tqdm import tqdm

from .abstract import AbstractSliceBuilder
from .utils import CorrelationImpossibleError, induce_correlation

ATTRIBUTES = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "enlarged_cardiomediastinum",
    "fracture",
    "lung_lesion",
    "lung_opacity",
    "no_finding",
    "pleural_effusion",
    "pleural_other",
    "pneumonia",
    "pneumothorax",
    "support_devices",
    "finding_group",
    "lung_group",
    "pleural_group",
    "cardio_group",
]


class MimicSliceBuilder(AbstractSliceBuilder):
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
        dp["input"] = dp["cxr_jpg_1024"]
        dp["id"] = dp["dicom_id"]
        return dp

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
                    for corr in [min_corr, max_corr]:
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
                                "dataset": "mimic",
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
        print("Number of Valid Settings:", len(settings))
        return mk.DataPanel(settings)

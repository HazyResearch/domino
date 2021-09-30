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

LABEL_HIERARCHY = {   
    "lung_group": ['lung_opacity', 'edema', 'consolidation', 'pneumonia', 'lung_lesion', 'atelectasis'],
    "pleural_group": ['pleural_other', 'pleural_effusion', 'pneumothorax'],
    "cardio_group": ['enlarged_cardiomediastinum', 'cardiomegaly']
}

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
    
    def build_rare_setting(
        self,
        data_dp: mk.DataPanel,
        target_name: str,
        slice_name: str,
        slice_frac: float,
        target_frac: float,
        n: int,
        **kwargs,
    ):
        data_dp = data_dp.view()
        data_dp["target"] = np.array(data_dp[target_name]==1)
        data_dp["slices"] = np.array((data_dp[target_name]==1) & (data_dp[slice_name]==1)).reshape(-1,1)
        data_dp["input"] = data_dp["cxr_jpg_1024"]
        data_dp["id"] = data_dp["dicom_id"]
        n_pos = int(n * target_frac)
        dp = data_dp.lz[
            np.random.permutation(
                np.concatenate(
                    (
                        np.random.choice(
                            np.where(data_dp["slices"][:, 0] == 1)[0],
                            int(slice_frac * n_pos),
                            replace=False,
                        ),
                        np.random.choice(
                            np.where((data_dp["target"] == 1) & (data_dp["slices"][:,0] == 0))[0],
                            int((1 - slice_frac) * n_pos),
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
                    for corr in [min_corr,max_corr]:
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
        print('Number of Valid Settings:', len(settings))
        return mk.DataPanel(settings)
    
    def collect_rare_settings(
        self,
        data_dp: mk.DataPanel,
        min_slice_frac: float = 0.001,
        max_slice_frac: float = 0.001,
        num_frac: int = 1,
        num_slices: int = 3,
        n: int = 100_000,
    ):
        data_dp = data_dp.view()
        target_groups = LABEL_HIERARCHY.keys()

        settings = []
        for target in tqdm(target_groups):
            targets = data_dp[target]==1
            for subgroup in LABEL_HIERARCHY[target]:
                in_slice = data_dp[subgroup]==1
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
                            "dataset": "mimic",
                            "slice_category": "rare",
                            "alpha": slice_frac,
                            "target_name": target,
                            "slice_names": [subgroup],
                            "build_setting_kwargs": {
                                "target_name": target,
                                "slice_name": subgroup,
                                "target_frac": target_frac,
                                "slice_frac": slice_frac,
                                "n": n,
                            },
                        }
                        for slice_frac in np.geomspace(
                            min_slice_frac, max_slice_frac, num_frac
                        )
                    ]
                )
        print('Number of Valid Settings:', len(settings))
        return mk.DataPanel(settings)
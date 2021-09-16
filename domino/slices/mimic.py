from typing import Dict, List, Mapping, Sequence

import meerkat as mk
import numpy as np
import terra
from torchvision import transforms
from tqdm import tqdm
import meerkat.contrib.mimic.gcs

from . import CorrelationImpossibleError, induce_correlation, synthesize_preds

def build_correlation_slice(target: str, correlate: str, corr: float, n: int, dataset_dir: str, **kwargs) -> mk.DataPanel:

    dp = mk.DataPanel.read(dataset_dir)
    
    indices = induce_correlation(
        dp=dp,
        corr=corr,
        attr_a=target,
        attr_b=correlate,
        match_mu=True,
        n=n,
    )

    dp = dp.lz[indices]

    return dp

'''
def build_rare_slice(target_objects: Sequence[str], objects: Sequence[str], attributes: Sequence[str], slice_frac: float, target_frac: float, n: int, \
                     dataset_dir: str = DATASET_DIR, gqa_dps: Mapping[str, mk.DataPanel] = None, **kwargs):
    dps = read_gqa_dps(dataset_dir=dataset_dir) if gqa_dps is None else gqa_dps
    attr_dp, object_dp = dps["attributes"], dps["objects"].view()

    object_dp["target"] = object_dp["name"].isin(target_objects).astype(int)

    object_ids = mk.concat(
        (
            object_dp.lz[object_dp["name"].isin(objects)]["object_id"],
            attr_dp[attr_dp["attribute"].isin(attributes)]["object_id"],
        )
    )
    object_dp["slice"] = (
        np.isin(object_dp["object_id"], object_ids).astype(int) & object_dp["target"]
        == 1
    ).astype(int)

    # Issue: other objects in the same image as an in-slice object may overlap with the
    # in-slice object (e.g. slice=surfer, other object is wave). This leads to a large
    # number of objects that exclude any other objects from the images containing
    # in-slice objects.
    slice_image_ids = object_dp["image_id"][object_dp["slice"] == 1]
    object_dp = object_dp.lz[
        (object_dp["slice"] == 1) | (~np.isin(object_dp["image_id"], slice_image_ids))
    ]

    object_dp["input"] = object_dp["object_image"]
    object_dp["id"] = object_dp["object_id"]
    n_pos = int(n * target_frac)

    dp = object_dp.lz[
        np.random.permutation(
            np.concatenate(
                (
                    np.random.choice(
                        np.where(object_dp["slice"] == 1)[0],
                        int(slice_frac * n_pos),
                        replace=False,
                    ),
                    np.random.choice(
                        np.where(
                            (object_dp["target"] == 1) & (object_dp["slice"] == 0),
                        )[0],
                        int((1 - slice_frac) * n_pos),
                        replace=False,
                    ),
                    np.random.choice(
                        np.where(object_dp["target"] == 0)[0], n - n_pos, replace=False
                    ),
                )
            )
        )
    ]
    return dp

'''
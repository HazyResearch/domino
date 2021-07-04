import os

import meerkat as mk
import numpy as np
import pandas as pd
import PIL
import terra
import torch
from meerkat.contrib.mimic import GCSImageColumn, build_mimic_dp
from torchvision.transforms import Compose, Lambda, Resize
from torchxrayvision.datasets import XRayCenterCrop, XRayResizer, normalize


def mimic_transform(img: PIL.Image.Image):
    transform = Compose(
        [
            Resize([224, 224]),
            Lambda(lambda x: normalize(np.array(x), 255)[None, :, :]),
            XRayCenterCrop(),
            Lambda(lambda x: torch.tensor(x).expand(3, -1, -1)),
        ]
    )
    return transform(img)


@terra.Task.make_task
def build_dp(
    dataset_dir: str, gcp_project: str, resize: int = 512, run_dir: str = None, **kwargs
):
    dp = build_mimic_dp(dataset_dir=dataset_dir, gcp_project=gcp_project, **kwargs)
    dp = dp.lz[np.isin(dp["ViewPosition"].data, ["PA", "AP"])]

    resized_paths = pd.Series(dp["path"].data).apply(
        lambda x: os.path.join(
            dataset_dir, os.path.splitext(x)[0] + f"_{resize}" + ".jpg"
        )
    )
    dp["input"] = mk.ImageColumn.from_filepaths(
        filepaths=resized_paths, transform=mimic_transform, loader=PIL.Image.open
    )

    for col in CHEXPERT_COLUMNS:
        dp[col] = (dp[col] == 1).astype(int)

    return dp


CHEXPERT_COLUMNS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged_Cardiomediastinum",
    "Fracture",
    "Lung_Lesion",
    "Lung_Opacity",
    "No_Finding",
    "Pleural_Effusion",
    "Pleural_Other",
    "Pneumonia",
    "Pneumothorax",
    "Support_Devices",
]

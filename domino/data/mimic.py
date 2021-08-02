import os
from functools import partial

import meerkat as mk
import numpy as np
import pandas as pd
import PIL
import terra
import torch
from meerkat.contrib.mimic import GCSImageColumn, build_mimic_dp
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor
from torchxrayvision.datasets import XRayCenterCrop, XRayResizer, normalize

CXR_MEAN = 0.48865
CXR_STD = 0.24621


def _mimic_transform(img: PIL.Image.Image, resolution: int = 224):
    transform = Compose(
        [
            Lambda(lambda x: np.array(x)[None, :, :]),
            XRayCenterCrop(),
            XRayResizer(resolution),
            Lambda(lambda x: torch.tensor(x).expand(3, -1, -1)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(img)


def mimic_transform(img: PIL.Image.Image, resolution: int = 224):
    transform = Compose(
        [
            Resize([resolution, resolution]),
            ToTensor(),
            Lambda(lambda x: torch.tensor(x).expand(3, -1, -1)),
            Normalize(CXR_MEAN, CXR_STD),
        ]
    )
    return transform(img)


@terra.Task.make_task
def build_dp(
    dataset_dir: str, gcp_project: str, resize: int = 512, run_dir: str = None, **kwargs
):
    dp = build_mimic_dp(dataset_dir=dataset_dir, gcp_project=gcp_project, **kwargs)
    dp = dp.lz[np.isin(dp["view_position"].data, ["PA", "AP"])]

    resized_paths = pd.Series(dp["path"].data).apply(
        lambda x: os.path.join(
            dataset_dir, os.path.splitext(x)[0] + f"_{resize}" + ".jpg"
        )
    )
    dp[f"img_{resize}"] = mk.ImageColumn.from_filepaths(
        filepaths=resized_paths, loader=PIL.Image.open
    )

    for resolution in [224, 512]:
        dp[f"input_{resolution}"] = mk.ImageColumn.from_filepaths(
            filepaths=resized_paths,
            transform=partial(mimic_transform, resolution=resolution),
            loader=PIL.Image.open,
        )

    for col in CHEXPERT_COLUMNS:
        dp[f"{col}_uzeros"] = (dp[col] == 1).astype(int)

    # convert some columns to binary
    dp["patient_orientation_rf"] = (
        dp["patient_orientation"].data == "['R', 'F']"
    ).astype(int)
    dp["young"] = (dp["anchor_age"] < 40).astype(int)
    dp["ethnicity_black"] = (dp["ethnicity"].data == "BLACK/AFRICAN AMERICAN").astype(
        int
    )
    dp["ethnicity_hisp"] = (dp["ethnicity"].data == "HISPANIC/LATINO").astype(int)
    dp["ethnicity_asian"] = (dp["ethnicity"].data == "ASIAN").astype(int)
    dp["burned_in_annotation"] = (dp["BurnedInAnnotation"].data == "YES").astype(int)
    dp["gender_male"] = (dp["gender"].data == "M").astype(int)

    return dp


CHEXPERT_COLUMNS = [
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
]

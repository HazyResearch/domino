import hashlib
import math
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

    resized_paths = pd.Series(dp["dicom_path"].data).apply(
        lambda x: os.path.join(
            dataset_dir, os.path.splitext(x)[0] + f"_{resize}" + ".jpg"
        )
    )
    dp[f"cxr_jpg_{resize}"] = mk.ImageColumn.from_filepaths(
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
    dp["ethnicity_white"] = (dp["ethnicity"].data == "WHITE").astype(int)
    dp["ethnicity_asian"] = (dp["ethnicity"].data == "ASIAN").astype(int)
    dp["burned_in_annotation"] = (dp["burned_in_annotation"].data == "YES").astype(int)
    dp["gender_male"] = (dp["gender"].data == "M").astype(int)

    return dp


def hash_for_split(example_id: str, salt=""):
    GRANULARITY = 100000
    hashed = hashlib.sha256((str(example_id) + salt).encode())
    hashed = int(hashed.hexdigest().encode(), 16) % GRANULARITY + 1
    return hashed / float(GRANULARITY)


@terra.Task.make_task
def split_dp(
    dp: mk.DataPanel,
    train_frac: float = 0.7,
    valid_frac: float = 0.1,
    test_frac: float = 0.2,
    other_splits: dict = None,
    salt: str = "",
    run_dir: str = None,
):
    other_splits = {} if other_splits is None else other_splits
    splits = {
        "train": train_frac,
        "valid": valid_frac,
        "test": test_frac,
        **other_splits,
    }

    if not math.isclose(sum(splits.values()), 1):
        raise ValueError("Split fractions must sum to 1.")

    dp["subject_hash"] = dp["subject_id"].apply(partial(hash_for_split, salt=salt))
    start = 0
    split_column = pd.Series(["unassigned"] * len(dp))
    for split, frac in splits.items():
        end = start + frac
        split_column[(start < dp["subject_hash"]) & (dp["subject_hash"] <= end)] = split
        start = end
    dp["split"] = split_column
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

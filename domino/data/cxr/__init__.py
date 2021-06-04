import os
from mosaic.cells.imagepath import ImagePath
import numpy as np
from functools import partial
from typing import List
from glob import glob
import pickle

import torch
import torch.nn as nn
from torchvision.models import resnet50
import pandas as pd
from terra import Task
import torchvision.transforms as transforms
import torchvision.datasets.folder as folder
from mosaic import DataPanel, ImageColumn
from mosaic.cells.volume import MedicalVolumeCell
from PIL import Image
from dosma import DicomReader

from domino.utils import hash_for_split

ROOT_DIR = "/home/common/datasets/cxr-tube"
CXR_MEAN = 0.48865
CXR_STD = 0.24621
CXR_SIZE = 224

@Task.make_task
def get_cxr_activations(dp: DataPanel, model_path: str, run_dir: str=None):
    from domino.bss_dp import SourceSeparator
    model= CXRResnet(model_path=model_path)
    separator = SourceSeparator(config={
        "activation_dim": 2048,
        "lr": 1e-3
    }, model=model)
    act_dp = separator.prepare_dp(
        dp=dp, layers={
            "block2": model.cnn_encoder[-3],
            "block3": model.cnn_encoder[-2],
            "block4": model.cnn_encoder[-1]
        }, batch_size=128
    )
    return act_dp

class CXRResnet(nn.Module):

    def __init__(self, model_path: str = None):
        super().__init__()
        input_module = resnet50(pretrained=False)
        modules = list(input_module.children())[:-2]
        self.avgpool = input_module.avgpool
        self.cnn_encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(in_features=2048, out_features=2)
        if model_path is not None:
            state_dict = torch.load(model_path)
            self.cnn_encoder.load_state_dict(state_dict["model"]["module_pool"]["cnn"])
            self.load_state_dict(
                state_dict["model"]["module_pool"]["classification_module_target"],
                strict=False
            )

    def forward(self, x):
        x = self.cnn_encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def cxr_transform_pil(volume: MedicalVolumeCell):
    array = volume._volume.squeeze()
    return Image.fromarray(np.uint8(array))


def cxr_transform(volume: MedicalVolumeCell):
    img = cxr_transform_pil(volume)
    img = transforms.Compose(
        [
            transforms.Resize([CXR_SIZE, CXR_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize(CXR_MEAN, CXR_STD),
        ]
    )(img)
    return img.repeat([3,1,1])

def get_dp(df: pd.DataFrame):
    dp = DataPanel.from_pandas(df)
    loader = DicomReader(group_by=None, default_ornt=("SI", "AP"))
    dp.add_column("input", dp["filepath"].map(
        lambda x: MedicalVolumeCell(
            paths=x, loader=loader, transform=cxr_transform
        ),
        num_workers=0, 
    ), overwrite=True
    )
    dp.add_column("img", dp["filepath"].map(
        lambda x: MedicalVolumeCell(
            paths=x, loader=loader, transform=cxr_transform_pil
        ),
        num_workers=0, 
    ), overwrite=True
    )
    return dp


@Task.make_task
def build_cxr_df(root_dir: str = ROOT_DIR, run_dir: str=None):
    # get segment annotations
    segment_df = pd.read_csv(os.path.join(ROOT_DIR, "train-rle.csv"))
    segment_df = segment_df.rename(
        columns={"ImageId": "image_id", " EncodedPixels": "encoded_pixels"}
    )
    # there are some image ids with multiple label rows, we'll just take the first
    segment_df = segment_df[~segment_df.image_id.duplicated(keep="first")]
    
    # get binary labels for pneumothorax, any row with a "-1" for encoded pixels is 
    # considered a negative 
    segment_df["pmx"] = (segment_df.encoded_pixels != "-1").astype(int)

    # start building up a main dataframe with a few `merge` operations (i.e. join)
    df = segment_df

    # get filepaths for all images in the "dicom-images-train" directory
    filepaths = sorted(glob(os.path.join(root_dir, "dicom-images-train/*/*/*.dcm")))
    filepath_df = pd.DataFrame(
        [
            {
                "filepath": filepath,
                "image_id": os.path.splitext(os.path.basename(filepath))[0],
            }
            for filepath in filepaths
        ]
    )

    # important to perform a left join here, because there are some images in the 
    # directory without labels in `segment_df`
    df = df.merge(filepath_df, how="left", on="image_id")

    # add in chest tube annotations
    rows = []
    for split in ["train", "test"]:
        tube_dict = pickle.load(
            open(
                os.path.join(root_dir, f"cxr_tube_labels/cxr_tube_dict_{split}.pkl"),
                "rb",
            )
        )
        rows.extend(
            [
                {"image_id": k, "chest_tube": int(v), "split": split}
                for k, v in tube_dict.items()
            ]
        )
    tube_df = pd.DataFrame(rows)

    df = df.merge(tube_df, how="left", on="image_id")
    df.split = df.split.fillna("train")

    return df

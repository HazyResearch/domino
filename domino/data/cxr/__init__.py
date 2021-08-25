import os
import pickle
from functools import partial
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dosma import DicomReader
from meerkat import DataPanel
from meerkat.cells.volume import MedicalVolumeCell
from PIL import Image
from terra import Task
from torchvision.models import resnet50

from domino.data.gaze_utils import (
    apply_lf,
    diffusivity,
    make_heatmaps,
    max_visit,
    total_time,
    unique_visits,
)
from domino.vision import score

ROOT_DIR = "/home/common/datasets/cxr-tube"
CXR_MEAN = 0.48865
CXR_STD = 0.24621
CXR_SIZE = 224


# @Task.make_task
def get_cxr_activations(
    dp: DataPanel,
    model_path: str,
    run_type: str = "siim",
    num_classes=2,
    run_dir: str = None,
):
    if run_type == "mimic":
        model = resnet50(num_classes=14, pretrained=False)
        state_dict = {}
        for name, key in torch.load(model_path)["state_dict"].items():

            state_dict[name.split("model.")[-1]] = key
        model.load_state_dict(state_dict)
        modules = list(model.children())[:-2]
        cnn_encoder = nn.Sequential(*modules)
    elif run_type == "domino":
        model = resnet50(num_classes=num_classes, pretrained=False)
        state_dict = {}
        for name, key in torch.load(model_path)["state_dict"].items():
            state_dict[name.split("model.")[-1]] = key
        model.load_state_dict(state_dict)
        modules = list(model.children())[:-2]
        cnn_encoder = nn.Sequential(*modules)
    elif run_type == "siim":
        model = CXRResnet(model_path=model_path)
        cnn_encoder = model.cnn_encoder

    act_dp = score(
        model=model,
        dp=dp,
        layers={
            # "block2": model.cnn_encoder[-3],
            "block3": cnn_encoder[-2],
            "block4": cnn_encoder[-1],
        },
        batch_size=16,
    )
    return act_dp


def minmax_norm(dict_arr, key):
    arr = np.array([dict_arr[id][key] for id in dict_arr])
    for id in dict_arr:
        dict_arr[id][key] -= arr.min()
        dict_arr[id][key] /= arr.max()

    return dict_arr


@Task.make_task
def create_gaze_df(root_dir: str = ROOT_DIR, run_dir: str = None):

    # hypers for CXR gaze features:
    s1 = 3
    s2 = 3
    stride = 2
    view_pct = 0.1

    gaze_seq_dict = pickle.load(open(os.path.join(root_dir, "cxr_gaze_data.pkl"), "rb"))
    expert_labels_dict = pickle.load(
        open(os.path.join(root_dir, "expert_labels_dict.pkl"), "rb")
    )

    # Extract gaze features
    gaze_feats_dict = {}
    for img_id in gaze_seq_dict:
        gaze_seq = gaze_seq_dict[img_id]
        gaze_heatmap = make_heatmaps([gaze_seq]).squeeze()
        gaze_time = apply_lf([gaze_heatmap], total_time)[0]
        gaze_max_visit = apply_lf([gaze_heatmap], partial(max_visit, pct=view_pct))[0]
        gaze_unique = apply_lf([gaze_heatmap], unique_visits)[0]
        gaze_diffusivity = apply_lf(
            [gaze_heatmap], partial(diffusivity, s1=s1, s2=s2, stride=stride)
        )[0]

        gaze_feats_dict[img_id] = {
            "gaze_heatmap": gaze_heatmap,
            "gaze_time": gaze_time,
            "gaze_max_visit": gaze_max_visit,
            "gaze_unique": gaze_unique,
            "gaze_diffusivity": gaze_diffusivity,
        }

    # normalize features
    gaze_feats_dict = minmax_norm(gaze_feats_dict, "gaze_time")
    gaze_feats_dict = minmax_norm(gaze_feats_dict, "gaze_max_visit")
    gaze_feats_dict = minmax_norm(gaze_feats_dict, "gaze_unique")
    gaze_feats_dict = minmax_norm(gaze_feats_dict, "gaze_diffusivity")

    # merge sequences and features into a df
    gaze_df = pd.DataFrame(
        [
            {
                "gaze_seq": gaze_seq_dict[img_id],
                "gaze_heatmap": gaze_feats_dict[img_id]["gaze_heatmap"],
                "gaze_max_visit": gaze_feats_dict[img_id]["gaze_max_visit"],
                "gaze_unique": gaze_feats_dict[img_id]["gaze_unique"],
                "gaze_time": gaze_feats_dict[img_id]["gaze_time"],
                "gaze_diffusivity": gaze_feats_dict[img_id]["gaze_diffusivity"],
                "expert_label": expert_labels_dict[img_id],
                "image_id": os.path.splitext(os.path.basename(img_id))[0],
            }
            for img_id in expert_labels_dict
        ]
    )

    return gaze_df


class CXRResnet(nn.Module):
    def __init__(self, model_path: str = None, domino_run: bool = False):
        super().__init__()
        input_module = resnet50(pretrained=False)
        modules = list(input_module.children())[:-2]
        self.avgpool = input_module.avgpool
        self.cnn_encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(in_features=2048, out_features=2)
        if model_path is not None:
            state_dict = torch.load(model_path)
            if domino_run:
                state_dict = state_dict["state_dict"]

            else:
                self.cnn_encoder.load_state_dict(
                    state_dict["model"]["module_pool"]["cnn"]
                )
                self.load_state_dict(
                    state_dict["model"]["module_pool"]["classification_module_target"],
                    strict=False,
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
    return img.repeat([3, 1, 1])


def cxr_transform2(volume: MedicalVolumeCell):
    img = cxr_transform_pil(volume)
    img = transforms.Compose(
        [
            transforms.Resize([2 * CXR_SIZE, 2 * CXR_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize(CXR_MEAN, CXR_STD),
        ]
    )(img)
    return img.repeat([3, 1, 1])


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position : current_position + lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)


def get_dp(df: pd.DataFrame):
    dp = DataPanel.from_pandas(df)
    loader = DicomReader(group_by=None, default_ornt=("SI", "AP"))
    dp.add_column(
        "input",
        dp["filepath"].map(
            lambda x: MedicalVolumeCell(
                paths=x, loader=loader, transform=cxr_transform
            ),
            num_workers=0,
        ),
        overwrite=True,
    )
    # dp.add_column(
    #     "input2",
    #     dp["filepath"].map(
    #         lambda x: MedicalVolumeCell(
    #             paths=x, loader=loader, transform=cxr_transform2
    #         ),
    #         num_workers=0,
    #     ),
    #     overwrite=True,
    # )
    dp.add_column(
        "img",
        dp["filepath"].map(
            lambda x: MedicalVolumeCell(
                paths=x, loader=loader, transform=cxr_transform_pil
            ),
            num_workers=0,
        ),
        overwrite=True,
    )
    return dp


@Task.make_task
def build_cxr_df(root_dir: str = ROOT_DIR, run_dir: str = None):
    # get segment annotations
    segment_df = pd.read_csv(os.path.join(root_dir, "train-rle.csv"))
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

    # integrate gaze features
    create_gaze_df(root_dir)
    gaze_df = create_gaze_df.out(load=True)
    df = df.merge(gaze_df, how="left", on="image_id")

    return df

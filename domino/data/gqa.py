import os

import meerkat as mk
import terra
from meerkat.contrib.gqa import read_gqa_dps

from domino.utils import split_dp

DATASET_DIR = "/home/common/datasets/gqa"


@terra.Task.make_task
def split_gqa(
    dataset_dir: str = DATASET_DIR,
    train_frac: float = 0.7,
    valid_frac: float = 0.1,
    test_frac: float = 0.2,
    other_splits: dict = None,
    salt: str = "",
    run_dir: str = None,
):
    dps = read_gqa_dps(dataset_dir=dataset_dir)
    return split_dp(
        dps["images"],
        split_on="image_id",
        train_frac=train_frac,
        valid_frac=valid_frac,
        test_frac=test_frac,
        other_splits=other_splits,
        salt=salt,
    )[["image_id", "split"]]


def crop_object(row):
    img = row["image"]

    length = max(row["h"], row["w"])
    box = (
        max(row["x"] - ((length - row["w"]) / 2), 0),
        max(row["y"] - ((length - row["h"]) / 2), 0),
        min(row["x"] + row["w"] + ((length - row["w"]) / 2), img.width),
        min(row["y"] + row["h"] + ((length - row["h"]) / 2), img.height),
    )
    return img.crop(box)


ATTRIBUTE_GROUPS = {
    "darkness": ["dark", "bright"],
    "dryness": ["wet", "dry"],
    "colorful": ["colorful", "shiny"],
    "leaf": ["leafy", "bare"],
    "emotion": ["happy", "calm"],
    "sports": ["baseball", "tennis"],
    "flatness": ["flat", "curved"],
    "lightness": ["light", "heavy"],
    "gender": ["male", "female"],
    "width": ["wide", "narrow"],
    "depth": ["deep", "shallow"],
    "hardness": ["hard", "soft"],
    "cleanliness": ["clean", "dirty"],
    "switch": ["on", "off"],
    "thickness": ["thin", "thick"],
    "openness": ["open", "closed"],
    "height": ["tall", "short"],
    "length": ["long", "short"],
    "fullness": ["full", "empty"],
    "age": ["young", "old", "new"],
    "size": ["large", "small"],
    "pattern": ["checkered", "striped", "dress", "dotted"],
    "shape": ["round", "rectangular", "triangular", "square"],
    "activity": [
        "waiting",
        "drinking",
        "playing",
        "eating",
        "cooking",
        "resting",
        "sleeping",
        "posing",
        "talking",
        "looking down",
        "looking up",
        "driving",
        "reading",
        "brushing teeth",
        "flying",
        "surfing",
        "skiing",
        "hanging",
    ],
    "pose": [
        "walking",
        "standing",
        "lying",
        "sitting",
        "running",
        "jumping",
        "crouching",
        "bending",
        "grazing",
    ],
    "material": [
        "wood",
        "plastic",
        "metal",
        "glass",
        "leather",
        "leather",
        "porcelain",
        "concrete",
        "paper",
        "stone",
        "brick",
    ],
    "colors": [
        "white",
        "red",
        "black",
        "green",
        "silver",
        "gold",
        "khaki",
        "gray",
        "dark",
        "pink",
        "dark blue",
        "dark brown",
        "blue",
        "yellow",
        "tan",
        "brown",
        "orange",
        "purple",
        "beige",
        "blond",
        "brunette",
        "maroon",
        "light blue",
        "light brown",
    ],
}

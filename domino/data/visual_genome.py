from meerkat.contrib.visual_genome import read_visual_genome_dps

DATASET_DIR = "/home/common/datasets/visual-genome"

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


def get_dps(dataset_dir: str = DATASET_DIR):
    dps = read_visual_genome_dps(dataset_dir)
    image_dp = dps["images"]
    attr_dp = dps["attributes"]
    relationship_dp = dps["relationships"]
    object_dp = dps["images"].merge(dps["objects"], on="image_id")

    object_dp["object_image"] = object_dp.to_lambda(crop_object)
    return image_dp, attr_dp, object_dp, relationship_dp

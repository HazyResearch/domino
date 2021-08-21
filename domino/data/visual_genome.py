from meerkat.contrib.visual_genome import read_visual_genome_dps

DATASET_DIR = "/home/common/datasets/visual-genome"

ATTRIBUTE_GROUPS = {
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
    ]
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
    object_dp = dps["images"].merge(dps["objects"], on="image_id")

    object_dp["object_image"] = object_dp.to_lambda(crop_object)
    return image_dp, attr_dp, object_dp

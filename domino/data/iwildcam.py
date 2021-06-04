import os

import torch
import torch.nn as nn
import pandas as pd
from terra import Task
import torchvision.transforms as transforms
from torchvision.transforms.transforms import ToTensor
import torchvision


def iwildcam_transform(img: torch.Tensor):
    transform = transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(img)


iwildcam_task_config = {
    "img_column": "img_path",
    "id_column": "image_id",
    "target_column": "y",
    "img_transform": iwildcam_transform,
    "num_classes": 182,
    "metrics": ["accuracy", "macro_f1", "macro_recall"],
}


@Task.make_task
def build_iwildcam_df(
    dataset_dir: str = "/home/common/datasets/iwildcam_v2.0", run_dir: str = None
):
    df = pd.read_csv(os.path.join(dataset_dir, "metadata.csv"), index_col=0)

    df["img_path"] = df.filename.apply(lambda x: os.path.join(dataset_dir, "train", x))

    # rename valid split to be consistent with name in domino-21
    df.split = df.split.str.replace("val", "valid")

    # get category names into the dataframe
    category_df = pd.read_csv(os.path.join(dataset_dir, "categories.csv"))
    df = df.merge(category_df[["category_id", "name"]], on="category_id", how="left")
    df = df.rename(columns={"name": "category_name"})

    return df


def get_iwildcam_model(
    model_path="/home/common/datasets/iwildcam_v2.0/models/iwildcam_erm_seed2/best_model.pth",
    model_type="resnet50",
    **kwargs,
):
    d_out = iwildcam_task_config["num_classes"]
    # get constructor and last layer names
    if model_type == "wideresnet50":
        constructor_name = "wide_resnet50_2"
        last_layer_name = "fc"
    elif model_type == "densenet121":
        constructor_name = model_type
        last_layer_name = "classifier"
    elif model_type in ("resnet50", "resnet34"):
        constructor_name = model_type
        last_layer_name = "fc"
    else:
        raise ValueError(f"Torchvision model {model_type} not recognized")
    # construct the default model, which has the default last layer
    constructor = getattr(torchvision.models, constructor_name)
    model = constructor(**kwargs)
    # adjust the last layer
    d_features = getattr(model, last_layer_name).in_features
    last_layer = nn.Linear(d_features, d_out)
    model.d_out = d_out
    setattr(model, last_layer_name, last_layer)

    # wrap the model so it resembles the wilds setup
    class WrapperModule(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

    model = WrapperModule(model)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict["algorithm"])
    return model.model

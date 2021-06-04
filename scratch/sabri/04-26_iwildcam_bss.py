import os

os.environ["TERRA_CONFIG_PATH"] = "/home/sabri/code/domino-21/terra_config.json"
from domino.data.iwildcam import build_iwildcam_df, iwildcam_task_config
from domino.vision import Classifier
from domino.vision import fit_bss


model = Classifier(config={"model_name": "iwildcam", **iwildcam_task_config})

data_df = build_iwildcam_df.out(load=True)
data_df = data_df[data_df.split == "id_valid"]

separator = fit_bss(
    data_df=data_df.sample(2000),  # build_iwildcam_df.out(),
    model=model,
    config={
        "num_classes": 182,
        "class_idx": 49,
        "activation_dim": 2048,
        "lr": 1e-3,
        "pred_loss_weight": 10000,
    },
    split="id_valid",
    memmap=True,
    **iwildcam_task_config
)

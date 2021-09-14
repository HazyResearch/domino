import os
from typing import Sequence, Tuple

import meerkat as mk
import ray
import terra
from ray import tune
from torch import nn

from domino.data.mimic import build_dp, split_dp
from domino.vision import train


@terra.Task
def train_model(
    dp: mk.DataPanel,
    config: dict,
    target_column: Sequence[str],
    input_column: str = "input",
    id_column: str = "file",
    model: nn.Module = None,
    parent_run_id: int = None,
    run_dir: str = None,
    **kwargs,
):
    metadata = {
        "target_column": target_column,
        "input_column": input_column,
        "run_id": int(os.path.basename(run_dir)),
        "parent_run_id": parent_run_id,
        **config,
    }
    config["targets"] = target_column
    # model.config["lr"] = 1e-5
    train(
        dp=dp,
        # model=model,
        config=config,
        input_column=input_column,
        id_column=id_column,
        target_column=target_column,
        run_dir=run_dir,
        wandb_config=metadata,
        **kwargs,
    )

    return metadata


if __name__ == "__main__":
    train_model(
        dp=terra.out(4411),
        # model=terra.get_artifacts(4485, "best_chkpt")["model"],
        config={"lr": 1e-4, "arch": "resnet50"},
        input_column="input_224",
        target_column=[
            f"{tgt}_uzeros"
            for tgt in [
                "atelectasis",
                "cardiomegaly",
                "consolidation",
                "edema",
                "enlarged_cardiomediastinum",
                "fracture",
                "lung_opacity",
                "pleural_effusion",
                "pleural_other",
                "pneumonia",
                "pneumothorax",
                "support_devices",
                "lung_lesion",
                "no_finding",
            ]
        ],
        id_column="dicom_id",
        num_workers=6,
        val_check_interval=2000,
        num_sanity_val_steps=20,
        ckpt_monitor="valid/accuracy/no_finding_uzeros",
        max_epochs=4,
        gpus=[0],
        batch_size=64,
    )

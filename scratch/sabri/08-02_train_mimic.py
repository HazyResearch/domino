import os
from typing import Sequence, Tuple

import meerkat as mk
import ray
import terra
from ray import tune

from domino.data.mimic import build_dp, split_dp
from domino.vision import train


@terra.Task
def train_model(
    dp: mk.DataPanel,
    config: dict,
    target_column: str,
    input_column: str = "input",
    id_column: str = "file",
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

    train(
        dp=dp,
        config=config,
        input_column=input_column,
        id_column=id_column,
        target_column=target_column,
        run_dir=run_dir,
        wandb_config=metadata,
        **kwargs,
    )

    return metadata


@terra.Task
def train_models(
    dp_run_id: int,
    targets: Sequence[str],
    id_column: str,
    num_samples: float = 1,
    run_dir: str = None,
    **kwargs,
):
    def _train_model(config):
        import meerkat.contrib.mimic

        return train_model(
            terra.out(dp_run_id),
            id_column=id_column,
            pbar=False,
            weighted_sampling=True,
            parent_run_id=int(os.path.basename(run_dir)),
            **config,
            **kwargs,
        )

    ray.init(num_gpus=4, num_cpus=32)
    analysis = tune.run(
        _train_model,
        config={
            "input_column": tune.grid_search(["input_512"]),
            "target_column": tune.grid_search(list(targets)),
            "config": {"lr": 1e-4, "arch": "resnet50"},
        },
        num_samples=num_samples,
        resources_per_trial={"gpu": 1},
    )
    return analysis.dataframe()


if __name__ == "__main__":

    train_models(
        dp_run_id=4411,
        targets=[
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
            ]
        ],
        id_column="dicom_id",
        num_workers=6,
        val_check_interval=200,
        num_sanity_val_steps=20,
        max_epochs=1,
        gpus=[0],
        batch_size=16,
    )

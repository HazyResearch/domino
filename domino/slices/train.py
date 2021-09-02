import os

import meerkat as mk
import ray
import terra
from ray import tune

from domino.slices.gqa import build_correlation_slice
from domino.vision import train


@terra.Task.make_task
def train_model(
    dp: mk.DataPanel,
    config: dict,
    run_dir: str = None,
    **kwargs,
):
    # set seed
    metadata = config
    metadata["run_id"] = int(os.path.basename(run_dir))

    train(
        dp=dp,
        input_column="input",
        id_column="id",
        target_column="target",
        run_dir=run_dir,
        wandb_config=metadata,
        num_sanity_val_steps=30,
        batch_size=128,
        val_check_interval=20,
        max_epochs=3,
        **kwargs,
    )
    return metadata


@terra.Task.make_task
def train_slices(
    slices_dp: mk.DataPanel,
    split_run_id: int,
    num_samples: int = 1,
    **kwargs,
):
    def _train_model(config):
        dp = build_correlation_slice(**config, split_run_id=split_run_id)
        return train_model(
            dp=dp,
            config=config,
            pbar=False,
            **kwargs,
        )

    ray.init(num_gpus=1, num_cpus=6)
    analysis = tune.run(
        _train_model,
        config=tune.grid_search(list(slices_dp)),
        num_samples=num_samples,
        resources_per_trial={"gpu": 1},
    )
    return analysis.dataframe()

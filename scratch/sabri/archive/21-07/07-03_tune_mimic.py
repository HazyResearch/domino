import meerkat as mk
import ray
import terra
from meerkat.contrib.mimic import GCSImageColumn
from ray import tune

from domino.data.mimic import CHEXPERT_COLUMNS, build_dp
from domino.vision import train


@terra.Task.make_task
def train_mimic(
    dp: mk.DataPanel,
    input_column: str,
    target_column: str,
    lr: float = 1e-4,
    arch: list = "resnet18",
    run_dir: str = None,
    **kwargs,
):

    train(
        dp=dp,
        input_column=input_column,
        id_column="dicom_id",
        target_column=target_column,
        gpus=1,
        batch_size=32,
        num_workers=6,
        run_dir=run_dir,
        valid_split="validate",
        val_check_interval=20,
        weighted_sampling=True,
        samples_per_epoch=50000,
        config={"lr": lr, "arch": arch},
        **kwargs,
    )


@terra.Task.make_task
def tune_mimic(
    dp_run_id: int,
    run_dir: str = None,
    **kwargs,
):
    def _train_model(config):
        from meerkat.contrib.mimic import GCSImageColumn

        return train_mimic(
            terra.out(dp_run_id), wandb_config=config, **config, **kwargs
        )

    ray.init(num_gpus=4, num_cpus=32)
    analysis = tune.run(
        _train_model,
        config={
            "target_column": tune.grid_search(CHEXPERT_COLUMNS),
            "input_column": tune.grid_search(["input_512"]),
            "lr": tune.qloguniform(1e-5, 1e-2, 5e-6),
            "arch": tune.choice(["resnet18", "resnet50"]),
        },
        num_samples=10,
        resources_per_trial={"gpu": 1, "cpu": 8},
    )
    return analysis.dataframe()


if __name__ == "__main__":
    tune_mimic(dp_run_id=1063, max_epochs=1)

import pandas as pd
import terra

from domino.data.celeb import build_celeb_df, build_celeb_dp
from domino.vision import train


@terra.Task.make_task
def train_celeba(
    df: pd.DataFrame,
    target_column: str,
    input_column: str,
    gdro: bool = False,
    run_dir: str = None,
):

    dp = build_celeb_dp(df)
    loss_config = {
        "gdro": gdro,
        "alpha": 0.2,
        "gamma": 0.1,
        "min_var_weight": 0,
        "robust_step_size": 0.1,
        "use_normalized_loss": False,
        "btl": False,
    }
    config = {
        "lr": 1e-5,
        "model_name": "resnet",
        "arch": "resnet50",
        "loss_config": loss_config,
    }
    subgroup_columns = ["blond_hair"]

    train(
        dp=dp,
        input_column=input_column,
        id_column="file",
        target_column=target_column,
        batch_size=32,  # default: 128
        num_workers=4,
        run_dir=run_dir,
        valid_split="test",
        val_check_interval=20,
        config=config,
        subgroup_columns=subgroup_columns,
    )


if __name__ == "__main__":

    train_celeba(
        df=build_celeb_df(dataset_dir="/media/4tb_hdd/celeba"),
        target_column="male",
        input_column="input",
        gdro=True,
    )

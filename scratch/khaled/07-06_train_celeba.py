import pandas as pd

from domino.data.celeb import build_celeb_df, build_celeb_dp
from domino.vision import train


def train_celeba(
    df: pd.DataFrame,
    target_column: str,
    input_column: str,
    gdro: bool = False,
):

    dp = build_celeb_dp(df)
    config = {}
    config["model"] = {"model_name": "resnet", "arch": "resnet50"}
    config["train"] = {"lr": 1e-5, "wd": 0.1}
    loss_config = {
        "gdro": gdro,
        "alpha": 0.01,
        "gamma": 0.1,
        "min_var_weight": 0,
        "robust_step_size": 0.01,
        "use_normalized_loss": False,
        "btl": False,
    }
    config["train"]["loss"] = loss_config
    config["dataset"] = {"subgroup_columns": ["male"]}
    config["wandb"] = {"project": "domino", "group": "celeba_repro"}

    train(
        dp=dp,
        input_column=input_column,
        id_column="file",
        target_column=target_column,
        batch_size=128,  # default: 128
        num_workers=4,
        valid_split="test",
        val_check_interval=20,
        config=config,
    )


if __name__ == "__main__":

    train_celeba(
        df=build_celeb_df(dataset_dir="/media/4tb_hdd/celeba"),
        target_column="blond_hair",
        input_column="input",
        gdro=True,
    )

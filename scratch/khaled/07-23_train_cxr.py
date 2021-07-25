import meerkat as mk
import terra

from domino.vision import train


@terra.Task.make_task
def train_cxr(
    dp: mk.DataPanel,
    target_column: str,
    input_column: str,
    gdro: bool = False,
    run_dir: str = None,
):

    loss_config = {
        "gdro": gdro,
        "alpha": 0.01,
        "gamma": 0.1,
        "min_var_weight": 0,
        "robust_step_size": 0.1,
        "use_normalized_loss": False,
        "btl": False,
    }
    config = {
        "lr": 1e-5,
        "wd": 0.1,
        "model_name": "resnet",
        "arch": "resnet50",
        "loss_config": loss_config,
    }
    subgroup_columns = ["tube"]  # ["gazeslicer_time"]

    train(
        dp=dp,
        input_column=input_column,
        id_column="image_id",
        target_column=target_column,
        batch_size=128,
        num_workers=4,
        run_dir=run_dir,
        valid_split="test",
        val_check_interval=20,
        config=config,
        subgroup_columns=subgroup_columns,
    )


if __name__ == "__main__":

    # dp = mk.DataPanel.read(path="/media/4tb_hdd/siim/gazeslicer_dp_07-23-21.dp")
    dp = mk.DataPanel.read(path="/media/4tb_hdd/siim/tubescribble_dp_07-24-21.dp")
    train_cxr(
        dp=dp,
        target_column="pmx",
        input_column="input",
        gdro=True,
    )

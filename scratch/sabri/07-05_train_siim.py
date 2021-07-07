import meerkat as mk
import pandas as pd
import terra

from domino.data.cxr import build_cxr_df, get_dp
from domino.vision import train


@terra.Task.make_task
def train_siim(
    df: pd.DataFrame, target_column: str, input_column: str, run_dir: str = None
):
    dp = get_dp(df)

    train(
        dp=dp,
        input_column=input_column,
        id_column="image_id",
        target_column=target_column,
        gpus=1,
        batch_size=24,
        num_workers=12,
        run_dir=run_dir,
        valid_split="test",
        val_check_interval=20,
        config={"lr": 1e-4, "model_name": "resnet", "arch": "resnet50"},
    )


if __name__ == "__main__":

    train_siim(
        df=build_cxr_df.out(),
        target_column="pmx",
        input_column="input",
    )

import pandas as pd
import terra

from domino.data.celeb import build_celeb_df, get_celeb_dp
from domino.vision import train


@terra.Task.make_task
def train_celeb(df: pd.DataFrame, run_dir: str = None):

    dp = get_celeb_dp(df)
    train(
        dp=dp,
        input_column="input",
        id_column="file",
        target_column="bald",
        batch_size=128,
        num_workers=4,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    train_celeb(build_celeb_df.out(474))

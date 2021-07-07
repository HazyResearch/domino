import numpy as np
import pandas as pd
import terra

from domino.data.celeb import build_celeb_df, get_celeb_dp
from domino.evaluate.linear import induce_correlation
from domino.vision import train


@terra.Task.make_task
def train_celeb(df: pd.DataFrame, target: str, correlate: str, run_dir: str = None):

    dp = get_celeb_dp(df)
    indices = induce_correlation(
        dp, corr=0.85, attr_a=target, attr_b=correlate, n=30000, match_mu=True
    )
    train(
        dp=dp.lz[indices],
        input_column="input",
        id_column="file",
        target_column=target,
        batch_size=128,
        num_workers=4,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    train_model(
        target="high_cheekbones", correlate="wearing_hat", df=build_celeb_df.out(474)
    )

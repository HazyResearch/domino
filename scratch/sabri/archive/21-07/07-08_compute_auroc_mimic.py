import meerkat.contrib.mimic
import numpy as np
import pandas as pd
import terra
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from domino.evaluate.linear import induce_correlation
from domino.evaluate.train import score_linear_slices, score_model


@terra.Task.make_task
def compute_auroc_on_slices(model_df: pd.DataFrame, run_dir: str = None):
    df = []
    for row in tqdm(
        model_df[["run_id", "target", "correlate", "corr", "num_examples"]].to_dict(
            "records"
        )
    ):
        test_dp = score_model.out(row["run_id"])[0].load()

        target, correlate, corr = row["target"], row["correlate"], row["corr"]
        for slyce in ["equal", "not_equal"]:
            if slyce == "equal":
                mask = test_dp[target] == test_dp[correlate]
            else:
                mask = test_dp[target] != test_dp[correlate]
            slice_dp = test_dp.lz[mask]
            df.append(
                {
                    "auroc": roc_auc_score(
                        slice_dp[target], slice_dp["output"].data[:, -1]
                    ),
                    "slice": slyce,
                    "target": target,
                    "correlate": correlate,
                    "train correlation": corr,
                    "Y,S": f"{target},{correlate}",
                }
            )
    return pd.DataFrame(df)


# compute_auroc_on_slices(score_linear_slices.out(1481))

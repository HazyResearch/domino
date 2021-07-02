import numpy as np
import pandas as pd
import terra

from domino.evaluate.train import score_linear_slices

df = terra.out(691)
score_linear_slices(dp_run_id=542, model_df=df, num_workers=6, batch_size=512)


def compute_auroc_on_slices(
    model_df: pd.DataFrame,
):
    model_df = score_linear_slices.out(815, load=True)
    df = []
    for row in tqdm(
        model_df[["run_id", "target", "correlate", "corr", "num_examples"]].to_dict(
            "records"
        )
    ):
        test_dp = score_model.out(row["run_id"])[0].load()

        target, correlate, corr = row["target"], row["correlate"], row["corr"]
        for test_corr in ["train_corr", "uncorrelated"]:
            indices = induce_correlation(
                test_dp,
                attr_a=target,
                attr_b=correlate,
                corr=corr if test_corr == "train_corr" else 0,
                match_mu=True,
                n=3000,
            )
            curr_dp = test_dp.lz[indices]
            df.append(
                {
                    "auroc": roc_auc_score(
                        curr_dp[target], curr_dp["output"].data[:, -1]
                    ),
                    "test correlation": test_corr,
                    "target": target,
                    "correlate": correlate,
                    "train correlation": corr,
                    "Y,S": f"{target},{correlate}",
                }
            )

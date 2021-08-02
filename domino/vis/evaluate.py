import os
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import terra
from tqdm import tqdm


@terra.Task.make_task
def plot_evaluate_sdms(
    df: Union[pd.DataFrame, Sequence[pd.DataFrame]], run_dir: str = None
):
    from sklearn.metrics import roc_auc_score

    from domino.evaluate.evaluate import run_sdm

    print("Preparing dataframe...", flush=True)
    if not isinstance(df, pd.DataFrame):
        df = pd.concat(df)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        dp = run_sdm.out(row.run_id).load()
        for class_idx in np.unique(dp[row.target]):
            curr_dp = dp.lz[dp[row.target] == class_idx]
            if len(np.unique(curr_dp[row.correlate])) <= 1:
                print(row.run_id)
                print(row["corr"])
                continue

            for slice_idx in range(dp["slices"].shape[-1]):
                rows.append(
                    {
                        "sdm_class": row.sdm_class,
                        "target": row.target,
                        "correlate": row.correlate,
                        "corr": row["corr"],
                        "auroc": roc_auc_score(
                            curr_dp[row.correlate], curr_dp["slices"].data[:, slice_idx]
                        ),
                        "class_idx": class_idx,
                        "slice_idx": slice_idx,
                        "run_id": row.run_id,
                    }
                )
    plot_df = pd.DataFrame(rows)
    plot_df["auroc_abs"] = np.maximum(plot_df["auroc"], 1 - plot_df["auroc"])
    plot_df["Y,S"] = plot_df["target"] + "," + plot_df["correlate"]
    # plot
    print("Plotting...")
    sns.set_style("whitegrid", rc={"font_scale": 6})
    g = sns.FacetGrid(
        plot_df, col="Y,S", hue="sdm_class", col_wrap=4, legend_out=True, height=3
    )
    g = g.map(
        sns.lineplot,
        "corr",
        "auroc_abs",
        marker="o",
        estimator="max",
        ci=False,
        label="small",
    )
    g.add_legend(bbox_to_anchor=(0, 0), loc="upper left")
    # plt.savefig(os.path.join(run_dir, "plot.pdf"))

    return plot_df

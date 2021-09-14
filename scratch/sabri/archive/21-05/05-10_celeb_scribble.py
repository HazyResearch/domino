import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import terra
import torch.nn as nn
from meerkat import DataPanel, NumpyArrayColumn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from terra import Task
from terra.io import json_load

from domino.bss_dp import SourceSeparator
from domino.data.celeb import build_celeb_df, get_celeb_dp
from domino.vision import Classifier


def merge_in_feedback(celeb_dp: DataPanel, feedback_dp: DataPanel):
    feedback_files = set(feedback_dp["file"])
    size = feedback_dp["feedback_pos_mask"].shape[1:]
    celeb_dp = celeb_dp.copy()
    if "feedback_label" not in celeb_dp.column_names:
        celeb_dp.add_column(
            "feedback_label", NumpyArrayColumn(["unlabeled"] * len(celeb_dp))
        )
    if "feedback_pos_mask" not in celeb_dp.column_names:
        celeb_dp.add_column(
            "feedback_pos_mask",
            NumpyArrayColumn(np.zeros((len(celeb_dp), *size))),
        )
    if "feedback_neg_mask" not in celeb_dp.column_names:
        celeb_dp.add_column(
            "feedback_neg_mask",
            NumpyArrayColumn(np.zeros((len(celeb_dp), *size))),
        )

    for batch in feedback_dp.batch(batch_size=1, num_workers=0):
        index = np.where(celeb_dp["file"] == batch["file"])
        celeb_dp["feedback_label"][index[0]] = batch["feedback_label"]
        celeb_dp["feedback_neg_mask"][index[0]] = batch["feedback_neg_mask"]
        celeb_dp["feedback_pos_mask"][index[0]] = batch["feedback_pos_mask"]
    return celeb_dp


def pool_feedback(x: DataPanel):
    return {
        f"{feedback_mask}_pool": nn.functional.avg_pool2d(
            input=torch.tensor(x[feedback_mask]).to(float), kernel_size=(32, 32)
        ).numpy()
        for feedback_mask in ["feedback_neg_mask", "feedback_pos_mask"]
    }


@Task
def score_scribbles(
    model_df: pd.DataFrame,
    scribble_run_ids: List[int],
    split: str = "valid",
    threshold: float = 0.2,
    run_dir: str = None,
):
    print("Loading Celeb DataPanel...")
    celeb_dp = get_celeb_dp(build_celeb_df.out(141, load=True))

    # select valid split
    celeb_dp["input"]._materialize = False
    celeb_dp["img"]._materialize = False
    celeb_dp = celeb_dp[np.where(celeb_dp["split"].data == split)[0]]
    celeb_dp["input"]._materialize = True
    celeb_dp["img"]._materialize = True

    results = []
    for _, row in model_df.iterrows():
        row = row.to_dict()
        model = Classifier.__terra_read__(row["model_path"])
        model_target_column = row["target_column"]

        # add activations and predictions
        separator = SourceSeparator(model=model)
        print("Getting activations and predictions...")
        celeb_dp = separator.prepare_dp(celeb_dp, batch_size=16)
        print(
            roc_auc_score(celeb_dp[model_target_column], celeb_dp["probs"].data[:, 1])
        )

        for run_id in scribble_run_ids:
            print("Merging in feedback...")
            scribble_target_column = terra.inp(run_id)["label"]
            feedback_dp = terra.out(run_id=run_id)[0].load()
            dp = merge_in_feedback(celeb_dp, feedback_dp)
            feedback_dp = dp[np.where(dp["feedback_label"].data != "unlabeled")[0]]

            print(f"Pooling masks for {len(feedback_dp)} examples...")
            pooled_masks = {
                f"{feedback_mask}_pool": nn.functional.avg_pool2d(
                    input=feedback_dp[feedback_mask].to_tensor().to(float),
                    kernel_size=(32, 32),
                ).numpy()
                for feedback_mask in ["feedback_neg_mask", "feedback_pos_mask"]
            }
            x = feedback_dp["activation"].transpose(0, 2, 3, 1).reshape(-1, 512)
            y = pooled_masks["feedback_pos_mask_pool"].flatten() > threshold

            print(f"Fitting {x.shape=}, {y.shape=}...")
            lr = LogisticRegression()
            x = normalize(x)
            lr = lr.fit(x, y)

            x_test = celeb_dp["activation"].transpose(0, 2, 3, 1).reshape(-1, 512)
            x_test = normalize(x_test)
            y_test_preds = lr.predict_proba(x_test)
            celeb_dp.add_column(
                "feedback_preds", y_test_preds[:, 1].reshape(-1, 8, 8), overwrite=True
            )

            celeb_dp.add_column(
                "feedback_preds_max",
                celeb_dp["feedback_preds"].max(axis=(1, 2)),
                overwrite=True,
            )
            results.append(
                {
                    "model_target": model_target_column,
                    "model_run_id": row["run_id"],
                    "scribble_run_id": run_id,
                    "scribble_target": scribble_target_column,
                    "auroc": roc_auc_score(
                        celeb_dp[scribble_target_column], celeb_dp["feedback_preds_max"]
                    ),
                }
            )
    return pd.DataFrame(results)


@Task
def plot_score_scribbles(dfs: List[pd.DataFrame], run_dir=None):
    df = pd.concat(dfs)
    plt.figure(figsize=(25, 3))
    pivot_df = df.pivot(columns="model_target", index="scribble_target", values="auroc")
    sns.heatmap(data=pivot_df, cmap="PiYG", annot=True, vmin=0.5, vmax=1)
    plt.savefig(os.path.join(run_dir, "plot.pdf"))
    return pivot_df


if __name__ == "__main__":

    model_df = terra.out(290, load=True)

    score_scribbles(
        model_df=model_df,
        scribble_run_ids=[321, 322, 323],
        split="valid",
    )

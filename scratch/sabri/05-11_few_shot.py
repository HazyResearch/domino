import os
from typing import List

from terra.io import json_load
import terra
from terra import Task
from mosaic import DataPanel, NumpyArrayColumn
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import normalize
import seaborn as sns
import matplotlib.pyplot as plt


from domino.vision import Classifier
from domino.bss_dp import SourceSeparator
from domino.data.celeb import get_celeb_dp, build_celeb_df
from sklearn.metrics import roc_auc_score


@Task.make_task
def score_few_shot(
    model_df: pd.DataFrame,
    targets: List[str],
    cache_activations_run_id: int,
    split: str = "valid",
    run_dir: str = None,
):

    results = []
    for _, row in model_df.iterrows():
        row = row.to_dict()
        act_target = row["target_column"]

        print(f"Loading artifacts for model_target={act_target}...")
        dp = terra.get_artifacts(
            run_id=cache_activations_run_id, group_name=f"{act_target}_activations"
        )["dp"].load()

        for num_examples in [10, 20, 40, 80, 160]:
            for target_column in targets:

                pos_indices = np.random.choice(
                    np.where(dp[target_column] == 1)[0], num_examples // 2
                )
                neg_indices = np.random.choice(
                    np.where(dp[target_column] == 0)[0], num_examples // 2
                )
                indices = np.concatenate([pos_indices, neg_indices])
                x = dp["activation"][indices].mean(axis=(2, 3))
                y = dp[target_column][indices]

                print(
                    f"Fitting logistic regression for few_shot_target={target_column}"
                )
                print(f"num_examples={num_examples}")
                lr = LogisticRegression()
                x = normalize(x)
                lr = lr.fit(x, y)

                x_test = dp["activation"].mean(axis=(2, 3))
                x_test = normalize(x_test)
                y_test_preds = lr.predict_proba(x_test)
                dp.add_column("lr_preds", y_test_preds[:, 1], overwrite=True)
                # TODO: remove training examples
                results.append(
                    {
                        "num_examples": num_examples,
                        "model_target": act_target,
                        "model_run_id": row["run_id"],
                        "few_shot_target": target_column,
                        "auroc": roc_auc_score(dp[target_column], dp["lr_preds"]),
                    }
                )
    return pd.DataFrame(results)


@Task.make_task
def plot_score_few_shot(df: pd.DataFrame, run_dir=None):
    plt.figure(figsize=(25, 3))
    pivot_df = df.pivot(columns="model_target", index="few_shot_target", values="auroc")
    sns.heatmap(data=pivot_df, cmap="PiYG", annot=True, vmin=0.5, vmax=1)
    plt.savefig(os.path.join(run_dir, "plot.pdf"))
    return pivot_df


@Task.make_task
def plot_diff(
    few_dfs: List[pd.DataFrame], scribble_dfs: List[pd.DataFrame], run_dir=None
):
    plt.figure(figsize=(40, 4))
    few_df = pd.concat(few_dfs)
    plot_few_df = few_df.pivot(
        columns="model_target", index="few_shot_target", values="auroc"
    )

    scribble_df = pd.concat(scribble_dfs)
    plot_scribble_df = scribble_df.pivot(
        columns="model_target", index="scribble_target", values="auroc"
    )

    diff_df = plot_scribble_df - plot_few_df
    sns.heatmap(data=diff_df, cmap="PiYG", annot=True, vmin=-0.6, vmax=0.6)


if __name__ == "__main__":

    model_df = terra.out(290, load=True)

    score_few_shot(
        model_df=model_df,
        targets=[
            "wearing_necktie",
            "eyeglasses",
            "blond_hair",
            "wearing_earrings",
            "wearing_lipstick",
            "no_beard",
            "smiling",
        ],
        cache_activations_run_id=327,
        split="valid",
    )

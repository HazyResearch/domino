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
from domino.feedback import ScribbleModel, merge_in_feedback
from domino.vision import Classifier


@Task
def score_few_shot(
    model_df: pd.DataFrame,
    scribble_run_ids: List[int],
    cache_activations_run_id: int,
    strat: str = "scribble",
    threshold: float = 0.2,
    run_dir: str = None,
):

    results = []
    for idx, (_, row) in enumerate(model_df.iterrows()):
        row = row.to_dict()
        act_target = row["target_column"]

        print(f"({idx}) Loading activations for model_target={act_target}...")
        dp = terra.get_artifacts(
            run_id=cache_activations_run_id, group_name=f"{act_target}_activations"
        )["dp"].load()

        for run_id in scribble_run_ids:
            target_column = terra.inp(run_id)["label"]
            feedback_dp = terra.out(run_id=run_id)[0].load()
            dp = merge_in_feedback(dp, feedback_dp, remove=True)
            feedback_dp = dp[np.where(dp["feedback_label"].data != "unlabeled")[0]]
            test_dp = dp["activation", target_column, "feedback_label"][
                np.where(dp["feedback_label"].data == "unlabeled")[0]
            ]

            model = ScribbleModel(strategy=strat, threshold=0.1)
            model.fit(feedback_dp)
            lr_preds = model.predict(test_dp)
            test_dp.add_column("lr_preds", lr_preds, overwrite=True)

            results.append(
                {
                    "model_target": act_target,
                    "model_run_id": row["run_id"],
                    "lr_target": target_column,
                    "auroc": roc_auc_score(test_dp[target_column], test_dp["lr_preds"]),
                }
            )
    return pd.DataFrame(results)


@Task
def plot_score_few_shot(df: pd.DataFrame, run_dir=None):
    plt.figure(figsize=(25, 3))
    pivot_df = df.pivot(columns="model_target", index="lr_target", values="auroc")
    sns.heatmap(data=pivot_df, cmap="PiYG", annot=True, vmin=0.5, vmax=1)
    plt.savefig(os.path.join(run_dir, "plot.pdf"))
    return pivot_df


@Task
def plot_score_diff(
    example_df: pd.DataFrame,
    scribble_df: pd.DataFrame,
    size=(25, 4),
    window=0.3,
    marg_size=2,
    run_dir=None,
):
    plot_scribble_df = scribble_df.pivot(
        columns="model_target", index="lr_target", values="auroc"
    )
    plot_example_df = example_df.pivot(
        columns="model_target", index="lr_target", values="auroc"
    )
    diff_df = plot_scribble_df - plot_example_df

    width, height = size
    f = plt.figure(figsize=(width + marg_size, height + marg_size))
    gs = plt.GridSpec(height + marg_size, width + marg_size)

    ax_marg_x = f.add_subplot(gs[:marg_size, :-marg_size])
    ax_marg_y = f.add_subplot(gs[marg_size:, -marg_size:])
    ax_joint = f.add_subplot(
        gs[marg_size:, :-marg_size], sharex=ax_marg_x, sharey=ax_marg_y
    )

    marg_y = diff_df.mean(axis=1)
    sns.scatterplot(
        x=marg_y,
        y=np.arange(len(marg_y)) + 0.5,
        hue=marg_y,
        palette="PiYG",
        hue_norm=(-window, window),
        linewidth=1,
        legend=False,
        edgecolor="k",
        s=100,
        ax=ax_marg_y,
    )
    ax_marg_y.axvline(0, ls="--")
    ax_marg_y.yaxis.grid(True)
    ax_marg_y.set_xlim(-window, window)
    ax_marg_y.set_xlabel("Mean $\Delta$ in AUROC")
    ax_marg_y.set_ylabel("")

    marg_x = diff_df.mean(axis=0)
    sns.scatterplot(
        y=marg_x,
        x=np.arange(len(marg_x)) + 0.5,
        hue=marg_x,
        palette="PiYG",
        hue_norm=(-window, window),
        linewidth=1,
        edgecolor="k",
        s=100,
        legend=False,
        ax=ax_marg_x,
    )
    ax_marg_x.xaxis.grid(True)
    ax_marg_x.axhline(0, ls="--")
    ax_marg_x.set_ylim(-window, window)
    ax_marg_x.set_ylabel("Mean $\Delta$ in AUROC")
    ax_marg_x.set_xlabel("")

    # important that this is plotted last
    sns.heatmap(
        data=diff_df,
        cmap="PiYG",
        annot=True,
        vmin=-window,
        vmax=window,
        ax=ax_joint,
        cbar=False,
    )

    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_x.get_xticklabels(minor=True), visible=True)
    plt.setp(ax_marg_y.get_yticklabels(minor=True), visible=False)
    plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=True)

    sns.despine(f)

    f.tight_layout()
    f.subplots_adjust(hspace=0.4, wspace=0.2)


if __name__ == "__main__":

    model_df = terra.out(290, load=True)

    score_few_shot(
        model_df=model_df,
        scribble_run_ids=[270, 282, 286, 319, 321, 322, 323],
        cache_activations_run_id=327,
        strat="mask_pos_v_neg",
    )

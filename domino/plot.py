import os
from re import I
from typing import List, Mapping

import matplotlib.pyplot as plt
import meerkat as mk
import pandas as pd
import seaborn as sns
import terra

from domino.evaluate import run_sdm, run_sdms, score_sdm_explanations, score_sdms

# PALETTE = ["#9CBDE8", "#53B7AE", "#EFAB79", "#E27E51", "#19416E", "#1B6C7B"]
PALETTE = ["#9CBDE8", "#316FAE", "#29B2A1", "#007C6E", "#FFA17A", "#A4588F"]


def coherence_metric(grouped_df):
    return (grouped_df["auroc"] > 0.85) & (grouped_df["precision_at_10"] > 0.5)


EMB_PALETTE = {
    "random": "#9CBDE8",
    "imagenet": "#9CBDE8",
    "bit": "#1B6C7B",
    "clip": "#E27E51",
    "mimic_multimodal": "#EFAB79",
    "convirt": "#E27E51",
    "activations": PALETTE[5],
    "eeg": "#1B6C7B",
    "multimodal": "#E27E51",
}


def generate_group_df(score_sdms_id: int):
    setting_dp = score_sdms.inp(score_sdms_id)["setting_dp"].load()
    score_df = score_sdms.out(score_sdms_id).load()
    # if a method returns nans, we assign a score of 0
    score_df = score_df.fillna(0)
    score_dp = mk.DataPanel.from_pandas(score_df)

    spec_columns = [
        col
        for col in ["emb_group", "alpha", "sdm_class", "slice_category"]
        if col not in score_dp
    ]
    results_dp = mk.merge(
        score_dp,
        setting_dp[spec_columns + ["run_sdm_run_id"]],
        on="run_sdm_run_id",
    )

    results_df = results_dp.to_pandas()
    grouped_df = results_df.iloc[
        results_df.reset_index()
        .groupby(["slice_name", "slice_idx", "sdm_class", "alpha", "emb_group",])[
            # .groupby(["slice_idx", "sdm_class", "alpha", "emb_group", "run_sdm_run_id"])[
            "precision_at_10"
        ]
        .idxmax()
        .astype(int)
    ]
    grouped_df["alpha"] = grouped_df["alpha"].round(3)

    # we want to exclude correlation slices with alpha=0
    grouped_df = grouped_df[
        (grouped_df["slice_category"] != "correlation") | (grouped_df["alpha"] != 0)
    ]
    grouped_df["success"] = coherence_metric(grouped_df)

    # hard coded exclusions
    if score_sdms_id == 77006:
        grouped_df = grouped_df[grouped_df["slice_category"] != "rare"]

    return grouped_df


@terra.Task
def sdm_barplot(
    score_sdm_ids: List[int],
    emb_groups: List[str] = None,
    sdm_classes: List[str] = None,
    filter: callable = None,
    run_dir: str = None,
):

    # formatting
    sns.set_style("whitegrid")
    plt.tight_layout()
    plt.figure(figsize=(5, 3))

    # preparing dataframe
    df = pd.concat(
        [generate_group_df(score_sdms_id=run_id) for run_id in score_sdm_ids]
    )

    if emb_groups is not None:
        df = df[df["emb_group"].isin(emb_groups)]

    if sdm_classes is not None:
        df = df[df["sdm_class"].isin(sdm_classes)]

    if filter is not None:
        df = filter(df)

    # preparing pallette
    pallette = {
        group: color for group, color in EMB_PALETTE.items() if group in emb_groups
    }

    sns.barplot(
        data=df,
        y="precision_at_10",
        x="slice_category",
        order=["rare", "correlation", "noisy_label"],
        hue="emb_group",
        hue_order=pallette.keys(),
        palette=sns.color_palette(pallette.values(), len(pallette)),
    )
    sns.despine()

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.ylim([0, 1])
    plt.savefig(os.path.join(run_dir, "plot.pdf"))
    plt.savefig("figures/sdm_barplot.pdf")


@terra.Task
def sdm_displot(
    score_sdm_ids: List[int],
    emb_groups: List[str] = None,
    sdm_classes: List[str] = None,
    filter: callable = None,
    run_dir: str = None,
):

    # formatting
    sns.set_style("whitegrid")
    plt.tight_layout()
    plt.figure(figsize=(3, 3))

    # preparing dataframe
    df = pd.concat(
        [generate_group_df(score_sdms_id=run_id) for run_id in score_sdm_ids]
    )

    if emb_groups is not None:
        df = df[df["emb_group"].isin(emb_groups)]

    if sdm_classes is not None:
        df = df[df["sdm_class"].isin(sdm_classes)]

    if filter is not None:
        df = filter(df)

    # preparing pallette
    pallette = {
        group: color for group, color in EMB_PALETTE.items() if group in emb_groups
    }

    plt.figure(figsize=(2, 20))
    plt.tight_layout()
    sns.displot(
        data=df,
        x="precision_at_10",
        hue="emb_group",
        multiple="dodge",
        bins=5,
        shrink=0.8,
        hue_order=pallette.keys(),
        palette=sns.color_palette(pallette.values(), len(pallette)),
        height=3,
    )
    sns.despine()
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(os.path.join(run_dir, "plot.pdf"))
    plt.savefig("figures/sdm_displot.pdf")

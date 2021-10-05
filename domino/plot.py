import os
from re import I
from typing import List, Mapping

import matplotlib.pyplot as plt
import meerkat as mk
import numpy as np
import pandas as pd
import seaborn as sns
import terra

from domino.evaluate import run_sdm, run_sdms, score_sdm_explanations, score_sdms
from domino.train import score_model

# PALETTE = ["#9CBDE8", "#53B7AE", "#EFAB79", "#E27E51", "#19416E", "#1B6C7B"]
PALETTE = ["#9CBDE8", "#316FAE", "#29B2A1", "#007C6E", "#FFA17A", "#A4588F"]


def coherence_metric(grouped_df):
    return (grouped_df["auroc"] > 0.85) & (grouped_df["precision_at_10"] > 0.5)


EMB_PALETTE = {
    "random": "#9CBDE8",
    "imagenet": "#9CBDE8",
    "bit": "#1B6C7B",
    "activations": "#19416E",
    "clip": "#E27E51",
    "mimic_multimodal": "#EFAB79",
    "convirt": "#E27E51",
    "activations": PALETTE[5],
    "eeg": "#1B6C7B",
    "multimodal": "#E27E51",
}

SDM_PALETTE = {
    "domino.sdm.confusion.ConfusionSDM": "#9CBDE8",
    "domino.sdm.george.GeorgeSDM": "#19416E",
    "domino.sdm.multiaccuracy.MultiaccuracySDM": "#53B7AE",
    "domino.sdm.spotlight.SpotlightSDM": "#1B6C7B",
    "domino.sdm.gmm.MixtureModelSDM": "#E27E51",
}


def _is_degraded(row: dict, threshold: float = 0):
    metrics = terra.io.load_nested_artifacts(
        score_model.out(row["score_model_run_id"])[1]
    )
    try:
        return (
            metrics["out_slice_accuracy_lower"]
            > metrics["in_slice_0_accuracy_upper"] + threshold
        )
    except KeyError:
        pass

    if row["slice_category"] == "rare" or row["slice_category"] == "noisy_label":
        return (
            metrics["out_slice_recall_lower"]
            > metrics["in_slice_0_recall_upper"] + threshold
        )
    elif row["slice_category"] == "correlation":
        return True


def generate_group_df(
    score_sdms_id: int,
    metric: str = "precision_at_10",
    degraded_threshold: float = None,
):
    setting_dp = score_sdms.inp(score_sdms_id)["setting_dp"].load()
    score_df = score_sdms.out(score_sdms_id).load()
    # if a method returns nans, we assign a score of 0
    score_df = score_df.fillna(0)
    score_dp = mk.DataPanel.from_pandas(score_df)

    spec_columns = [
        col
        for col in ["emb_group", "alpha", "sdm_class", "slice_category", "setting_id"]
        if col not in score_dp
    ]
    results_dp = mk.merge(
        score_dp,
        setting_dp[spec_columns + ["run_sdm_run_id"]],
        on="run_sdm_run_id",
    )

    results_df = results_dp.to_pandas()

    # activations are stored as 0, need to map to "activation"
    results_df["emb_group"] = results_df["emb_group"].map(
        lambda x: "activations" if ((x == 0) or (x is None)) else x,
    )

    grouped_df = results_df.iloc[
        results_df.reset_index()
        .groupby(["slice_name", "slice_idx", "sdm_class", "alpha", "emb_group"])[
            # .groupby(["slice_idx", "sdm_class", "alpha", "emb_group", "run_sdm_run_id"])[
            metric
        ]
        .idxmax()
        .astype(int)
    ].copy()
    grouped_df["alpha"] = grouped_df["alpha"].round(3)

    # we want to exclude correlation slices with alpha=0
    grouped_df = grouped_df[
        (grouped_df["slice_category"] != "correlation") | (grouped_df["alpha"] != 0)
    ]
    grouped_df["success"] = coherence_metric(grouped_df)

    # hard coded exclusions
    if score_sdms_id == 77006:
        grouped_df = grouped_df[grouped_df["slice_category"] != "rare"]

    if degraded_threshold is not None and score_sdms_id not in [99336, 99862, 102466]:
        grouped_df = grouped_df[
            grouped_df.apply(
                lambda x: _is_degraded(x, threshold=degraded_threshold), axis=1
            )
        ]

    return grouped_df


@terra.Task
def sdm_barplot(
    score_sdm_ids: List[int],
    hue: str = "emb_group",
    emb_groups: List[str] = None,
    sdm_classes: List[str] = None,
    run_dir: str = None,
    path: str = None,
    **kwargs,
):

    # formatting
    sns.set_style("whitegrid")
    plt.figure(figsize=(5, 3))

    # preparing dataframe
    df = pd.concat(
        [generate_group_df(score_sdms_id=run_id, **kwargs) for run_id in score_sdm_ids]
    )
    if emb_groups is not None:
        df = df[df["emb_group"].isin(emb_groups)]

    if sdm_classes is not None:
        df = df[df["sdm_class"].isin(sdm_classes)]

    # preparing pallette
    pallette = (
        {group: color for group, color in EMB_PALETTE.items() if group in emb_groups}
        if hue == "emb_group"
        else {
            sdm_class: color
            for sdm_class, color in SDM_PALETTE.items()
            if sdm_class in sdm_classes
        }
    )
    sns.barplot(
        data=df,
        y="precision_at_10",
        x="slice_category",
        order=["rare", "correlation", "noisy_label"],
        hue=hue,
        hue_order=pallette.keys(),
        palette=sns.color_palette(pallette.values(), len(pallette)),
    )
    sns.despine()
    plt.legend([], [], frameon=False)
    plt.ylim([0, 1])
    plt.savefig(os.path.join(run_dir, "plot.pdf"))
    if path is not None:
        plt.savefig(path)
    return df


@terra.Task
def sdm_displot(
    score_sdm_ids: List[int],
    hue: str = "emb_group",
    emb_groups: List[str] = None,
    sdm_classes: List[str] = None,
    filter: callable = None,
    run_dir: str = None,
    path: str = None,
    **kwargs,
):

    # formatting
    sns.set_style("whitegrid")

    # preparing dataframe
    df = pd.concat(
        [
            generate_group_df(
                score_sdms_id=run_id,
                **kwargs,
            )
            for run_id in score_sdm_ids
        ]
    )

    if emb_groups is not None:
        df = df[df["emb_group"].isin(emb_groups)]

    if sdm_classes is not None:
        df = df[df["sdm_class"].isin(sdm_classes)]

    if filter is not None:
        df = filter(df)

    # preparing pallette
    pallette = (
        {group: color for group, color in EMB_PALETTE.items() if group in emb_groups}
        if hue == "emb_group"
        else {
            sdm_class: color
            for sdm_class, color in SDM_PALETTE.items()
            if sdm_class in sdm_classes
        }
    )
    sns.displot(
        data=df,
        x="precision_at_10",
        hue=hue,
        multiple="dodge",
        bins=5,
        shrink=0.8,
        hue_order=pallette.keys(),
        palette=sns.color_palette(pallette.values(), len(pallette)),
        height=3,
    )
    plt.legend([], [], frameon=False)
    sns.despine()
    plt.savefig(os.path.join(run_dir, "plot.pdf"))
    if path is not None:
        plt.savefig(path)
    return df


def generate_expl_group_df(score_sdm_expl_id: int, metric: str = "precision_at_10"):
    setting_dp = score_sdm_explanations.inp(score_sdm_expl_id)["setting_dp"].load()
    score_df = score_sdm_explanations.out(score_sdm_expl_id).load()
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

    # activations are stored as 0, need to map to "activation"
    results_df["emb_group"] = results_df["emb_group"].map(
        lambda x: "activations" if x == 0 else x, na_action="ignore"
    )

    grouped_df = results_df.iloc[
        results_df.reset_index()
        .groupby(["slice_name", "slice_idx", "sdm_class", "alpha", "emb_group",])[
            # .groupby(["slice_idx", "sdm_class", "alpha", "emb_group", "run_sdm_run_id"])[
            metric
        ]
        .idxmax()
        .astype(int)
    ]
    grouped_df["alpha"] = grouped_df["alpha"].round(3)

    # we want to exclude correlation slices with alpha=0
    grouped_df = grouped_df[
        (grouped_df["slice_category"] != "correlation") | (grouped_df["alpha"] != 0)
    ]

    # hard coded exclusions
    if score_sdm_expl_id == 122560:
        grouped_df = grouped_df[grouped_df["slice_category"] != "rare"]

    return grouped_df


def expl_plot(
    score_sdm_explanation_ids: List[int],
    emb_groups: str = ["clip", "bit", "random"],
    run_dir: str = None,
):
    rows = []
    for run_id in score_sdm_explanation_ids:
        df = generate_expl_group_df(run_id, metric="max_reciprocal_rank")
        for emb_group in emb_groups:
            df = df[df["emb_group"] == emb_group]

            for slice_category in df["slice_category"].unique():
                curr_df = df[df["slice_category"] == slice_category]
                hist = np.histogram(
                    curr_df["min_rank"],
                    bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, curr_df["min_rank"].max()],
                )
                fracs = np.cumsum(hist[0][:-1]) / len(curr_df)
                bins = hist[1][:-2]
                rows.extend(
                    [
                        {
                            "frac": frac,
                            "min_rank": fr"$\leq{bin_end}$",
                            "emb_group": emb_group,
                            "slice_category": slice_category,
                        }
                        for frac, bin_end in zip(fracs, bins)
                    ]
                )
    plot_df = pd.DataFrame(rows)
    # plot_df = plot_df[plot_df["slice_category"] == "rare"]
    pallette = {
        group: color for group, color in EMB_PALETTE.items() if group in emb_groups
    }
    plt.figure(figsize=(4, 4))
    sns.pointplot(
        data=plot_df,
        x="min_rank",
        y="frac",
        hue="emb_group",
        hue_order=pallette.keys(),
        palette=sns.color_palette(pallette.values(), len(pallette)),
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    sns.despine()
    plt.ylim(0, 0.7)
    plt.savefig("figures/pointplot.pdf")
    return plot_df

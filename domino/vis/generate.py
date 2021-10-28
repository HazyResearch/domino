import os
from typing import List

import pandas as pd
import terra
from matplotlib.collections import PathCollection
from tqdm import tqdm

from domino.plot import sdm_barplot, sdm_displot

FIGURES = [
    # natural settings
    {  # check
        "comparison": "method",
        "model": "real",
        "domain": "natural",
        "score_sdm_run_ids": [102466, 142612],
    },
    {  # check
        "comparison": "embed",
        "model": "real",
        "domain": "natural",
        "score_sdm_run_ids": [136271, 99336, 99862],
    },
    {  # check
        "comparison": "method",
        "model": "synthetic",
        "domain": "natural",
        "score_sdm_run_ids": [90363, 94854],
    },
    {  # check
        "comparison": "embed",
        "model": "synthetic",
        "domain": "natural",
        "score_sdm_run_ids": [82222, 77006, 87801, 89044],
    },
    # mimic settings
    {
        "comparison": "method",
        "model": "real",
        "domain": "mimic",
        "score_sdm_run_ids": [155866, 162811, 170018],
        "kwargs": dict(
            emb_groups=["convirt", "mimic_multimodal"],
            degraded_threshold=0.1,
        ),
    },
    {
        "comparison": "embed",
        "model": "real",
        "domain": "mimic",
        "score_sdm_run_ids": [110256, 83004, 60240, 168595],
    },
    {
        "comparison": "method",
        "model": "synthetic",
        "domain": "mimic",
        "score_sdm_run_ids": [152130, 119202, 117095],
    },
    {
        "comparison": "embed",
        "model": "synthetic",
        "domain": "mimic",
        "score_sdm_run_ids": [45159, 79801, 58676, 77753, 64524, 77296],
    },
]


EMBS = {
    "mimic": {
        "core": ["convirt", "mimic_multimodal"],
        "all": ["bit", "imagenet", "convirt", "mimic_multimodal"],
    },
    "natural": {"core": ["clip"], "all": ["bit", "clip", "random"]},
    "eeg": {"core": ["multimodal"], "all": ["multimodal", "eeg"]},
}


@terra.Task
def generate_results(figures: List = FIGURES, run_dir: str = None):
    dfs = []
    for figure in tqdm(figures):
        # real model embed comparison natural images

        emb_groups = EMBS[figure["domain"]][
            "all" if figure["comparison"] == "embed" else "core"
        ].copy()

        if figure["model"] != "synthetic" and figure["comparison"] == "embed":
            emb_groups.append("activations")

        default_kwargs = dict(
            hue="emb_group" if figure["comparison"] != "method" else "sdm_class",
            degraded_threshold=0.1 if figure["model"] == "real" else None,
            emb_groups=emb_groups,
            sdm_classes=[
                "domino.sdm.gmm.MixtureModelSDM",
                "domino.sdm.george.GeorgeSDM",
                "domino.sdm.multiaccuracy.MultiaccuracySDM",
                "domino.sdm.spotlight.SpotlightSDM",
                "domino.sdm.confusion.ConfusionSDM",
            ]
            if figure["comparison"] == "method"
            else ["domino.sdm.gmm.MixtureModelSDM"],
        )

        if "kwargs" in figure:
            default_kwargs.update(figure["kwargs"])
        df = sdm_barplot(
            score_sdm_ids=figure["score_sdm_run_ids"],
            path=os.path.join(
                run_dir,
                f"comparison={figure['comparison']}-model={figure['model']}-domain={figure['domain']}-barplot.pdf",
            ),
            **default_kwargs,
        ).load()
        for k, v in figure.items():
            if k == "kwargs" or k == "score_sdm_run_ids":
                continue
            df[k] = v
        dfs.append(df)

        sdm_displot(
            score_sdm_ids=figure["score_sdm_run_ids"],
            path=os.path.join(
                run_dir,
                f"comparison={figure['comparison']}-model={figure['model']}-domain={figure['domain']}-displot.pdf",
            ),
            **default_kwargs,
        )
    return pd.concat(dfs)

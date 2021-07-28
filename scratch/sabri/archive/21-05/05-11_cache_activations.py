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


@Task.make_task
def cache_activations(
    model_df: pd.DataFrame, splits: List[str] = None, run_dir: str = None
):
    if splits is None:
        splits = ["valid", "test"]
    print("Loading Celeb DataPanel...")
    from domino.data.celeb import build_celeb_df, get_celeb_dp

    celeb_df = build_celeb_df.out(450, load=True)
    celeb_dp = get_celeb_dp(celeb_df[celeb_df["split"].isin(splits)])

    results = []
    for idx, (_, row) in enumerate(model_df.iterrows()):
        row = row.to_dict()
        model = Classifier.__terra_read__(row["model_path"])
        model_target_column = row["target_column"]

        # add activations and predictions
        separator = SourceSeparator(model=model)
        print(
            f"({idx}) Getting activations and predictions for {model_target_column}..."
        )
        celeb_dp = separator.prepare_dp(
            celeb_dp,
            batch_size=128,
            layers={
                "block2": model.model.layer2,
                "block3": model.model.layer3,
                "block4": model.model.layer4,
            },
        )

        Task.dump(
            {
                "model_run_id": row["run_id"],
                "dp": celeb_dp,
            },
            run_dir=run_dir,
            group_name=f"{model_target_column}_activations",
        )


if __name__ == "__main__":

    model_df = terra.out(290, load=True)

    cache_activations(model_df=model_df, splits=["valid", "test"])

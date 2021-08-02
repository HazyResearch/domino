import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import terra
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from domino.evaluate.evaluate import run_sdm


@terra.Task.make_task
def compute_sdm_metrics(evaluate_df: pd.DataFrame, run_dir: str = None):
    def compute_metrics(row: pd.Series, slice_idx: int = 0):
        dp = run_sdm.out(row.run_id, load=True)
        dp[f"slice_{slice_idx}"] = dp["slices"].data[:, slice_idx]
        df = dp.to_pandas()

        # can't compute metrics if there is no variance in the slice
        if len(df[f"slice_{slice_idx}"].unique()) <= 1:
            return None

        metrics = {
            "slice_idx": slice_idx,
            "corr_s": pg.corr(x=df[row.correlate], y=df[f"slice_{slice_idx}"]).r[0],
            "partial_corr_s": pg.partial_corr(
                data=df, x=row.correlate, y=f"slice_{slice_idx}", covar=[row.target]
            ).r[0],
            "auroc": roc_auc_score(df[row.correlate], df[f"slice_{slice_idx}"]),
            **{
                f"auroc_{y=}": roc_auc_score(
                    df[df[row.target] == y][row.correlate],
                    df[df[row.target] == y][f"slice_{slice_idx}"],
                )
                for y in df[row.target].unique()
            },
        }
        return pd.concat(
            [
                row[["run_id", "corr", "target", "correlate", "sdm_class"]],
                pd.Series(metrics),
            ]
        )

    metrics = []
    for slice_idx in tqdm(range(5)):
        for _, row in evaluate_df.iterrows():
            out = compute_metrics(row, slice_idx=slice_idx)
            if out is not None:
                metrics.append(out)
    return pd.DataFrame(metrics)

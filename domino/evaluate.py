import os
from typing import Mapping, Union

import meerkat as mk
import pandas as pd
import ray
import terra
import torch.nn as nn
from ray import tune
from ray.tune.utils.placement_groups import PlacementGroupFactory
from tqdm import tqdm

from domino.metrics import compute_sdm_metrics
from domino.sdm.abstract import SliceDiscoveryMethod
from domino.sdm.george import GeorgeSDM
from domino.slices.gqa import build_correlation_slice, build_rare_slice, build_slice
from domino.train import score_model, train_model


@terra.Task
def run_sdm(
    model: nn.Module,
    data_dp: mk.DataPanel,
    emb_dp: mk.DataPanel,
    sdm_class: type,
    sdm_config: SliceDiscoveryMethod.Config,
    **kwargs,
):
    data_dp = data_dp.lz[data_dp["split"].isin(["valid", "test"])].merge(
        emb_dp[["object_id", "emb"]], on="object_id"
    )

    print("Creating slice discovery method...")
    sdm: SliceDiscoveryMethod = sdm_class(sdm_config)
    print("Fitting slice discovery method...")
    sdm.fit(
        data_dp=data_dp.lz[data_dp["split"] == "valid"],
        model=model,
    )
    print("Transforming slice discovery method...")
    slice_dp = sdm.transform(data_dp=data_dp.lz[data_dp["split"] == "test"])
    return slice_dp


@terra.Task.make(no_load_args={"emb_dp"})
def evaluate_sdms(
    sdm_config: dict,
    slices_dp: mk.DataPanel,
    emb_dp: mk.DataPanel = None,
    resources_per_trial: Union[
        None, Mapping[str, Union[float, int, Mapping]], PlacementGroupFactory
    ] = None,
    run_dir: str = None,
):
    def _evaluate(config):

        if config["slice"]["synthetic_preds"]:
            dp = build_slice.out(config["slice"]["build_run_id"])
            model = None
        else:
            dp = score_model.out(config["slice"]["score_run_id"])
            model = score_model.inp(config["slice"]["score_run_id"])["model"]

        run_id, _ = run_sdm(
            data_dp=dp,
            emb_dp=emb_dp,
            model=model,
            **config["slice"],
            **config["sdm"],
            return_run_id=True,
        )
        # need to return metadata to tune so we get it in the analysis dp
        return {"run_sdm_run_id": run_id, **config["slice"], **config["sdm"]}

    analysis = tune.run(
        _evaluate,
        config={
            "slice": tune.grid_search(list(slices_dp)),
            "sdm": sdm_config,
        },
        resources_per_trial=tune.sample_from(
            lambda spec: spec.config.sdm.get("resources_per_trial", resources_per_trial)
        ),
        raise_on_failed_trial=False,  # still want to return dataframe even if some trials fails
        max_failures=3,  # retrying when there is a OOM error is a reasonable strategy
        local_dir=run_dir,
    )
    return mk.DataPanel.from_pandas(analysis.dataframe())


@terra.Task
def score_sdms(evaluate_dp: mk.DataPanel):
    dfs = []
    for row in tqdm(evaluate_dp):
        dp = run_sdm.out(run_id=row["run_sdm_run_id"], load=True)
        metrics_df = compute_sdm_metrics(dp)
        metrics_df["target"] = row["target_name"]
        metrics_df["slice"] = row["name"]
        metrics_df["run_sdm_run_id"] = row["run_sdm_run_id"]
        dfs.append(metrics_df)
    return pd.concat(dfs, axis=0)


# @terra.Task
# def compute_sdm_metrics(evaluate_df: pd.DataFrame, run_dir: str = None):
#     def compute_metrics(row: pd.Series, slice_idx: int = 0):
#         dp = run_sdm.out(row.run_id, load=True)
#         dp[f"slice_{slice_idx}"] = dp["slices"].data[:, slice_idx]
#         df = dp.to_pandas()

#         # can't compute metrics if there is no variance in the slice
#         if len(df[f"slice_{slice_idx}"].unique()) <= 1:
#             return None

#         metrics = {
#             "slice_idx": slice_idx,
#             "corr_s": pg.corr(x=df[row.correlate], y=df[f"slice_{slice_idx}"]).r[0],
#             "partial_corr_s": pg.partial_corr(
#                 data=df, x=row.correlate, y=f"slice_{slice_idx}", covar=[row.target]
#             ).r[0],
#             "auroc": roc_auc_score(df[row.correlate], df[f"slice_{slice_idx}"]),
#             **{
#                 f"auroc_{y=}": roc_auc_score(
#                     df[df[row.target] == y][row.correlate],
#                     df[df[row.target] == y][f"slice_{slice_idx}"],
#                 )
#                 for y in df[row.target].unique()
#             },
#         }
#         return pd.concat(
#             [
#                 row[["run_id", "corr", "target", "correlate", "sdm_class"]],
#                 pd.Series(metrics),
#             ]
#         )

#     metrics = []
#     for slice_idx in tqdm(range(5)):
#         for _, row in evaluate_df.iterrows():
#             out = compute_metrics(row, slice_idx=slice_idx)
#             if out is not None:
#                 metrics.append(out)
#     return pd.DataFrame(metrics)

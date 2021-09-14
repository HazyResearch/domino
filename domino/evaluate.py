from typing import Dict, Mapping, Sequence, Tuple, Union

import meerkat as mk
import pandas as pd
import terra
import torch.nn as nn
from meerkat.datapanel import DataPanel
from ray import tune
from ray.tune.utils.placement_groups import PlacementGroupFactory
from tqdm import tqdm

from domino.metrics import compute_sdm_metrics
from domino.sdm.abstract import SliceDiscoveryMethod
from domino.slices.gqa import build_slice
from domino.train import score_model


@terra.Task
def run_sdm(
    model: nn.Module,
    data_dp: mk.DataPanel,
    emb_dp: Union[mk.DataPanel, Mapping[str, mk.DataPanel]],
    sdm_class: type,
    sdm_config: SliceDiscoveryMethod.Config,
    **kwargs,
):
    print("Creating slice discovery method...")
    sdm: SliceDiscoveryMethod = sdm_class(sdm_config)

    print("Loading embeddings...")
    data_dp = data_dp.lz[data_dp["split"].isin(["valid", "test"])].merge(
        emb_dp[["object_id", sdm.config.emb]], on="object_id"
    )

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
    emb_dp: Union[mk.DataPanel, Dict[str, mk.DataPanel]] = None,
    run_dir: str = None,
):
    def _evaluate(config):

        if config["slice"]["synthetic_preds"]:
            dp = build_slice.out(config["slice"]["build_run_id"])
            model = None
        else:
            dp = score_model.out(config["slice"]["score_run_id"])
            model = score_model.inp(config["slice"]["score_run_id"])["model"]

        if isinstance(emb_dp, Mapping):
            emb_tuple = config["sdm"]["sdm_config"]["emb"]
            if not isinstance(emb_tuple, Tuple):
                raise ValueError(
                    "Must 'emb' in the sdm config must be a tuple when "
                    "providing multiple `emb_dp`."
                )
            _emb_dp = emb_dp[emb_tuple[0]]
            config["sdm"]["sdm_config"]["emb"] = emb_tuple[1]
        else:
            _emb_dp = emb_dp

        run_id, _ = run_sdm(
            data_dp=dp,
            emb_dp=_emb_dp,
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
            lambda spec: spec.config.sdm["sdm_class"].RESOURCES_REQUIRED
        ),
        raise_on_failed_trial=False,  # still want to return dataframe even if some trials fails
        local_dir=run_dir,
    )

    result_dp = mk.DataPanel.from_pandas(analysis.dataframe())
    result_dp["sdm_class"] = (
        result_dp["sdm_class"].str.extract(r"'(.*)'", expand=False).data
    )
    return result_dp


@terra.Task
def score_sdms(evaluate_dp: mk.DataPanel, spec_columns: Sequence[str] = None):
    cols = ["target_name", "name", "run_sdm_run_id"]
    if spec_columns is not None:
        cols += spec_columns
    dfs = []
    for row in tqdm(evaluate_dp):
        dp = run_sdm.out(run_id=row["run_sdm_run_id"], load=True)
        metrics_df = compute_sdm_metrics(dp)
        for col in cols:
            metrics_df[col] = row[col]
        dfs.append(metrics_df)
    return pd.concat(dfs, axis=0)

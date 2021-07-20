import os
from typing import Mapping, Union

import meerkat as mk
import pandas as pd
import ray
import terra
import torch.nn as nn
from ray import tune
from ray.tune.utils.placement_groups import PlacementGroupFactory

from domino.evaluate.train import score_model
from domino.sdm.abstract import SliceDiscoveryMethod
from domino.sdm.george import GeorgeSDM


@terra.Task.make_task
def run_sdm(
    model: nn.Module,
    data_dp: mk.DataPanel,
    target: str,
    correlate: str,
    sdm_class: type,
    sdm_config: SliceDiscoveryMethod.Config,
    id_column: str,
    corr: float = None,
    **kwargs
):
    aliases = {"target": target, "correlate": correlate, "act": "activation_layer4"}
    print("Creating slice discovery method...")
    sdm: SliceDiscoveryMethod = sdm_class(sdm_config)
    print("Fitting slice discovery method...")
    sdm.fit(data_dp=data_dp, model=model, aliases=aliases)
    print("Transforming slice discovery method...")
    slice_dp = sdm.transform(data_dp=data_dp, aliases=aliases)
    return slice_dp[[target, correlate, id_column, "slices"]]


@terra.Task.make_task
def evaluate_sdms(
    acts_df: pd.DataFrame,
    sdm_config: dict,
    id_column: str,
    resources_per_trial: Union[
        None, Mapping[str, Union[float, int, Mapping]], PlacementGroupFactory
    ] = None,
    run_dir: str = None,
):
    def _evaluate(config):
        import meerkat.contrib.mimic  # required otherwise we get a yaml import error

        score_run_id = config["triple"].pop("run_id")
        model = score_model.inp(score_run_id)["model"]
        data_dp = score_model.out(score_run_id)[0]
        run_id, _ = run_sdm(
            data_dp=data_dp,
            model=model,
            id_column=id_column,
            **config["triple"],
            **config["sdm"],
            return_run_id=True,
        )
        # need to return metadata to tune so we get it in the analysis dp
        return {"run_id": run_id, **config["triple"], **config["sdm"]}

    analysis = tune.run(
        _evaluate,
        config={
            "triple": tune.grid_search(
                acts_df[["run_id", "target", "correlate", "corr"]].to_dict("records")
            ),
            "sdm": sdm_config,
        },
        resources_per_trial=tune.sample_from(
            lambda spec: spec.config.sdm.get("resources_per_trial", resources_per_trial)
        ),
        raise_on_failed_trial=False,  # still want to return dataframe even if some trials fails
        max_failures=3,  # retrying when there is a OOM error is a reasonable strategy
        local_dir=run_dir,
    )
    return analysis.dataframe()

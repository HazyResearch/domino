import os
from functools import partial
from typing import Collection, List, Mapping, Sequence, Union

import meerkat as mk
import numpy as np
import pandas as pd
import ray
import terra
import torch
from ray import tune
from ray.tune.progress_reporter import CLIReporter
from torch import nn
from tqdm.auto import tqdm

from domino.metrics import compute_model_metrics
from domino.slices.abstract import build_setting
from domino.utils import get_wandb_runs, get_worker_assignment, nested_getattr
from domino.vision import Classifier, score, train


@terra.Task
def train_model(
    dp: mk.DataPanel,
    setting_spec: dict,
    model_config: dict,
    run_dir: str = None,
    **kwargs,
):
    return train(
        dp=dp,
        input_column="input",
        id_column="id",
        target_column="target",
        run_dir=run_dir,
        wandb_config={
            "train_model_run_id": int(os.path.basename(run_dir)),
            **setting_spec,
        },
        config=model_config,
        **kwargs,
    )


@terra.Task.make(no_load_args={"data_dp", "split_dp"})
def train_settings(
    setting_dp: mk.DataPanel,
    data_dp: mk.DataPanel,
    split_dp: mk.DataPanel,
    model_config: dict,
    num_samples: int = 1,
    num_gpus: int = 1,
    num_cpus: int = 8,
    continue_run_ids: List[int] = None,
    run_dir: str = None,
    **kwargs,
):
    train_settings_run_id = int(os.path.basename(run_dir))

    # support for splitting up the job among multiple worker machines
    setting_dp = get_worker_assignment(
        dp=setting_dp,
        worker_idx=kwargs.pop("worker_idx", None),
        num_workers=kwargs.pop("num_workers", None),
    )

    def _train_model(setting_spec):
        import terra

        build_setting_run_id, dp = build_setting(
            data_dp=data_dp,
            split_dp=split_dp,
            dataset=setting_spec["dataset"],
            slice_category=setting_spec["slice_category"],
            build_setting_kwargs=setting_spec["build_setting_kwargs"],
            return_run_id=True,
        )
        train_model_run_id, _ = train_model(
            dp=dp,
            setting_spec={
                **setting_spec,
                "train_settings_run_id": train_settings_run_id,
                "build_setting_run_id": build_setting_run_id,
            },
            model_config=model_config,
            pbar=False,
            **kwargs,
            return_run_id=True,
        )

        result = {
            "setting_id": setting_spec["setting_id"],
            "train_settings_run_id": train_settings_run_id,
            "train_model_run_id": train_model_run_id,
            "build_setting_run_id": build_setting_run_id,
        }
        return result

    if continue_run_ids is not None:
        # support for restarting failed runs
        df = get_wandb_runs()
        finished_df = df[
            df["train_settings_run_id"].isin(continue_run_ids)
            & (df["state"] == "finished")
        ]
        setting_to_run_dp = setting_dp.lz[
            ~setting_dp["setting_id"].isin(finished_df["setting_id"])
        ]
    else:
        setting_to_run_dp = setting_dp
        finished_df = None

    analysis = tune.run(
        _train_model,
        config=tune.grid_search(list(setting_to_run_dp)),
        num_samples=num_samples,
        resources_per_trial={"gpu": 1},
        verbose=1,
        local_dir=run_dir,
    )
    cols = [
        "setting_id",
        # "build_setting_run_id",
        "train_model_run_id",
        "train_settings_run_id",
    ]
    analysis_df = analysis.dataframe()[cols]
    if finished_df is not None:
        analysis_df = pd.concat([analysis_df, finished_df[cols]])

    return (
        mk.merge(
            setting_dp,
            mk.DataPanel.from_pandas(analysis_df),
            on="setting_id",
        ),
        analysis.dataframe(),
    )


@terra.Task
def score_model(
    dp: mk.DataPanel,
    model: Classifier,
    split: Union[str, Collection[str]],
    layers: Union[nn.Module, Mapping[str, nn.Module]] = None,
    reduction_fns: Sequence[str] = None,
    run_dir: str = None,
    **kwargs,
):

    if layers is not None:
        layers = {name: nested_getattr(model, layer) for name, layer in layers.items()}

    if reduction_fns is not None:
        # get the actual function corresponding to the str passed in
        def _get_reduction_fn(reduction_name):
            if reduction_name == "max":
                reduction_fn = partial(torch.mean, dim=[-1, -2])
            elif reduction_name == "mean":
                reduction_fn = partial(torch.mean, dim=[-1, -2])
            else:
                raise ValueError(f"reduction_fn {reduction_name} not supported.")
            reduction_fn.__name__ = reduction_name
            return reduction_fn

        reduction_fns = list(map(_get_reduction_fn, reduction_fns))

    split_mask = (
        (dp["split"].data == split)
        if isinstance(split, str)
        else np.isin(dp["split"].data, split)
    )
    score_dp = score(
        model,
        dp=dp.lz[split_mask],
        input_column="input",
        run_dir=run_dir,
        layers=layers,
        reduction_fns=reduction_fns,
        **kwargs,
    )
    metrics = compute_model_metrics(score_dp, num_iter=1000, flat=True)
    return score_dp, metrics


@terra.Task
def score_settings(
    model_dp: pd.DataFrame,
    split: Union[str, Collection[str]] = "test",
    layers: Union[nn.Module, Mapping[str, str]] = None,
    reduction_fns: Sequence[str] = None,
    num_gpus: int = 1,
    num_cpus: int = 8,
    run_dir: str = None,
    **kwargs,
):
    def _score_model(config):
        run_id = config.pop("train_model_run_id")
        score_run_id, score_dp = score_model(
            model=train_model.get(run_id, "best_chkpt")["model"],
            dp=train_model.inp(run_id)["dp"],
            split=split,
            layers=layers,
            pbar=True,
            reduction_fns=reduction_fns,
            return_run_id=True,
            **kwargs,
        )
        return {
            "train_model_run_id": run_id,
            "score_settings_run_id": int(os.path.basename(run_dir)),
            "score_model_run_id": score_run_id,
            "synthetic_preds": False,
        }

    analysis = tune.run(
        _score_model,
        config=tune.grid_search(list(model_dp)),
        resources_per_trial={"gpu": 1},
        raise_on_failed_trial=False,  # still want to return dataframe even if some trials fails
        local_dir=run_dir,
    )
    cols = [
        "train_model_run_id",
        "score_settings_run_id",
        "score_model_run_id",
        "synthetic_preds",
    ]
    return (
        mk.merge(
            model_dp,
            mk.DataPanel.from_pandas(analysis.dataframe()[cols]),
            on="train_model_run_id",
        ),
        analysis.dataframe(),
    )


@terra.Task.make(no_load_args={"data_dp", "split_dp"})
def synthetic_score_settings(
    setting_dp: mk.DataPanel,
    data_dp: mk.DataPanel,
    split_dp: mk.DataPanel,
    synthetic_kwargs: Mapping[str, object] = None,
    run_dir: str = None,
    **kwargs,
):
    rows = []
    for setting_spec in tqdm(setting_dp):
        run_id, _ = build_setting(
            data_dp=data_dp,
            split_dp=split_dp,
            return_run_id=True,
            synthetic_preds=True,
            synthetic_kwargs=synthetic_kwargs,
            dataset=setting_spec["dataset"],
            slice_category=setting_spec["slice_category"],
            build_setting_kwargs=setting_spec["build_setting_kwargs"],
            **kwargs,
        )
        rows.append(
            {
                "synthetic_preds": True,
                "build_setting_run_id": run_id,
                "score_model_run_id": run_id,
                "score_settings_run_id": int(os.path.basename(run_dir)),
                "setting_id": setting_spec["setting_id"],
            }
        )
    return mk.merge(mk.DataPanel(rows), setting_dp, on="setting_id")

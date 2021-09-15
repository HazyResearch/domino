import os
from functools import partial
from typing import Collection, Mapping, Sequence, Union

import meerkat as mk
import numpy as np
import pandas as pd
import ray
import terra
import torch
from ray import tune
from torch import nn
from tqdm.auto import tqdm

from domino.metrics import compute_model_metrics
from domino.slices.abstract import build_setting
from domino.utils import nested_getattr, requires_columns
from domino.vision import Classifier, score, train


@terra.Task
def train_model(
    dp: mk.DataPanel,
    setting_config: dict,
    run_dir: str = None,
    **kwargs,
):
    # set seed
    metadata = setting_config
    metadata["run_id"] = int(os.path.basename(run_dir))
    train(
        dp=dp,
        input_column="input",
        id_column="id",
        target_column="target",
        run_dir=run_dir,
        wandb_config=metadata,
        config=kwargs["train_config"],
        batch_size=kwargs["batch_size"],
        ckpt_monitor=kwargs["ckpt_monitor"],
        max_epochs=kwargs["max_epochs"],
        drop_last=kwargs["drop_last"],
    )
    return metadata


@terra.Task.make(no_load_args={"data_dp", "split_dp"})
def train_slices(
    slices_dp: mk.DataPanel,
    data_dp: mk.DataPanel,
    split_dp: mk.DataPanel,
    num_samples: int = 1,
    run_dir: str = None,
    **kwargs,
):
    def _train_model(setting_config):
        dp = build_setting(data_dp=data_dp, split_dp=split_dp, **setting_config)
        setting_config["parent_run_id"] = int(os.path.basename(run_dir))
        return train_model(
            dp=dp,
            setting_config=setting_config,
            pbar=False,
            **kwargs,
        )

    ray.init(num_gpus=1, num_cpus=6)
    analysis = tune.run(
        _train_model,
        config=tune.grid_search(list(slices_dp)),
        num_samples=num_samples,
        resources_per_trial={"gpu": 1},
    )
    return analysis.dataframe()


@terra.Task
def score_model(
    dp: mk.DataPanel,
    model: Classifier,
    split: str,
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

    return score_dp


@terra.Task
def score_slices(
    model_df: pd.DataFrame,
    split: Union[str, Collection[str]] = "test",
    layers: Union[nn.Module, Mapping[str, str]] = None,
    reduction_fns: Sequence[str] = None,
    slice_col: str = "slices",
    num_gpus: int = 1,
    num_cpus: int = 8,
    run_dir: str = None,
    **kwargs,
):
    def _score_model(config):
        run_id = config.pop("run_id")
        score_run_id, score_dp = score_model(
            model=train_model.get(run_id, "best_chkpt")["model"],
            dp=train_model.inp(run_id)["dp"],
            split=split,
            layers=layers,
            pbar=False,
            reduction_fns=reduction_fns,
            return_run_id=True,
            **kwargs,
        )
        return {
            "synthetic_preds": False,
            "train_run_id": run_id,
            "parent_run_id": int(os.path.basename(run_dir)),
            "score_run_id": score_run_id,
            **compute_model_metrics(
                score_dp.load(), num_iter=1000, flat=True, aliases={"slices": slice_col}
            ),
            **config,
        }

    ray.init(num_gpus=num_gpus, num_cpus=num_cpus)
    analysis = tune.run(
        _score_model,
        config=tune.grid_search(model_df[list(model_df)].to_dict("records")),
        resources_per_trial={"gpu": 1},
    )
    return mk.DataPanel.from_pandas(analysis.dataframe())


@terra.Task.make(no_load_args={"data_dp", "split_dp"})
def synthetic_score_slices(
    slices_dp: mk.DataPanel,
    data_dp: mk.DataPanel,
    split_dp: mk.DataPanel,
    synthetic_kwargs: Mapping[str, object] = None,
    run_dir: str = None,
    **kwargs,
):
    rows = []
    for config in tqdm(slices_dp):
        run_id, _ = build_setting(
            data_dp=data_dp,
            split_dp=split_dp,
            return_run_id=True,
            synthetic_preds=True,
            synthetic_kwargs=synthetic_kwargs,
            **config,
            **kwargs,
        )
        rows.append(
            {
                "synthetic_preds": True,
                "build_run_id": run_id,
                "parent_run_id": int(os.path.basename(run_dir)),
                **config,
            }
        )
    return mk.DataPanel(rows)

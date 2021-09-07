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
from tqdm import tqdm

from domino.metrics import compute_slice_metrics
from domino.slices.gqa import build_correlation_slice, build_rare_slice, build_slice
from domino.utils import nested_getattr
from domino.vision import Classifier, score, train


@terra.Task
def train_model(
    dp: mk.DataPanel,
    config: dict,
    run_dir: str = None,
    **kwargs,
):
    # set seed
    metadata = config
    metadata["run_id"] = int(os.path.basename(run_dir))

    train(
        dp=dp,
        config={"pretrained": False},
        input_column="input",
        id_column="id",
        target_column="target",
        run_dir=run_dir,
        wandb_config=metadata,
        num_sanity_val_steps=30,
        batch_size=128,
        val_check_interval=20,
        max_epochs=6,
        **kwargs,
    )
    return metadata


@terra.Task
def train_slices(
    slices_dp: mk.DataPanel,
    split_dp: mk.DataPanel,
    num_samples: int = 1,
    run_dir: str = None,
    **kwargs,
):
    def _train_model(config):
        dp = build_slice(split_dp=split_dp, **config)
        config["parent_run_id"] = int(os.path.basename(run_dir))
        return train_model(
            dp=dp,
            config=config,
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
    num_gpus: int = 1,
    num_cpus: int = 8,
    run_dir: str = None,
    **kwargs,
):
    def _score_model(config):
        run_id = config.pop("run_id")
        score_run_id, score_dp = score_model(
            model=train_model.get_artifacts("best_chkpt", run_id)["model"],
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
            **compute_slice_metrics(score_dp.load(), num_iter=1000, flat=True),
            **config,
        }

    ray.init(num_gpus=num_gpus, num_cpus=num_cpus)
    analysis = tune.run(
        _score_model,
        config=tune.grid_search(
            model_df[
                ["run_id", "target_name", "name", "slice_frac", "target_frac"]
            ].to_dict("records")
        ),
        resources_per_trial={"gpu": 1},
    )
    return mk.DataPanel.from_pandas(analysis.dataframe())


@terra.Task.make(no_load_args={"split_dp"})
def synthetic_score_slices(
    slices_dp: mk.DataPanel, split_dp: mk.DataPanel, run_dir: str = None, **kwargs
):
    rows = []
    for config in tqdm(slices_dp):
        run_id, _ = build_slice(
            split_dp=split_dp,
            return_run_id=True,
            synthetic_preds=True,
            **config,
            **kwargs,
        )
        config["run_id"] = run_id
        config
        rows.append(
            {
                "synthetic_preds": True,
                "build_run_id": run_id,
                "parent_run_id": int(os.path.basename(run_dir)),
                **config,
            }
        )
    return mk.DataPanel(rows)

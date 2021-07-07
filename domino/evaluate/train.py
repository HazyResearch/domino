import os
from typing import Mapping, Sequence, Tuple, Union

import meerkat as mk
import numpy as np
import pandas as pd
import ray
import terra
import torch.nn as nn
from ray import tune

from domino.evaluate.linear import CorrelationImpossibleError, induce_correlation
from domino.utils import nested_getattr
from domino.vision import Classifier, score, train


@terra.Task.make_task
def train_model(
    dp: mk.DataPanel,
    target_correlate: Tuple[str],
    corr: float,
    num_examples: int,
    input_column: str = "input",
    id_column: str = "file",
    run_dir: str = None,
    **kwargs,
):
    # set seed
    target, correlate = target_correlate

    metadata = {
        "target": target,
        "correlate": correlate,
        "corr": corr,
        "num_examples": num_examples,
        "run_id": int(os.path.basename(run_dir)),
    }
    try:
        indices = induce_correlation(
            dp,
            corr=corr,
            attr_a=target,
            attr_b=correlate,
            n=num_examples,
            match_mu=True,
        )
    except CorrelationImpossibleError as e:
        print(e)
        return

    train(
        dp=dp.lz[indices],
        input_column=input_column,
        id_column=id_column,
        target_column=target,
        run_dir=run_dir,
        wandb_config=metadata,
        **kwargs,
    )

    return metadata


@terra.Task.make_task
def train_linear_slices(
    dp_run_id: int,
    target_correlate_pairs: Sequence[Tuple[str]],
    input_column: str,
    id_column: str,
    max_corr: float = 0.8,
    num_corrs: int = 9,
    num_examples: int = 3e4,
    num_samples: float = 1,
    run_dir: str = None,
    **kwargs,
):
    def _train_model(config):
        import meerkat.contrib.mimic

        return train_model(
            terra.out(dp_run_id),
            input_column=input_column,
            id_column=id_column,
            **config,
            num_examples=num_examples,
            **kwargs,
        )

    ray.init(num_gpus=4, num_cpus=32)
    analysis = tune.run(
        _train_model,
        config={
            "corr": tune.grid_search(list(np.linspace(0, max_corr, num_corrs))),
            "target_correlate": tune.grid_search(list(target_correlate_pairs)),
        },
        num_samples=num_samples,
        resources_per_trial={"gpu": 1},
    )
    return analysis.dataframe()


@terra.Task.make_task
def score_model(
    dp: mk.DataPanel,
    model: Classifier,
    target: str,
    correlate: str,
    corr: float,
    num_examples: int,
    split: str,
    layers: Union[nn.Module, Mapping[str, nn.Module]] = None,
    run_dir: str = None,
    **kwargs,
):
    # set seed
    metadata = {
        "target": target,
        "correlate": correlate,
        "corr": corr,
        "num_examples": num_examples,
        "run_id": int(os.path.basename(run_dir)),
    }
    dp = score(
        model,
        dp=dp.lz[dp["split"].data == split],
        input_column="input",
        id_column="file",
        target_column=target,
        run_dir=run_dir,
        layers={name: nested_getattr(model, layer) for name, layer in layers.items()},
        wandb_config=metadata,
        **kwargs,
    )
    cols = ["file", target, correlate, "output"] + (
        [] if layers is None else [f"activation_{name}" for name in layers.keys()]
    )
    return dp[cols], metadata


@terra.Task.make_task
def score_linear_slices(
    dp_run_id: int,
    model_df: pd.DataFrame,
    num_samples: float = 1,
    split: str = "test",
    layers: Union[nn.Module, Mapping[str, nn.Module]] = None,
    num_gpus: int = 1,
    num_cpus: int = 8,
    run_dir: str = None,
    **kwargs,
):
    def _score_model(config):
        args = config["args"]
        args["model"] = terra.get_artifacts(args.pop("run_id"), "best_chkpt")["model"]
        _, metadata = score_model(
            terra.out(dp_run_id), split=split, layers=layers, **args, **kwargs
        )
        return metadata

    ray.init(num_gpus=num_gpus, num_cpus=num_gpus)
    analysis = tune.run(
        _score_model,
        config={
            "args": tune.grid_search(
                model_df[
                    ["run_id", "target", "correlate", "corr", "num_examples"]
                ].to_dict("records")
            )
        },
        num_samples=num_samples,
        resources_per_trial={"gpu": 1},
    )
    return analysis.dataframe()

from typing import Sequence, Tuple

import meerkat as mk
import numpy as np
import ray
import terra
from ray import tune

from domino.evaluate.linear import CorrelationImpossibleError, induce_correlation
from domino.vision import train


@terra.Task.make_task
def train_model(
    dp: mk.DataPanel,
    target_correlate: Tuple[str],
    corr: float,
    num_examples: int,
    run_dir: str = None,
    **kwargs,
):
    # set seed
    target, correlate = target_correlate
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
        input_column="input",
        id_column="file",
        target_column=target,
        run_dir=run_dir,
        wandb_config={
            "target": target,
            "correlate": correlate,
            "corr": corr,
        },
        **kwargs,
    )


@terra.Task.make_task
def train_linear_slices(
    dp_run_id: int,
    target_correlate_pairs: Sequence[Tuple[str]],
    max_corr: float = 0.8,
    num_corrs: int = 9,
    num_examples: int = 3e4,
    num_samples: float = 1,
    run_dir: str = None,
    **kwargs,
):
    def _train_model(config):
        train_model(terra.out(dp_run_id), **config, num_examples=num_examples, **kwargs)

    ray.init(num_gpus=4, num_cpus=32)
    tune.run(
        _train_model,
        config={
            "corr": tune.grid_search(list(np.linspace(0, max_corr, num_corrs))),
            "target_correlate": tune.grid_search(target_correlate_pairs),
        },
        num_samples=num_samples,
        resources_per_trial={"gpu": 1},
    )

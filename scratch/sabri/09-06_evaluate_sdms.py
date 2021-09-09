import os
from typing import Sequence, Tuple, Type

import numpy as np
import ray
import terra
from ray import tune
from ray.tune.suggest.variant_generator import grid_search

from domino.emb.clip import embed_images

# from domino.emb.imagenet import embed_images
from domino.evaluate import evaluate_sdms
from domino.sdm import (
    ICASDM,
    PCASDM,
    GeorgeSDM,
    KernelPCASDM,
    MixtureModelSDM,
    PredSDM,
    SpotlightSDM,
)
from domino.utils import ConditionalSample

ray.init(num_gpus=1, num_cpus=6, resources={"ram_gb": 32})
evaluate_sdms(
    slices_dp=terra.out(5702).load()[:2],
    emb_dp={
        "clip": terra.out(5145),
        "imagenet": terra.out(5765),
        "bit": terra.out(5796),
    },
    sdm_config={
        "sdm_class": MixtureModelSDM,
        "sdm_config": {
            "n_slices": 25,
            "emb": tune.grid_search(
                [("clip", "emb"), ("imagenet", "emb"), ("bit", "body")]
            ),
        },
    },
)

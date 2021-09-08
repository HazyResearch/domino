import os

import ray
import terra
from ray import tune
from ray.tune.suggest.variant_generator import grid_search

# from domino.emb.clip import embed_images
from domino.emb.imagenet import embed_images
from domino.evaluate import evaluate_sdms
from domino.sdm import ICASDM, PCASDM, GeorgeSDM, KernelPCASDM, PredSDM, SpotlightSDM


def fn(spec):
    """Need this to not be a lambda for pickling!"""
    return spec.config.sdm["sdm_class"].RESOURCES_REQUIRED


ray.init(num_gpus=1, num_cpus=8, resources={"ram_gb": 32})
evaluate_sdms(
    slices_dp=terra.out(5702),
    emb_dp=embed_images.out(5765),
    sdm_config={
        "sdm_class": tune.grid_search([SpotlightSDM]),
        "sdm_config": {"n_slices": 5, "layer": "emb"},
        "resources_per_trial": tune.sample_from(fn),
    },
)

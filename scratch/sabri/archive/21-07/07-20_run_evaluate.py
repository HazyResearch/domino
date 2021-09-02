import os

import ray
import terra
from ray import tune
from ray.tune.suggest.variant_generator import grid_search

from domino.evaluate.evaluate import evaluate_sdms
from domino.evaluate.train import score_linear_slices
from domino.sdm import ICASDM, PCASDM, GeorgeSDM, KernelPCASDM, PredSDM, SupervisedSDM


def fn(spec):
    """Need this to not be a lambda for pickling!"""
    return spec.config.sdm["sdm_class"].RESOURCES_REQUIRED


ray.init(num_cpus=30, resources={"ram_gb": 110})
evaluate_sdms(
    score_linear_slices.out(3190),
    sdm_config={
        "sdm_class": tune.grid_search([SupervisedSDM]),
        "sdm_config": {"n_slices": 5, "layer": "layer4.0_mean"},
        "resources_per_trial": tune.sample_from(fn),
    },
    id_column="dicom_id",
)

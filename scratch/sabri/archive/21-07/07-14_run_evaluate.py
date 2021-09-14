import os

import ray
import terra
from ray import tune
from ray.tune.suggest.variant_generator import grid_search

from domino.evaluate.evaluate import evaluate_sdms
from domino.sdm import ICASDM, PCASDM, GeorgeSDM, KernelPCASDM, PredSDM


def fn(spec):
    """Need this to not be a lambda for pickling!"""
    return spec.config.sdm["sdm_class"].RESOURCES_REQUIRED


ray.init(num_gpus=1, num_cpus=8, resources={"ram_gb": 32})
evaluate_sdms(
    terra.out(1593),
    sdm_config={
        "sdm_class": tune.grid_search([ICASDM]),
        "sdm_config": {"n_slices": 5},
        "resources_per_trial": tune.sample_from(fn),
    },
    id_column="dicom_id",
)

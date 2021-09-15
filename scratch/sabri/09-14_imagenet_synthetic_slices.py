import numpy as np
import ray
from ray import tune

from domino import evaluate
from domino.data.imagenet import get_imagenet_dp
from domino.emb.clip import embed_images
from domino.evaluate import run_sdms, score_sdms
from domino.sdm import MixtureModelSDM
from domino.slices.imagenet import collect_rare_slices
from domino.train import synthetic_score_slices
from domino.utils import split_dp

data_dp = get_imagenet_dp.out(6617)
split = split_dp.out(6478)


if True:

    if True:
        slices_dp = collect_rare_slices(
            data_dp=data_dp,
            num_slices=1,
            min_slice_frac=0.01,
            max_slice_frac=0.01,
        ).load()
    else:
        slices_dp = collect_rare_slices.out(6654).load()

    slices_dp = slices_dp.lz[np.random.choice(len(slices_dp), 5)]
    slices_dp = synthetic_score_slices(
        slices_dp=slices_dp,
        data_dp=data_dp,
        split_dp=split,
        synthetic_kwargs={"sensitivity": 0.8, "slice_sensitivities": 0.5},
    )
else:
    slices_dp = synthetic_score_slices.out(6703)
if False:
    dp = embed_images(
        dp=data_dp,
        split_dp=split,
        splits=["valid", "test"],
        img_column="image",
        num_workers=7,
        mmap=True,
    )
else:
    emb_dp = embed_images.out(6662)

if True:
    ray.init(num_gpus=1, num_cpus=6, resources={"ram_gb": 32})
    slices_dp = run_sdms(
        slices_dp=slices_dp,
        emb_dp={
            "clip": emb_dp,  # terra.out(5145),
            # "imagenet": emb_dp,
            # "bit": terra.out(5796),
        },
        sdm_config={
            "sdm_class": MixtureModelSDM,
            "sdm_config": {
                "n_slices": 25,
                "weight_y_log_likelihood": 10,
                "emb": tune.grid_search([("clip", "emb")]),
            },
        },
    )
else:
    slices_dp = run_sdms.out(6678)

slices_df = score_sdms(slices_dp)

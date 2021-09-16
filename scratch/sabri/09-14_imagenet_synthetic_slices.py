import numpy as np
import ray
from ray import tune

from domino import evaluate
from domino.data.imagenet import get_imagenet_dp
from domino.emb.clip import embed_images, embed_words, pca_embeddings
from domino.evaluate import run_sdms, score_sdms
from domino.sdm import MixtureModelSDM
from domino.slices import collect_settings
from domino.train import synthetic_score_settings
from domino.utils import split_dp

data_dp = get_imagenet_dp.out(6617)
split = split_dp.out(6478)
words_dp = embed_words.out(5143).load()


if True:

    if True:

        setting_dp = collect_settings(
            dataset="imagenet",
            slice_category="rare",
            data_dp=data_dp,
            words_dp=words_dp.lz[: int(10_000)],
            num_slices=1,
            min_slice_frac=0.01,
            max_slice_frac=0.01,
        ).load()
    else:
        setting_dp = collect_settings.out(6654)

    setting_dp = setting_dp.lz[np.random.choice(len(setting_dp), 25)]
    setting_dp = synthetic_score_settings(
        setting_dp=setting_dp,
        data_dp=data_dp,
        split_dp=split,
        synthetic_kwargs={"sensitivity": 0.8, "slice_sensitivities": 0.5},
    )
else:
    setting_dp = synthetic_score_settings.out()

if False:

    emb_dp = embed_images(
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
    setting_dp = run_sdms(
        setting_dp=setting_dp,
        emb_dp={
            "clip": emb_dp,  # terra.out(5145),
            # "imagenet": emb_dp,
            # "bit": terra.out(5796),
        },
        sdm_config={
            "sdm_class": MixtureModelSDM,
            "sdm_config": {
                "n_slices": 5,
                "weight_y_log_likelihood": tune.grid_search([10]),
                "emb": tune.grid_search([("clip", "emb")]),
            },
        },
    )
else:
    setting_dp = run_sdms.out(6678)

slices_df = score_sdms(setting_dp)

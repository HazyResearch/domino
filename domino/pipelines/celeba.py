from typing import List

import numpy as np
import ray
import terra
from ray import tune

from domino.data.celeba import get_celeba_dp
from domino.emb import embed_images
from domino.emb.clip import (
    CELEBA_GENDER_PHRASE_TEMPLATES,
    CELEBA_PHRASE_TEMPLATES,
    embed_phrases,
    generate_phrases,
    get_wiki_words,
)
from domino.evaluate import run_sdms, score_sdm_explanations, score_sdms
from domino.sdm import MixtureModelSDM, SpotlightSDM
from domino.slices import collect_settings
from domino.train import score_settings, synthetic_score_settings, train_settings
from domino.utils import split_dp

NUM_GPUS = 1
NUM_CPUS = 8


data_dp = get_celeba_dp()

split = split_dp(dp=data_dp, split_on="identity")


words_dp = get_wiki_words(top_k=20_000, eng_only=True)
phrase_dp = generate_phrases(
    words_dp=words_dp,
    templates=CELEBA_GENDER_PHRASE_TEMPLATES,
    k=3,
    num_candidates=100_000,
)
words_dp = embed_phrases(words_dp=phrase_dp, top_k=50_000)

embs = {
    "bit": embed_images(
        emb_type="bit",
        dp=data_dp,
        split_dp=split,
        splits=["valid", "test"],
        img_column="image",
        num_workers=7,
        mmap=True,
    ),
    "imagenet": embed_images(
        emb_type="imagenet",
        dp=data_dp,
        split_dp=split,
        layers={"emb": "layer4"},
        splits=["valid", "test"],
        img_column="image",
        num_workers=7,
        mmap=True,
    ),
    "clip": embed_images(
        emb_type="clip",
        dp=data_dp,
        split_dp=split,
        splits=["valid", "test"],
        img_column="image",
        num_workers=7,
        mmap=True,
    ),
}

# words_dp = embed_words.out(5143).load()

setting_dp = collect_settings(
    dataset="celeba",
    slice_category="correlation",
    data_dp=data_dp,
    num_corr=5,
    n=30_000,
)

# setting_dp = setting_dp.load()
# setting_dp = setting_dp.lz[np.random.choice(len(setting_dp), 4)]

if True:
    setting_dp = synthetic_score_settings(
        setting_dp=setting_dp,
        data_dp=data_dp,
        split_dp=split,
        synthetic_kwargs={
            "sensitivity": 0.8,
            "slice_sensitivities": 0.4,
            "specificity": 0.8,
            "slice_specificities": 0.4,
        },
    )
elif False:
    setting_dp = synthetic_score_settings.out()
else:
    setting_dp, _ = train_settings(
        setting_dp=setting_dp,
        data_dp=data_dp,
        split_dp=split,
        model_config={},  # {"pretrained": False},
        batch_size=256,
        val_check_interval=50,
        max_epochs=15,
        ckpt_monitor="valid_auroc",
        num_cpus=NUM_CPUS,
        num_gpus=NUM_GPUS,
    )

    setting_dp, _ = score_settings(
        model_dp=setting_dp,
        layers={"layer4": "model.layer4"},
        batch_size=512,
        reduction_fns=["mean"],
        num_cpus=NUM_CPUS,
        num_gpus=NUM_GPUS,
        split=["test", "valid"],
    )


common_config = {
    "n_slices": 5,
    "emb": tune.grid_search(
        [
            ("imagenet", "emb"),
            ("bit", "body"),
            ("clip", "emb"),
        ]
    ),
    "xmodal_emb": "emb",
}
setting_dp = run_sdms(
    setting_dp=setting_dp,
    emb_dp=embs,
    xmodal_emb_dp=embs["clip"],
    word_dp=words_dp,
    sdm_config=[
        # {
        #     "sdm_class": SpotlightSDM,
        #     "sdm_config": {
        #         "learning_rate": tune.grid_search([1e-2, 1e-3]),
        #         **common_config,
        #     },
        # },
        {
            "sdm_class": MixtureModelSDM,
            "sdm_config": {
                "weight_y_log_likelihood": tune.grid_search([10]),
                **common_config,
            },
        },
    ],
    num_cpus=NUM_CPUS,
    num_gpus=NUM_GPUS,
    skip_terra_cache=False,
)


slices_df = score_sdms(setting_dp=setting_dp, spec_columns=["emb_group", "alpha"])
# slices_df = score_sdm_explanations(setting_dp=setting_dp)

from typing import List

import nltk
import numpy as np
import psutil
import ray
import torch
from ray import tune

from domino.data.celeba import get_celeba_dp
from domino.emb import embed_images
from domino.emb.clip import (
    CELEBA_GENDER_PHRASE_TEMPLATES,
    CELEBA_PHRASE_TEMPLATES,
    embed_phrases,
    embed_words,
    generate_phrases,
    get_wiki_words,
)
from domino.evaluate import run_sdms, score_sdm_explanations, score_sdms
from domino.pipelines.utils import parse_pipeline_args
from domino.sdm import (
    ConfusionSDM,
    GeorgeSDM,
    MixtureModelSDM,
    MultiaccuracySDM,
    SpotlightSDM,
)
from domino.slices import collect_settings
from domino.slices.abstract import concat_settings, random_filter_settings
from domino.train import (
    filter_settings,
    score_settings,
    synthetic_score_settings,
    train_settings,
)
from domino.utils import split_dp

# support for splitting up the job among multiple worker machines
args = parse_pipeline_args()
worker_idx, num_workers = args.worker_idx, args.num_workers
print(f"{worker_idx=}, {num_workers=}")

NUM_GPUS = torch.cuda.device_count()
NUM_CPUS = psutil.cpu_count()
print(f"Found {NUM_GPUS=}, {NUM_CPUS=}")
ray.init(num_gpus=NUM_GPUS, num_cpus=NUM_CPUS, ignore_reinit_error=True)

# simpler to install earlier on to avoid race condition with train
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

data_dp = get_celeba_dp()

split = split_dp(dp=data_dp, split_on="identity")


setting_dp = concat_settings(
    [
        collect_settings(
            dataset="celeba",
            slice_category="correlation",
            data_dp=data_dp,
            min_corr=0.2,
            max_corr=0.8,
            num_corr=4,
            match_mu=True,
            n=30_000,
        ),
    ]
)


if args.sanity:
    # filture
    setting_dp = random_filter_settings(setting_dp, subset_size=64)


if args.synthetic:
    setting_dp = synthetic_score_settings(
        setting_dp=setting_dp,
        data_dp=data_dp,
        split_dp=split,
        synthetic_kwargs={
            "sensitivity": 0.75,
            "slice_sensitivities": 0.4,
            "specificity": 0.75,
            "slice_specificities": 0.4,
        },
    )
else:
    train_settings_kwargs = dict(
        setting_dp=setting_dp,
        data_dp=data_dp,
        split_dp=split,
        # we do not use imagenet pretrained models, since the classification task is
        # a subset of imagenet
        model_config={"pretrained": False},
        batch_size=256,
        # val_check_interval=250,
        check_val_every_n_epoch=2,
        max_epochs=10,
        ckpt_monitor="valid_auroc",
        continue_run_ids=[56437, 56436, 56427, 56418],
    )

    score_settings_kwargs = dict(
        layers={"layer4": "model.layer4"},
        batch_size=512,
        reduction_fns=["mean"],
        split=["test", "valid"],
    )

    if num_workers is not None and worker_idx is None:
        # supported for distributed training
        setting_dp = concat_settings(
            [
                score_settings(
                    model_dp=train_settings(
                        **train_settings_kwargs,
                        worker_idx=worker_idx,
                        num_workers=num_workers,
                    )[0],
                    **score_settings_kwargs,
                )[0]
                for worker_idx in range(num_workers)
            ]
        )
    elif worker_idx is None:
        setting_dp, _ = train_settings(**train_settings_kwargs)
        setting_dp = score_settings(model_dp=setting_dp, **score_settings_kwargs)[0]
    else:
        # setting_dp, _ = train_settings(
        #     **train_settings_kwargs, worker_idx=worker_idx, num_workers=num_workers
        # )
        setting_dp, _ = train_settings.out(
            {2: 69628, 3: 69611, 4: 69606, 1: 69599, 5: 69588, 0: 69581}[worker_idx]
        )
        setting_dp = score_settings(model_dp=setting_dp, **score_settings_kwargs)[0]
    quit()

    setting_dp = filter_settings(setting_dp)
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
    "random": embed_images(
        emb_type="imagenet",
        dp=data_dp,
        split_dp=split,
        layers={"emb": "layer4"},
        splits=["valid", "test"],
        img_column="image",
        num_workers=7,
        mmap=True,
        model="resnet50_random",
    ),
}


common_config = {
    "n_slices": 5,
    "emb": tune.grid_search(
        [
            # ("imagenet", "emb"),
            # ("bit", "body"),
            # ("clip", "emb"),
            ("random", "emb")
            # passing None for emb group tells run_sdms that the embedding is in
            # the score_dp – this for the model embeddings
            # (None, "layer4"),
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
        #         "learning_rate": 1e-3,
        #         **common_config,
        #     },
        # },
        # {
        #     "sdm_class": MultiaccuracySDM,
        #     "sdm_config": {
        #         **common_config,
        #     },
        # },
        # {
        #     "sdm_class": GeorgeSDM,
        #     "sdm_config": {
        #         **common_config,
        #     },
        # },
        {
            "sdm_class": MixtureModelSDM,
            "sdm_config": {
                "weight_y_log_likelihood": 10,
                **common_config,
            },
        },
        # {
        #     "sdm_class": ConfusionSDM,
        #     "sdm_config": {
        #         **common_config,
        #     },
        # },
    ],
    skip_terra_cache=False,
)


slices_df = score_sdms(
    setting_dp=setting_dp,
    spec_columns=["emb_group", "alpha", "sdm_class"],
    skip_terra_cache=True,
)
# slices_df = score_sdm_explanations(setting_dp=setting_dp)

from typing import List

import numpy as np
import ray
import terra
from ray import tune

from domino.data.celeba import get_celeba_dp
from domino.emb.clip import embed_images, embed_words, get_wiki_words
from domino.evaluate import run_sdms, score_sdm_explanations, score_sdms
from domino.sdm import MixtureModelSDM, SpotlightSDM
from domino.slices import collect_settings
from domino.train import score_settings, synthetic_score_settings, train_settings
from domino.utils import split_dp

NUM_GPUS = 1
NUM_CPUS = 8


class Pipeline:
    def __init__(self, to_rerun: List[str]):
        self.reran_tasks = set(to_rerun)

    def run(self, parent_tasks: List[str], task: terra.Task, **kwargs):
        name = task.__name__
        if set(parent_tasks + [name]) & self.reran_tasks:
            self.reran_tasks.add(name)
            return task(**kwargs)
        print(f"Using cached result from  task: {name}, run_id:{task.last_run_id}")
        return task.out()


p = Pipeline(to_rerun=["run_sdms"])

data_dp = p.run(
    parent_tasks=[],
    task=get_celeba_dp,
)

split = p.run(
    parent_tasks=["get_celeba_dp"], task=split_dp, dp=data_dp, split_on="identity"
)

# words_dp = embed_words.out(5143).load()

setting_dp = p.run(
    parent_tasks=["get_celeba_dp"],
    task=collect_settings,
    dataset="celeba",
    slice_category="correlation",
    data_dp=data_dp,
    num_corr=5,
    n=30_000,
)

setting_dp = setting_dp.load()
setting_dp = setting_dp.lz[np.random.choice(len(setting_dp), 10)]

if False:
    setting_dp = p.run(
        parent_tasks=["collect_settings"],
        task=synthetic_score_settings,
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
else:
    setting_dp = p.run(
        parent_tasks=["collect_settings"],
        task=train_settings,
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

    setting_dp, _ = p.run(
        parent_tasks=["train_settings"],
        task=score_settings,
        model_dp=setting_dp,
        layers={"layer4": "model.layer4"},
        batch_size=512,
        reduction_fns=["mean"],
        num_cpus=NUM_CPUS,
        num_gpus=NUM_GPUS,
        split=["test", "valid"],
    )

emb_dp = p.run(
    parent_tasks=["get_celeba_dp", "split_dp"],
    task=embed_images,
    dp=data_dp,
    split_dp=split,
    splits=["valid", "test"],
    img_column="image",
    num_workers=7,
    mmap=True,
)

words_dp = p.run(parent_tasks=[], task=get_wiki_words, top_k=10_000, eng_only=True)
words_dp = p.run(parent_tasks=["get_wiki_words"], task=embed_words, words_dp=words_dp)


common_config = {
    "n_slices": 5,
    "emb": tune.grid_search([("clip", "emb")]),
}
setting_dp = p.run(
    parent_tasks=["embed_images", "synthetic_score_settings", "score_settings"],
    task=run_sdms,
    setting_dp=setting_dp,
    emb_dp={
        "clip": emb_dp,  # terra.out(5145),
        # "imagenet": emb_dp,
        # "bit": terra.out(5796),
    },
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
                "weight_y_log_likelihood": 10,  # tune.grid_search([1, 5, 10, 20]),
                **common_config,
            },
        },
    ],
)


slices_df = p.run(parent_tasks=["run_sdms"], task=score_sdms, setting_dp=setting_dp)
slices_df = p.run(
    parent_tasks=["run_sdms"], task=score_sdm_explanations, setting_dp=setting_dp
)

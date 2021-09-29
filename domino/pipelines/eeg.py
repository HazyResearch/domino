import pdb
from typing import List

import numpy as np
import ray
import terra
from meerkat.contrib.eeg import build_stanford_eeg_dp
from ray import tune

from domino.emb.eeg import embed_eeg, embed_words, generate_words_dp
from domino.evaluate import run_sdms, score_sdm_explanations, score_sdms
from domino.sdm import MixtureModelSDM, SpotlightSDM
from domino.slices import collect_settings
from domino.train import score_settings, synthetic_score_settings, train_settings
from domino.utils import balance_dp, split_dp

NUM_GPUS = 2
NUM_CPUS = 32


class Pipeline:
    def __init__(self, to_rerun: List[str]):
        self.reran_tasks = set(to_rerun)

    def run(
        self,
        parent_tasks: List[str],
        task: terra.Task,
        task_run_id: int = None,
        **kwargs,
    ):
        name = task.__name__
        if set(parent_tasks + [name]) & self.reran_tasks:
            self.reran_tasks.add(name)
            return task(**kwargs)

        if task_run_id:
            print(f"Using cached result from  task: {name}, run_id: {task_run_id}")
            return task.out(task_run_id)

        print(f"Using cached result from  task: {name}, run_id: {task.last_run_id}")
        return task.out()


p = Pipeline(to_rerun=["run_sdms"])

data_dp = p.run(
    parent_tasks=[], task=build_stanford_eeg_dp, task_run_id=925
)  # for 60 sec: 618

data_dp = p.run(
    parent_tasks=["build_stanford_eeg_dp"], task=balance_dp, task_run_id=928
)  # for 60 sec: 623

split = p.run(parent_tasks=["balance_dp"], task=split_dp, task_run_id=625)

setting_dp = p.run(
    parent_tasks=["balance_dp"],
    task=collect_settings,
    dataset="eeg",
    slice_category="correlation",
    data_dp=data_dp,
    correlate_list=["age"],
    correlate_thresholds=[1],
    n=8000,
)

setting_dp = setting_dp.load()

# setting_dp = setting_dp.lz[np.random.choice(len(setting_dp), 8)]

if True:
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
    model_config = {
        "model_name": "dense_inception",
        "data_shape": (2400, 19),
        "train_transform": None,
        "transform": None,
        "lr": 1e-6,
    }
    setting_dp, _ = p.run(
        parent_tasks=["collect_settings"],
        task=train_settings,
        setting_dp=setting_dp,
        data_dp=data_dp,
        split_dp=split,
        model_config=model_config,
        batch_size=16,
        val_check_interval=20,
        max_epochs=15,
        drop_last=True,
        ckpt_monitor="valid_auroc",
        num_cpus=NUM_CPUS,
        num_gpus=NUM_GPUS,
    )

    setting_dp, _ = p.run(
        parent_tasks=["train_settings"],
        task=score_settings,
        model_dp=setting_dp,
        batch_size=16,
        num_cpus=NUM_CPUS,
        num_gpus=NUM_GPUS,
        split=["test", "valid"],
    )


eeg_emb_dp = p.run(
    parent_tasks=["build_stanford_eeg_dp", "balance_dp", "split_dp"],
    task=embed_eeg,
    task_run_id=1656,
    dp=data_dp,
    model_run_id=1583,  # 75 epochs: 837, original 10 epoch run id = 709
    layers={"emb": "model.fc1"},
    split_dp=split,
    splits=["valid", "test"],
    num_workers=7,
    batch_size=10,
    mmap=True,
)

multimodal_emb_dp = p.run(
    parent_tasks=["build_stanford_eeg_dp", "balance_dp", "split_dp"],
    task=embed_eeg,
    task_run_id=1657,
    dp=data_dp,
    model_run_id=1655,  # clip run for 35 epochs: 1655, multimodal run 75 epochs: 843, first multimodal run id = 704
    layers={"emb": "model.fc1"},
    split_dp=split,
    splits=["valid", "test"],
    num_workers=7,
    batch_size=10,
    mmap=True,
)

all_narratives = build_stanford_eeg_dp.out(run_id=696, load=True)["narrative"].data
words_dp = p.run(
    parent_tasks=[],
    task=generate_words_dp,
    task_run_id=838,
    all_reports=all_narratives,
    min_threshold=1,
)
words_dp = p.run(
    parent_tasks=["generate_words_dp"],
    task=embed_words,
    task_run_id=840,
    words_dp=words_dp,
    model_run_id=704,
    batch_size=10,
)


common_config = {
    "n_slices": 10,
    "n_clusters": 10,
    "emb": tune.grid_search([("eeg", "emb"), ("multimodal", "emb")]),
}
setting_dp = p.run(
    parent_tasks=[
        "collect_settings",
        "embed_eeg",
        "synthetic_score_settings",
        "score_settings",
    ],
    task=run_sdms,
    setting_dp=setting_dp,
    emb_dp={
        "eeg": eeg_emb_dp,
        "multimodal": multimodal_emb_dp,
    },
    xmodal_emb_dp=multimodal_emb_dp,
    word_dp=words_dp,
    id_column="id",
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
# slices_df = p.run(
#     parent_tasks=["run_sdms"], task=score_sdm_explanations, setting_dp=setting_dp
# )

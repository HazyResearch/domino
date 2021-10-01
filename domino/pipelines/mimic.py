from typing import List

import meerkat as mk
import meerkat.contrib.mimic.gcs
import numpy as np
import ray
import terra
from ray import tune

from domino.data.mimic import get_mimic_dp, split_dp_preloaded
from domino.emb import embed_images
#from domino.emb.mimic_multimodal import embed_images
from domino.evaluate import run_sdms, score_sdm_explanations, score_sdms
from domino.sdm import MixtureModelSDM, SpotlightSDM
from domino.slices import collect_settings
from domino.train import score_settings, synthetic_score_settings, train_settings
from domino.slices.abstract import concat_settings
from domino.pipelines.utils import parse_pipeline_args

NUM_GPUS = 4
NUM_CPUS = 16

args = parse_pipeline_args()

worker_idx, num_workers = args.worker_idx, args.num_workers
print(f"{worker_idx=}, {num_workers=}")

ray.init(num_gpus=NUM_GPUS, num_cpus=NUM_CPUS)

data_dp = get_mimic_dp(skip_terra_cache=False)

split = split_dp_preloaded(dp=data_dp, skip_terra_cache=False)

##CORRELATION##
setting_dp = collect_settings(
    dataset="mimic",
    slice_category="correlation",
    data_dp=data_dp,
    num_corr=5,
    n=30_000,
    skip_terra_cache=False,
)
'''
##RARE###
setting_dp = collect_settings(
    dataset="mimic",
    slice_category="rare",
    data_dp=data_dp,
    min_slice_frac=0.01,
    max_slice_frac=0.1,
    num_frac = 5,
    n = 30_000,
    skip_terra_cache=False,
)

###NOISY LABEL####
setting_dp = collect_settings(
    dataset="mimic",
    slice_category="noisy_label",
    data_dp=data_dp,
    min_slice_frac=0.01,
    max_slice_frac=0.1,
    num_frac = 5,
    n = 30_000,
    skip_terra_cache=False,
)
'''

###NOISY FEATURE####

#setting_dp = setting_dp.load()
#setting_dp = setting_dp.lz[np.random.choice(len(setting_dp), 3)]

if False:
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
        skip_terra_cache=True,
    )
else:
    #train_settings.out(57076) rare semi-synthetic
    '''
    setting_dp, _ = train_settings(
        setting_dp=setting_dp,
        data_dp=data_dp,
        split_dp=split,
        model_config={},  # {"pretrained": False},
        batch_size=256,
        val_check_interval=50,
        max_epochs=10,
        ckpt_monitor="valid_auroc",
        num_gpus=NUM_GPUS,
        num_cpus=NUM_CPUS,
        skip_terra_cache=False
    )
    '''
    train_settings_kwargs = dict(setting_dp=setting_dp,
        data_dp=data_dp,
        split_dp=split,
        model_config={},  # {"pretrained": False},
        batch_size=256,
        val_check_interval=50,
        max_epochs=10,
        ckpt_monitor="valid_auroc",
        num_gpus=NUM_GPUS,
        num_cpus=NUM_CPUS,
        skip_terra_cache=False)

    score_settings_kwargs = dict(
        layers={"layer4": "model.layer4"},
        batch_size=512,
        reduction_fns=["mean"],
        num_gpus=NUM_GPUS,
        num_cpus=NUM_CPUS,
        split=["test", "valid"],
        skip_terra_cache=False
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
        setting_dp = score_settings(model_dp=setting_dp, **score_settings_kwargs)
    else:
        setting_dp, _ = train_settings.out({2: 70248, 3: 70219, 4: 70182, 1: 70242, 5: 70169, 0: 70212, 6: 70195, 7: 70226}[worker_idx])
        #train_settings(
        #    **train_settings_kwargs, worker_idx=worker_idx, num_workers=num_workers
        #)
        
        setting_dp = score_settings(model_dp=setting_dp, **score_settings_kwargs)
   
    

embs = {
    "bit": embed_images(
        emb_type="bit",
        dp=data_dp,
        split_dp=split,
        splits=["valid", "test"],
        img_column="cxr_jpg_1024",
        num_workers=7,
        mmap=True,
        skip_terra_cache=False,
    ),
    "imagenet": embed_images(
        emb_type="imagenet",
        dp=data_dp,
        split_dp=split,
        layers={"emb": "layer4"},
        splits=["valid", "test"],
        img_column="cxr_jpg_1024",
        num_workers=7,
        mmap=True,
        skip_terra_cache=False,
    ),
    "clip": embed_images(
        emb_type="clip",
        dp=data_dp,
        split_dp=split,
        splits=["valid", "test"],
        img_column="cxr_jpg_1024",
        num_workers=7,
        mmap=True,
        skip_terra_cache=False,
    ),
    "mimic_multimodal": embed_images(
        emb_type="mimic_multimodal",
        dp=data_dp,
        split_dp=split,
        splits=["valid", "test"],
        img_column="cxr_jpg_1024",
        num_workers=7,
        mmap=True,
        skip_terra_cache=False,
        file_path='/pd/maya/rx-multimodal/classifier/checkpoints/0919_clip_vit_findingsimpressions_full/'
    ),
    "mimic_multimodal_class": embed_images(
        emb_type="mimic_multimodal_class",
        dp=data_dp,
        split_dp=split,
        splits=["valid", "test"],
        img_column="cxr_jpg_1024",
        num_workers=7,
        mmap=True,
        skip_terra_cache=False,
        file_path='/pd/maya/rx-multimodal/classifier/checkpoints/0927_clip_vit_findingsimpressions_full_classification/'
    ),
    "mimic_imageonly": embed_images(
        emb_type="mimic_imageonly",
        dp=data_dp,
        split_dp=split,
        splits=["valid", "test"],
        img_column="cxr_jpg_1024",
        num_workers=7,
        mmap=True,
        skip_terra_cache=True,
        file_path='/pd/maya/rx-multimodal/classifier/checkpoints/0926_domino_vit_imageonly/'
    )
}


#words_dp = get_wiki_words(top_k=10_000, eng_only=True, skip_terra_cache=False)
#words_dp = embed_words(words_dp=words_dp, skip_terra_cache=False)
#words_dp = embed_words.out(6537).load()
#words_dp = words_dp.lz[:int(1e4)]


common_config = {
    "n_slices": 5,
    "emb": tune.grid_search(
        [
            ("imagenet", "emb"),
            ("bit", "body"),
            ("clip", "emb"),
            ("mimic_multimodal", "emb"),
            ("mimic_multimodal_class", "emb"),
            ("mimic_imageonly", "emb"),
            (None, "layer4"),
        ]
    ),
    "xmodal_emb": "emb",
}
#print(setting_dp.load().columns)
setting_dp = run_sdms(
    setting_dp=setting_dp,
    id_column="dicom_id",
    emb_dp=embs,
    xmodal_emb_dp=embs['clip'],
    word_dp=None,
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
                "weight_y_log_likelihood": tune.grid_search([1,5,10,20]),
                **common_config,
            },
        },
    ],
    num_cpus=NUM_CPUS,
    num_gpus=NUM_GPUS,
    skip_terra_cache=True,
)


slices_df = score_sdms(setting_dp=setting_dp, skip_terra_cache=True)
#slices_df = score_sdm_explanations(setting_dp=setting_dp, skip_terra_cache=True)

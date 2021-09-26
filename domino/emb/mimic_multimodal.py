import os
from itertools import product
from typing import Iterable, List

import clip
import meerkat as mk
import meerkat.nn as mknn
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from terra import Task
from tqdm.auto import tqdm
import pickle

from domino.utils import batched_pearsonr, merge_in_split

@Task
def embed_images(
    dp: mk.DataPanel,
    img_column: str,
    batch_size: int = 128,
    num_workers: int = 4,
    model: str = "ViT-B/32",
    mmap: bool = False,
    split_dp: mk.DataPanel = None,
    splits: Iterable[str] = None,
    run_dir: str = None,
    **kwargs,
) -> mk.DataPanel:
    if splits is not None:
        dp = merge_in_split(dp, split_dp)
        dp = dp.lz[dp["split"].isin(splits)]

    ckpt_dir = '/pd/maya/rx-multimodal/classifier/checkpoints/0919_clip_vit_findingsimpressions_full/'
    with open(os.path.join(ckpt_dir, 'all_img_vectors.pkl'), 'rb') as f:
        keyToImgVector = pickle.load(f)
    print(len(keyToImgVector))

    dp['emb'] = [keyToImgVector[i] for i in list(dp['dicom_id'])]
    return dp
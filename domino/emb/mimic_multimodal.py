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

#embed_images must include a file_path parameter with a path to a folder with "all_img_vectors.pkl" (dict mapping dicom_ids to embeddings)
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
    file_path: str = None,
    **kwargs,
) -> mk.DataPanel:
    if splits is not None:
        dp = merge_in_split(dp, split_dp)
        dp = dp.lz[dp["split"].isin(splits)]

    if(file_path):
        print(f'Loading input vectors from file {file_path}')
    else:
        raise ValueError(f"File path required.")
    
    ckpt_dir = file_path
    with open(os.path.join(ckpt_dir, 'all_img_vectors.pkl'), 'rb') as f:
        keyToImgVector = pickle.load(f)
    print(f'Loaded {len(keyToImgVector)} multimodal embeddings')

    dp['emb'] = [keyToImgVector[i] for i in list(dp['dicom_id'])]
    return dp
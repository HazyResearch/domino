import os

import meerkat as mk
import numpy as np
import pandas as pd
import terra
from google.cloud import storage
from meerkat.contrib.mimic import build_mimic_dp


def download_mimic(dataset_dir: str):
    dp = build_mimic_dp(dataset_dir=dataset_dir, gcp_project="hai-gcp-fine-grained")
    dp = dp.lz[np.isin(dp["ViewPosition"].data, ["PA", "AP"])]

    dp["img"].map(lambda x: 0, num_workers=6, pbar=True)


download_mimic("/home/common/datasets/mimic")

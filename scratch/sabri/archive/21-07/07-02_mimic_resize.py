import os

import meerkat as mk
import numpy as np
import pandas as pd
import terra
from meerkat.contrib.mimic import build_mimic_dp
from torchvision.transforms import Compose, Lambda, Resize

from domino.data.mimic import build_dp


@terra.Task.make_task
def resize_mimic(dp: str, size: int = 512, run_dir: str = None):
    dp = dp.lz[np.isin(dp["ViewPosition"].data, ["PA", "AP"])]

    def write_resized(row):
        root, ext = os.path.splitext(row["jpg_path"])
        path = root + f"_{size}" + ext
        row["img_resized"].save(os.path.join(dp["img_resized"].local_dir, path))
        return 0

    dp["img_resized"] = dp["img"].copy()
    dp["img_resized"].transform = Resize([size, size])

    dp.map(
        write_resized,
        num_workers=6,
        pbar=True,
        input_columns=["img_resized", "jpg_path"],
    )


resize_mimic(dp=build_dp.out(948, load=True))

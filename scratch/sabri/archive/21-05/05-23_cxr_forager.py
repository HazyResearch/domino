import os

import numpy as np
import pandas as pd
from terra import Task

from domino.data.cxr_tube import build_cxr_df, get_dp


@Task
def convert_cxr_to_png(
    df: pd.DataFrame, dst_dir: str, image_ids: list = None, run_dir: str = None
):
    dp = get_dp(df)

    for split in ["train", "test"]:
        os.makedirs(os.path.join(dst_dir, split), exist_ok=True)
    if image_ids is None:

        def check_png(row: dict):
            image_id = row["image_id"]
            path = os.path.join(dst_dir, row["split"], f"img_{image_id}.png")
            return not os.path.exists(path)

        indices = np.where(
            dp[["image_id", "split"]].map(
                check_png, materialize=False, input_columns=["image_id", "split"]
            )
        )[0]
    else:
        indices = np.where(dp["image_id"].to_pandas().isin(image_ids))[0]
    filtered_dp = dp.lz[indices]

    def convert_to_png(row: dict):
        image_id = row["image_id"]
        path = os.path.join(dst_dir, row["split"], f"img_{image_id}.png")
        row["img"].save(path, "PNG")
        return path

    paths = filtered_dp.map(
        convert_to_png,
        columns=["image_id", "img", "split"],
        batched=False,
        batch_size=1,
        num_workers=0,
    )
    return paths.to_pandas()


if __name__ == "__main__":
    convert_cxr_to_png(
        df=build_cxr_df.out(),
        dst_dir="/home/common/datasets/cxr-tube/forager",
        image_ids=["1.2.276.0.7230010.3.1.4.8323329.11446.1517875232.986896"],
    )

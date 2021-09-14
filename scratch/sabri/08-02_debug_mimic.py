from meerkat.contrib.mimic import build_mimic_dp
import pandas as pd
import meerkat as mk
import os

dp = build_mimic_dp(
    dataset_dir="/Users/sabrieyuboglu/data/datasets/mimic",
    gcp_project="hai-gcp-fine-grained",
    tables=None,  # ["admit"],
    split=False,
)

dataset_dir = "/Users/sabrieyuboglu/data/datasets/mimic"
new_dp = dp.merge(
    mk.DataPanel.from_csv(os.path.join(dataset_dir, "mimic-cxr-2.0.0-split.csv"))[
        ["split", "dicom_id"]
    ],
    how="left",
    on="dicom_id",
)

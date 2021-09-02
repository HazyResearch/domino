import numpy as np
import pandas as pd
import terra

from domino.evaluate.train import score_linear_slices

df = terra.out(1302)
score_linear_slices(
    dp_run_id=terra.inp(1302)["dp_run_id"],
    model_df=df,
    num_workers=6,
    batch_size=64,
    num_gpus=4,
    num_cpus=31,
    layers={
        "layer4.2": "model.layer4",
        "layer4.0": "model.layer4.0",
        "layer4.1": "model.layer4.1",
        "layer3": "model.layer3",
        "layer2": "model.layer2",
    },
    reduction_fns=["mean"],
    input_column="input_224",
    id_column="dicom_id",
    split=["validate", "test"],
)

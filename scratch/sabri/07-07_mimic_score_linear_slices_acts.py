import numpy as np
import pandas as pd
import terra

from domino.evaluate.train import score_linear_slices

df = terra.out(1302)
score_linear_slices(
    dp_run_id=terra.inp(1302)["dp_run_id"],
    model_df=df,
    num_workers=7,
    batch_size=64,
    num_gpus=4,
    num_cpus=28,
    input_column="input_224",
    id_column="dicom_id",
    layers={"layer4": "model.layer4"},
)

import numpy as np
import pandas as pd
import terra

from domino.evaluate.train import score_linear_slices

df = terra.out(691)
score_linear_slices(
    dp_run_id=1381,
    model_df=df,
    num_workers=6,
    batch_size=512,
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
    split=["validate", "test"],
)

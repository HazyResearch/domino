import numpy as np
import pandas as pd
import terra

from domino.evaluate.train import score_linear_slices

df = terra.out(691).load()
score_linear_slices(
    dp_run_id=1381,
    model_df=df.iloc[18:27],
    num_workers=6,
    batch_size=512,
    layers={"layer4": "model.layer4"},
)

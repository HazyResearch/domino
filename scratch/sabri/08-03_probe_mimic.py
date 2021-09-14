from typing import Sequence

import meerkat as mk
import meerkat.contrib.mimic
import numpy as np
import pandas as pd
import terra
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


@terra.Task
def probe_layers(
    act_dp: mk.DataPanel,
    model_target: str,
    lr_targets: Sequence[str],
    run_dir: str = None,
):
    act_dp["ethnicity_white"] = act_dp["ethnicity"] == "WHITE"
    train_mask = np.random.random(len(act_dp)) > 0.5

    targets = [model_target] + lr_targets
    results = []
    for target in tqdm(targets):
        for layer in [col for col in act_dp.columns if col.startswith("act")]:
            lr = LogisticRegression()
            lr.fit(
                X=act_dp.lz[train_mask][layer].numpy(),
                y=act_dp.lz[train_mask][target].values,
            )
            preds = lr.predict_proba(
                X=act_dp.lz[~train_mask][layer].numpy(),
            )[:, -1]
            score = roc_auc_score(
                act_dp.lz[~train_mask][target].values,
                preds,
            )
            results.append({"target": target, "layer": layer[4:], "auroc": score})
    return pd.DataFrame(results)


probe_layers(
    model_target="support_devices_uzeros",
    act_dp=terra.out(4466),
    lr_targets=[
        "ethnicity_black",
        "ethnicity_asian",
        "ethnicity_hisp",
        "ethnicity_white",
        "patient_orientation_rf",
        "gender_male",
    ],
)

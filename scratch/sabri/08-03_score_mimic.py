from typing import Mapping, Sequence, Union

import meerkat as mk
import meerkat.contrib.mimic
import terra
import torch
import torch.nn as nn

from domino.vision import Classifier, score


def mean(x):
    return torch.mean(x, dim=[-1, -2])


@terra.Task
def score_model(
    dp: mk.DataPanel,
    model: Classifier,
    split: str,
    run_dir: str = None,
    **kwargs,
):
    dp = dp.lz[dp["split"] == split].lz[:10000]

    layers = {
        f"act_{name}.{idx}": module
        for name, layer in {
            "4": model.model.layer4,
            "3": model.model.layer3,
            "2": model.model.layer2,
            "1": model.model.layer1,
        }.items()
        for idx, module in enumerate(layer)
    }

    dp = score(
        model=model,
        dp=dp,
        input_column="input_512",
        id_column="dicom_id",
        layers=layers,
        reduction_fns=[mean],
        batch_size=64,
        device=1,
    )
    return dp


if __name__ == "__main__":
    run_id = 4448
    untrained = True
    if not untrained:
        model = terra.get_artifacts(run_id, "best_chkpt")["model"]
    else:
        model = Classifier(config={"arch": "resnet50"})
    inp = terra.inp(run_id)
    dp = inp["dp"]

    score_model(
        dp=dp,
        model=model,
        split="valid_test",
    )

from dataclasses import dataclass

import meerkat as mk
import numpy as np
import torch.nn as nn

from domino.utils import requires_columns

from .abstract import SliceDiscoveryMethod


class ConfusionSDM(SliceDiscoveryMethod):

    RESOURCES_REQUIRED = {"cpu": 0.25, "gpu": 0}

    def __init__(self, config: dict = None):
        super(ConfusionSDM, self).__init__(config)

    @requires_columns(dp_arg="data_dp", columns=[])
    def fit(
        self,
        data_dp: mk.DataPanel,
        model: nn.Module = None,
    ):
        return self

    @requires_columns(dp_arg="data_dp", columns=["probs", "target"])
    def transform(
        self,
        data_dp: mk.DataPanel,
    ):
        data_dp = data_dp.view()

        # get a slice corresponding to each cell of the confusion matrix.
        slices = np.stack(
            [
                (data_dp["target"] == target_idx)
                * (data_dp["probs"][:, pred_idx]).numpy()
                for target_idx in range(data_dp["probs"].shape[1])
                for pred_idx in range(data_dp["probs"].shape[1])
            ],
            axis=-1,
        )
        if slices.shape[1] > self.config.n_slices:
            raise ValueError(
                "ConfusionSDM is not configured to return enough slices to "
                "capture the full confusion matrix."
            )

        if slices.shape[1] < self.config.n_slices:
            # fill in the other predicted slices with zeros
            slices = np.concatenate(
                [
                    slices,
                    np.zeros((slices.shape[0], self.config.n_slices - slices.shape[1])),
                ],
                axis=1,
            )
        data_dp["pred_slices"] = slices

        return data_dp

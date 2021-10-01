from dataclasses import dataclass

import meerkat as mk
import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from domino.utils import VariableColumn, requires_columns

from .abstract import SliceDiscoveryMethod


class SupervisedSDM(SliceDiscoveryMethod):

    RESOURCES_REQUIRED = {"cpu": 1, "custom_resources": {"ram_gb": 2}}

    def __init__(self, config: dict = None):
        super().__init__(config)

        self.models = None

    @requires_columns(
        dp_arg="data_dp", columns=["slices", VariableColumn("self.config.emb")]
    )
    def fit(
        self,
        data_dp: mk.DataPanel,
        model: nn.Module = None,
    ):
        acts = data_dp[self.config.emb].data
        self.models = []
        for slice_idx in range(data_dp["slices"].shape[-1]):
            model = LogisticRegression(max_iter=10000)
            self.models.append(
                model.fit(X=acts, y=data_dp["slices"].data[:, slice_idx])
            )

        return self

    @requires_columns(
        dp_arg="data_dp", columns=["slices", VariableColumn("self.config.emb")]
    )
    def transform(
        self,
        data_dp: mk.DataPanel,
    ):
        assert self.models is not None
        acts = data_dp[self.config.emb].data
        dp = data_dp.view()
        slices = np.stack(
            [model.predict_proba(acts)[:, -1] for model in self.models], axis=-1
        )

        if slices.shape[1] > self.config.n_slices:
            raise ValueError(
                "SupervisedSDM is not configured to return enough slices to "
                "capture all the ground truth slices."
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

        dp["pred_slices"] = slices
        return dp

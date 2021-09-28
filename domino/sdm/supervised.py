from dataclasses import dataclass

import meerkat as mk
import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from domino.utils import VariableColumn, requires_columns

from .abstract import SliceDiscoveryMethod


class SupervisedSDM(SliceDiscoveryMethod):
    @dataclass
    class Config(SliceDiscoveryMethod.Config):
        layer: str = "model.layer4"

    RESOURCES_REQUIRED = {"cpu": 1, "custom_resources": {"ram_gb": 2}}

    def __init__(self, config: dict = None):
        super(SupervisedSDM, self).__init__(config)

        self.model = LogisticRegression()

    @requires_columns(
        dp_arg="data_dp", columns=["slices", VariableColumn("self.config.layer")]
    )
    def fit(
        self,
        data_dp: mk.DataPanel,
        model: nn.Module = None,
    ):
        acts = data_dp[self.config.layer].data

        self.model.fit(X=acts, y=data_dp["correlate"].data)

        return self

    @requires_columns(
        dp_arg="data_dp", columns=["slices", VariableColumn("self.config.layer")]
    )
    def transform(
        self,
        data_dp: mk.DataPanel,
    ):
        acts = data_dp[self.config.layer].data
        dp = data_dp.view()
        slices = self.model.predict_proba(acts)
        dp["slices"] = np.stack([slices[:, -1]] * self.config.n_slices, axis=-1)
        return dp

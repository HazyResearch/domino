from dataclasses import dataclass

import meerkat as mk
import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA, KernelPCA

from domino.utils import VariableColumn, requires_columns

from .abstract import SliceDiscoveryMethod


class PCASDM(SliceDiscoveryMethod):
    @dataclass
    class Config(SliceDiscoveryMethod.Config):
        layer: str = "model.layer4"
        whiten: bool = False

    RESOURCES_REQUIRED = {"cpu": 1, "custom_resources": {"ram_gb": 2}}

    def __init__(self, config: dict = None):
        super(PCASDM, self).__init__(config)

        self.pca = PCA(n_components=self.config.n_slices, whiten=self.config.whiten)

    @requires_columns(dp_arg="data_dp", columns=[VariableColumn("self.config.layer")])
    def fit(
        self,
        data_dp: mk.DataPanel,
        model: nn.Module = None,
    ):
        acts = data_dp[self.config.layer].data
        self.pca.fit(acts)

        return self

    @requires_columns(dp_arg="data_dp", columns=[VariableColumn("self.config.layer")])
    def transform(
        self,
        data_dp: mk.DataPanel,
    ):
        acts = data_dp[self.config.layer].data
        dp = data_dp.view()
        dp["slices"] = self.pca.transform(acts)
        return dp

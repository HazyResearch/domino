from dataclasses import dataclass

import meerkat as mk
import numpy as np
import torch.nn as nn

from domino.utils import requires_columns

from .abstract import SliceDiscoveryMethod


class PredSDM(SliceDiscoveryMethod):

    RESOURCES_REQUIRED = {"cpu": 0.75, "custom_resources": {"ram_gb": 1}}

    def __init__(self, config: dict = None):
        super(PredSDM, self).__init__(config)

    @requires_columns(dp_arg="data_dp", columns=[])
    def fit(
        self,
        data_dp: mk.DataPanel,
        model: nn.Module = None,
    ):
        return self

    @requires_columns(dp_arg="data_dp", columns=["output"])
    def transform(
        self,
        data_dp: mk.DataPanel,
    ):
        data_dp = data_dp.view()
        data_dp["slices"] = np.stack(
            [data_dp["output"].data[:, -1]] * self.config.n_slices, axis=-1
        )
        return data_dp

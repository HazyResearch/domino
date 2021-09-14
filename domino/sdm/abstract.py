from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

import meerkat as mk
import numpy as np
import torch
import torch.nn as nn

from domino.utils import requires_columns


class SliceDiscoveryMethod(ABC):
    @dataclass
    class Config:
        n_slices: int = 5
        emb_group: str = "main"
        emb: str = "emb"

    RESOURCES_REQUIRED = {"cpu": 1, "custom_resources": {"ram_gb": 4}}

    def __init__(self, config: dict = None, **kwargs):
        if config is not None:
            self.config = self.Config(**config, **kwargs)
        else:
            self.config = self.Config(**kwargs)

    @abstractmethod
    @requires_columns(dp_arg="data_dp", columns=["input", "target", "act", "pred"])
    def fit(
        self,
        model: nn.Module = None,
        data_dp: mk.DataPanel = None,
    ) -> SliceDiscoveryMethod:
        pass

    @abstractmethod
    def transform(self, data_dp: mk.DataPanel) -> mk.DataPanel:
        pass

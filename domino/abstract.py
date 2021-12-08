from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import meerkat as mk
import torch.nn as nn


class SliceDiscoveryMethod(ABC):
    @dataclass
    class Config:
        n_slices: int = 5

    RESOURCES_REQUIRED = {"cpu": 1, "custom_resources": {"ram_gb": 4}}

    def __init__(self, config: Union[Config, dict] = None):
        if isinstance(config, dict):
            config = self.Config(**config)
        self.config = config

    @abstractmethod
    def fit(
        self,
        model: nn.Module = None,
        data_dp: mk.DataPanel = None,
    ) -> SliceDiscoveryMethod:
        raise NotImplementedError()

    @abstractmethod
    def transform(self, data_dp: mk.DataPanel) -> mk.DataPanel:
        raise NotImplementedError()

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import meerkat as mk
import torch.nn as nn


class SliceDiscoveryMethod(ABC):
    @dataclass
    class Config:
        n_slices: int = 5
        emb: str = "emb"

    RESOURCES_REQUIRED = {"cpu": 1, "custom_resources": {"ram_gb": 4}}

    def __init__(self, config: dict = None, **kwargs):
        if config is not None:
            self.config = self.Config(**config, **kwargs)
        else:
            self.config = self.Config(**kwargs)

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

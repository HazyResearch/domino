from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import meerkat as mk
import torch.nn as nn


class SliceDiscoveryMethod(ABC):
    @dataclass
    class Config:
        pass

    RESOURCES_REQUIRED = {"cpu": 1, "custom_resources": {"ram_gb": 4}}

    def __init__(self, n_slices: int):

        self.config = self.Config()
        self.config.n_slices = n_slices

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

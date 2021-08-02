from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import meerkat as mk
import torch.nn as nn


class SliceDiscoveryMethod(ABC):
    @dataclass
    class Config:
        n_slices: int = 2

    def __init__(self, config: dict = None):
        if config is not None:
            self.config = self.Config(**config)
        else:
            self.config = self.Config()

    @abstractmethod
    def fit(self, data_dp: mk.DataPanel, model: nn.Module) -> SliceDiscoveryMethod:
        pass

    @abstractmethod
    def transform(self, data_dp: mk.DataPanel) -> mk.DataPanel:
        pass


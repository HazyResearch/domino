from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Union

import meerkat as mk
import numpy as np
import torch.nn as nn
from sklearn.base import BaseEstimator


@dataclass
class Config:
    pass


class Describer(ABC, BaseEstimator):
    def __init__(self):
        super().__init__()

        self.config = Config()

    @abstractmethod
    def describe(
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        slices: Union[str, np.ndarray] = "slices",
    ):
        raise NotImplementedError
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of this slicer. Returns a dictionary mapping from the names
        of the parameters (as they are defined in the ``__init__``) to their values. 

        Returns:
            Dict[str, Any]: A dictionary of parameters.

        """
        return self.config.__dict__

    def set_params(self, **params):
        raise ValueError(
            f"Slicer of type {self.__class__.__name__} does not support `set_params`."
        )

    def to(self, device: Union[str, int]):

        if device != "cpu":
            raise ValueError(f"Slicer of type {type(self)} does not support GPU.")
        # by default this is a no-op, but subclasses can override

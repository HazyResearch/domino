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


class Slicer(ABC, BaseEstimator):
    def __init__(self, n_slices: int):
        super().__init__()

        self.config = Config()
        self.config.n_slices = n_slices

    @abstractmethod
    def fit(
        self,
        model: nn.Module = None,
        data_dp: mk.DataPanel = None,
    ) -> Slicer:
        """
        Fit the slicer to data.

        Args:
            data (mk.DataPanel, optional): A `Meerkat DataPanel` with columns for
                embeddings, targets, and prediction probabilities. The names of the
                columns can be specified with the ``embeddings``, ``targets``, and
                ``pred_probs`` arguments. Defaults to None.
            embeddings (Union[str, np.ndarray], optional): The name of a column in
                ``data`` holding embeddings. If ``data`` is ``None``, then an np.ndarray
                of shape (n_samples, dimension of embedding). Defaults to
                "embedding".
            targets (Union[str, np.ndarray], optional): The name of a column in
                ``data`` holding class labels. If ``data`` is ``None``, then an
                np.ndarray of shape (n_samples,). Defaults to "target".
            pred_probs (Union[str, np.ndarray], optional): The name of
                a column in ``data`` holding model predictions (can either be "soft"
                probability scores or "hard" 1-hot encoded predictions). If
                ``data`` is ``None``, then an np.ndarray of shape (n_samples, n_classes)
                or (n_samples,) in the binary case. Defaults to "pred_probs".
            losses (Union[str, np.ndarray], optional): The name of a column in ``data``
                holding the loss of the model predictions. If ``data`` is ``None``,
                then an np.ndarray of shape (n_samples,). Defaults to "loss".

        Returns:
            Slicer: Returns a fit instance of the slicer.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(
        self,
        data: mk.DataPanel,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
    ) -> np.ndarray:
        """
        Get slice membership for data using the fit slicer.


        .. caution::
            Must call ``Slicer.fit`` prior to calling ``Slicer.predict``.


        Args:
            data (mk.DataPanel, optional): A `Meerkat DataPanel` with columns for
                embeddings, targets, and prediction probabilities. The names of the
                columns can be specified with the ``embeddings``, ``targets``, and
                ``pred_probs`` arguments. Defaults to None.
            embeddings (Union[str, np.ndarray], optional): The name of a colum in
                ``data`` holding embeddings. If ``data`` is ``None``, then an np.ndarray
                of shape (n_samples, dimension of embedding). Defaults to
                "embedding".
            targets (Union[str, np.ndarray], optional): The name of a column in
                ``data`` holding class labels. If ``data`` is ``None``, then an
                np.ndarray of shape (n_samples,). Defaults to "target".
            pred_probs (Union[str, np.ndarray], optional): The name of
                a column in ``data`` holding model predictions (can either be "soft"
                probability scores or "hard" 1-hot encoded predictions). If
                ``data`` is ``None``, then an np.ndarray of shape (n_samples, n_classes)
                or (n_samples,) in the binary case. Defaults to "pred_probs".
            losses (Union[str, np.ndarray], optional): The name of a column in ``data``
                holding the loss of the model predictions. If ``data`` is ``None``,
                then an np.ndarray of shape (n_samples,). Defaults to "loss".

        Returns:
            np.ndarray: A binary ``np.ndarray`` of shape (n_samples, n_slices) where
                values are either 1 or 0.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict_proba(
        self,
        data: mk.DataPanel,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
    ) -> np.ndarray:
        """
        Get probablisitic (**i.e.** soft) slice membership for data using the fit
        slicer.


        .. caution::
            Must call ``Slicer.fit`` prior to calling ``Slicer.predict``.


        Args:
            data (mk.DataPanel, optional): A `Meerkat DataPanel` with columns for
                embeddings, targets, and prediction probabilities. The names of the
                columns can be specified with the ``embeddings``, ``targets``, and
                ``pred_probs`` arguments. Defaults to None.
            embeddings (Union[str, np.ndarray], optional): The name of a colum in
                ``data`` holding embeddings. If ``data`` is ``None``, then an np.ndarray
                of shape (n_samples, dimension of embedding). Defaults to
                "embedding".
            targets (Union[str, np.ndarray], optional): The name of a column in
                ``data`` holding class labels. If ``data`` is ``None``, then an
                np.ndarray of shape (n_samples,). Defaults to "target".
            pred_probs (Union[str, np.ndarray], optional): The name of
                a column in ``data`` holding model predictions (can either be "soft"
                probability scores or "hard" 1-hot encoded predictions). If
                ``data`` is ``None``, then an np.ndarray of shape (n_samples, n_classes)
                or (n_samples,) in the binary case. Defaults to "pred_probs".
            losses (Union[str, np.ndarray], optional): The name of a column in ``data``
                holding the loss of the model predictions. If ``data`` is ``None``,
                then an np.ndarray of shape (n_samples,). Defaults to "loss".

        Returns:
            np.ndarray: A binary ``np.ndarray`` of shape (n_samples, n_slices) where
                values are either 1 or 0.
        """
        raise NotImplementedError()

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

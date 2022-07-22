from __future__ import annotations
from typing import Union
from domino._slice.abstract import Slicer
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import cross_entropy
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import numpy as np
import meerkat as mk

from ..utils import convert_to_torch, unpack_args, convert_to_numpy


class FusedSlicer(Slicer, nn.Module):
    def __init__(
        self,
        n_slices: int = 5,
        candidate_text: mk.DataPanel = None,
        text_column: Union[str, np.np.ndarray] = "text",
        text_embedding_column: Union[str, np.np.ndarray] = "embedding",
        device: Union[int, str] = "cpu",
    ):
        super().__init__(n_slices=n_slices)
        self.candidate_text, self.candidate_text_embeddings = unpack_args(
            candidate_text, text_column, text_embedding_column
        )
        (self.candidate_text_embeddings,) = convert_to_torch(
            self.candidate_text_embeddings
        )

        self.candidate_text_embeddings = self.candidate_text_embeddings

        self.device = device

        self.text_idxs = None
        self.text_embeddings = None
        self.text = None

    def _prepare_embs(self, *args):
        return [inp.to(device=self.device, dtype=torch.float) for inp in args]

    def fit(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = None,
        pred_probs: Union[str, np.ndarray] = None,
    ) -> FusedSlicer:
        embeddings, targets, pred_probs = unpack_args(
            data, embeddings, targets, pred_probs
        )
        (embeddings,) = convert_to_torch(embeddings)
        targets, pred_probs = convert_to_numpy(targets, pred_probs)
        embeddings, candidate_text_embeddings = self._prepare_embs(
            embeddings, self.candidate_text_embeddings
        )
        with torch.no_grad():
            slice_scores = torch.matmul(embeddings, candidate_text_embeddings.T)

        slice_scores = slice_scores.cpu().numpy()

        l = targets - pred_probs

        #slice_scores = MinMaxScaler().fit_transform(slice_scores)
        lr = Ridge(normalize=True).fit(slice_scores, l)  # Change this back!!!!

        coef = lr.coef_.squeeze()
        self.text_idxs = np.concatenate(
            [
                #np.argsort(coef)[: self.config.n_slices],
                np.argsort(-np.abs(coef))[: self.config.n_slices]
            ]
        )

        self.text_embeddings = candidate_text_embeddings[self.text_idxs]
        self.text = self.candidate_text[self.text_idxs]
        self.text_coefs = coef[self.text_idxs]

        return slice_scores 

    def predict(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = None,
        pred_probs: Union[str, np.ndarray] = None,
        losses: Union[str, np.ndarray] = None,
    ):
        return (
            self.predict(data, embeddings, targets, pred_probs, losses) > 0.5
        ).astype(int)

    def predict_proba(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = None,
        pred_probs: Union[str, np.ndarray] = None,
        losses: Union[str, np.ndarray] = None,
    ):
        if self.text_embeddings is None:
            raise ValueError("Must call `fit` before `predict`.")
        (embeddings,) = unpack_args(data, embeddings)
        (embeddings,) = convert_to_torch(embeddings)
        (embeddings,) = self._prepare_embs(embeddings)
        slice_scores = torch.matmul(embeddings, self.text_embeddings.T)
        return slice_scores.cpu().numpy()

    def describe(
        self,
        text_data: Union[dict, mk.DataPanel] = None,
        text_embeddings: Union[str, np.ndarray] = "embedding",
        text_descriptions: Union[str, np.ndarray] = "description",
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        num_descriptions: int = 3,
    ):
        output = []
        for pred_slice_idx in range(self.config.n_slices):
            output.append(
                {
                    "pred_slice_idx": pred_slice_idx,
                    "scores": [1],
                    "phrases": [self.text[pred_slice_idx]],
                }
            )

        return output

    def to(self, *args, **kwargs):
        """Intercept to on a device and set the self.device."""
        if isinstance(args[0], (int, str, torch.device)):
            self.device = args[0]
        return super().to(*args, **kwargs)

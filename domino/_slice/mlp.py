from __future__ import annotations
from typing import Union
from domino._slice.abstract import Slicer
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import cross_entropy
import torch
from tqdm import tqdm
import numpy as np
import meerkat as mk

from ..utils import convert_to_torch, unpack_args, convert_to_numpy


class MLPSlicer(Slicer, nn.Module):
    def __init__(
        self,
        n_slices: int = 5,
        lr: float = 1e-2,
        alpha: float = 1e-2,
        max_epochs: int = 100,
        batch_size: int = 1024,
        device: str = "cpu",
        pbar: bool = True,
        return_losses: bool = False,
    ):
        super().__init__(n_slices=n_slices)
        self.mlp = None
        self.config.lr = lr
        self.config.alpha = alpha
        self.config.max_epochs = max_epochs
        self.config.batch_size = batch_size

        self.device = device
        self.pbar = pbar
        self.return_losses = return_losses

        self.encoder = None
        self.decoder = None

    def fit(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = None,
        pred_probs: Union[str, np.ndarray] = None,
    ) -> MLPSlicer:
        embeddings, targets, pred_probs = unpack_args(
            data, embeddings, targets, pred_probs
        )
        embeddings, targets, pred_probs = convert_to_torch(
            embeddings, targets, pred_probs
        )
        # l = torch.stack([targets, pred_probs], dim=-1)
        l = targets - pred_probs
        self._fit(x=embeddings, l=l)

    def predict(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = None,
        pred_probs: Union[str, np.ndarray] = None,
        losses: Union[str, np.ndarray] = None,
    ):
        if self.encoder is None:
            raise ValueError("Must call `fit` before `predict`.")
        (embeddings,) = unpack_args(data, embeddings)
        (embeddings,) = convert_to_torch(embeddings)
        return self._predict(x=embeddings, return_probs=False)

    def predict_proba(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = None,
        pred_probs: Union[str, np.ndarray] = None,
        losses: Union[str, np.ndarray] = None,
    ):
        if self.encoder is None:
            raise ValueError("Must call `fit` before `predict`.")
        (embeddings,) = unpack_args(data, embeddings)
        (embeddings,) = convert_to_torch(embeddings)

        return self._predict_proba(x=embeddings).cpu().numpy()

    def describe(
        self,
        text_data: Union[dict, mk.DataPanel],
        text_embeddings: Union[str, np.ndarray] = "embedding",
        text_descriptions: Union[str, np.ndarray] = "description",
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        num_descriptions: int = 3,
    ):
        text_embeddings, text_descriptions = unpack_args(
            text_data, text_embeddings, text_descriptions
        )
        (text_embeddings,) = convert_to_torch(text_embeddings)

        # also produce predictions for the inverse
        probs = np.concatenate(
            [
                self._predict_proba(text_embeddings).cpu().numpy(),
                self._predict_proba(-text_embeddings).cpu().numpy(),
            ],
            axis=0,
        )
        text_descriptions = np.concatenate(
            [
                text_descriptions,
                "not " + text_descriptions,
            ],
            axis=0,
        )

        output = []
        for pred_slice_idx in range(self.config.n_slices):
            scores = probs[:, pred_slice_idx]
            idxs = np.argsort(-scores)[:num_descriptions]
            output.append(
                {
                    "pred_slice_idx": pred_slice_idx,
                    "scores": list(scores[idxs]),
                    "phrases": list(text_descriptions[idxs]),
                }
            )

        return output

    def to(self, *args, **kwargs):
        """Intercept to on a device and set the self.device."""
        if isinstance(args[0], (int, str, torch.device)):
            self.device = args[0]
        return super().to(*args, **kwargs)

    def _prepare_inputs(self, *args):
        return [inp.to(device=self.device, dtype=torch.float) for inp in args]

    def forward(self, x: torch.Tensor):
        return torch.sigmoid(self.encoder(x))

    def _fit(self, x: torch.Tensor, l: torch.Tensor):
        x, l = self._prepare_inputs(x, l)
        if len(l.shape) == 1:
            l = l.unsqueeze(1)

        self.embedding_dim = x.shape[1]
        self.response_dim = l.shape[1]
        self.encoder = nn.Linear(self.embedding_dim, self.config.n_slices).to(
            self.device
        )
        self.decoder = nn.Linear(self.config.n_slices, self.response_dim).to(
            self.device
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)

        losses = []
        with tqdm(total=self.config.max_epochs, disable=not self.pbar) as pbar:
            for epoch in range(self.config.max_epochs):
                batcher = lambda data: torch.split(data, self.config.batch_size, dim=0)
                for x_batch, l_batch in zip(batcher(x), batcher(l)):

                    s_batch = self.forward(x_batch)
                    l_hat = self.decoder(s_batch)

                    loss = F.mse_loss(l_hat, l_batch)

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                if self.pbar:
                    pbar.update()
                    pbar.set_postfix(epoch=epoch, loss=loss.detach().cpu().item())

                if self.return_losses:

                    losses.extend(
                        [
                            {
                                "value": value,
                                "name": name,
                                "epoch": epoch,
                            }
                            for name, value in [
                                ("response", response_loss.detach().cpu().item()),
                                ("embedding", embedding_loss.detach().cpu().item()),
                                ("total", loss.detach().cpu().item()),
                            ]
                        ]
                    )
        return losses

    @torch.no_grad()
    def _predict_proba(self, x: torch.Tensor):
        x = self._prepare_inputs(x)[0]
        return self.forward(x)

    @torch.no_grad()
    def _predict(self, x: torch.Tensor, return_probs: bool = False):
        probs = self._predict_proba(x)
        preds = (probs > 0.5).to(int)

        if return_probs:
            return preds, probs
        else:
            return preds

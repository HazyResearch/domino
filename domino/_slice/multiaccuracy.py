from __future__ import annotations


import datetime
from dataclasses import dataclass
from typing import Union

import meerkat as mk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from meerkat.columns.tensor_column import TensorColumn
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from torch.nn.functional import cross_entropy
from tqdm import tqdm

from domino.utils import VariableColumn, requires_columns
from domino.utils import convert_to_numpy, unpack_args

from .abstract import Slicer


class MultiaccuracySlicer(Slicer):

    r"""
    Slice discovery based on MultiAccuracy auditing [kim_2019].

    Discover slices by learning a simple function (e.g. ridge regression) that 
    correlates with the residual.

    Examples
    --------
    Suppose you've trained a model and stored its predictions on a dataset in
    a `Meerkat DataPanel <https://github.com/robustness-gym/meerkat>`_ with columns
    "emb", "target", and "pred_probs". After loading the DataPanel, you can discover
    underperforming slices of the validation dataset with the following:

    .. code-block:: python

        from domino import MultiaccuracySlicer
        dp = ...  # Load dataset into a Meerkat DataPanel

        # split dataset
        valid_dp = dp.lz[dp["split"] == "valid"]
        test_dp = dp.lz[dp["split"] == "test"]

        slicer = MultiaccuracySlicer()
        slicer.fit(
            data=valid_dp, embeddings="emb", targets="target", pred_probs="pred_probs"
        )
        dp["slicer"] = slicer.predict(
            data=test_dp, embeddings="emb", targets="target", pred_probs="pred_probs"
        )


    Args:
        n_slices (int, optional): The number of slices to discover.
            Defaults to 5.
        eta (float, optional): Step size for the logits update, see final line 
            Algorithm 1 in . Defaults to 0.1
        dev_valid_frac (float, optional): The fraction of data held out for computing 
            corr. Defaults to 0.3. 
    
    .. [kim_2019]

        @inproceedings{kim2019multiaccuracy,
            title={Multiaccuracy: Black-box post-processing for fairness in classification},
            author={Kim, Michael P and Ghorbani, Amirata and Zou, James},
            booktitle={Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society},
            pages={247--254},
            year={2019}
        }

    """

    def __init__(
        self,
        n_slices: int = 5,
        eta: float = 0.1,
        dev_valid_frac: float = 0.1,
        partition_size_threshold: int = 10,
        pbar: bool = False,
    ):
        super().__init__(n_slices=n_slices)

        self.config.eta = eta
        self.config.dev_valid_frac = dev_valid_frac
        self.config.partition_size_threshold = partition_size_threshold

        self.auditors = []
        
        self.pbar = pbar 

    def fit(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
    ) -> MultiaccuracySlicer:
        """
        Fit the mixture model to data.

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

        Returns:
            MultiaccuracySlicer: Returns a fit instance of MultiaccuracySlicer.
        """
        embeddings, targets, pred_probs = unpack_args(
            data, embeddings, targets, pred_probs
        )
        embeddings, targets, pred_probs = convert_to_numpy(
            embeddings, targets, pred_probs
        )

        pred_probs = pred_probs[:, 1] if pred_probs.ndim > 1 else pred_probs

        # inverse of sigmoid
        logits = np.log(pred_probs / (1 - pred_probs))

        dev_train_idxs, dev_valid_idxs = self._split_data(np.arange(len(targets)))
        for t in tqdm(range(self.config.n_slices), disable=not self.pbar):
            # partitioning the input space X based on the initial classifier predictions
            preds = (pred_probs > 0.5).astype(int)
            partitions = [1 - preds, preds, np.ones_like(preds)]

            # compute the partial derivative of the cross-entropy loss with respect to
            # the predictions
            delta = self._compute_partial_derivative(pred_probs, targets)
            residual = pred_probs - targets

            corrs = []
            candidate_auditors = []
            for partition in partitions:
                # for each partition, train a classifier to predict the partial
                # derivative of the cross entropy loss with respect to predictions
                partition_dev_train = np.where(partition[dev_train_idxs] == 1)[0]
                partition_dev_valid = np.where(partition[dev_valid_idxs] == 1)[0]
                if (
                    len(partition_dev_train) < self.config.partition_size_threshold
                ) or (len(partition_dev_valid) < self.config.partition_size_threshold):
                    continue
                rr = Ridge(alpha=1)
                rr.fit(
                    embeddings[dev_train_idxs][partition_dev_train],
                    delta[dev_train_idxs][partition_dev_train],
                )
                rr_prediction = rr.predict(
                    embeddings[dev_valid_idxs][partition_dev_valid]
                )

                candidate_auditors.append(rr)
                corrs.append(
                    np.mean(
                        rr_prediction
                        * np.abs(residual[dev_valid_idxs][partition_dev_valid])
                    )
                )

            partition_idx = np.argmax(corrs)
            auditor = candidate_auditors[partition_idx]
            h = (
                np.matmul(embeddings, np.expand_dims(auditor.coef_, -1))[:, 0]
                + auditor.intercept_
            )
            if partition_idx == 0:
                logits += self.config.eta * h * partitions[partition_idx]
            else:
                logits -= self.config.eta * h * partitions[partition_idx]
            pred_probs = torch.sigmoid(torch.tensor(logits)).numpy()
            self.auditors.append(auditor)

        return self

    def predict(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
    ) -> np.ndarray:
        """
        Get probabilistic slice membership for data using a fit mixture model.


        .. caution::
            Must call ``MultiaccuracySlicer.fit`` prior to calling ``MultiaccuracySlicer.predict``.


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

        Returns:
            np.ndarray: A binary ``np.ndarray`` of shape (n_samples, n_slices) where
                values are either 1 or 0.
        """
        probs = self.predict_proba(
            data=data,
            embeddings=embeddings,
            targets=targets,
            pred_probs=pred_probs,
        )
        return (probs > 0.5).astype(int)

    def predict_proba(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
    ) -> np.ndarray:
        """
        Get probabilistic slice membership for data using a fit mixture model.

        .. caution::
            Must call ``MultiaccuracySlicer.fit`` prior to calling
            ``MultiaccuracySlicer.predict_proba``.


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

        Returns:
            np.ndarray: A ``np.ndarray`` of shape (n_samples, n_slices) where values in
                are in range [0,1] and rows sum to 1.
        """
        (embeddings,) = unpack_args(data, embeddings)
        (embeddings,) = convert_to_numpy(embeddings)

        all_weights = []
        for slice_idx in range(self.config.n_slices):
            auditor = self.auditors[slice_idx]
            h = (
                np.matmul(embeddings, np.expand_dims(auditor.coef_, -1))[:, 0]
                + auditor.intercept_
            )
            all_weights.append(h)
        pred_slices = np.stack(all_weights, axis=1)
        max_scores = np.max(pred_slices, axis=0)
        return pred_slices / max_scores[None, :]

    def _compute_partial_derivative(self, p, y):
        """
        Compute a smoothed version of the partial derivative function of the cross-entropy
        loss with respect to the predictions.
        To help
        """
        y0 = (1 - y) * ((p < 0.9) / (1 - p + 1e-20) + (p >= 0.9) * (100 * p - 80))
        y1 = y * ((p >= 0.1) / (p + 1e-20) + (p < 0.1) * (20 - 100 * p))

        return y0 + y1

    def _split_data(self, data):
        ratio = [1 - self.config.dev_valid_frac, self.config.dev_valid_frac]
        num = (
            data[0].shape[0]
            if type(data) == list or type(data) == tuple
            else data.shape[0]
        )
        idx = np.arange(num)
        idx_train = idx[: int(ratio[0] * num)]
        idx_val = idx[int(ratio[0] * num) : int((ratio[0] + ratio[1]) * num)]
        train = data[idx_train]
        val = data[idx_val]
        return train, val

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

from .abstract import SliceDiscoveryMethod


class MultiaccuracySDM(SliceDiscoveryMethod):
    @dataclass
    class Config(SliceDiscoveryMethod.Config):
        eta: float = 0.1  # step size for the logits update, see final line algorithm 1
        dev_valid_frac: float = 0.3  # the fraction of data held out for computing corr

    RESOURCES_REQUIRED = {"cpu": 1, "gpu": 0}

    def __init__(self, config: dict = None, **kwargs):
        super().__init__(config, **kwargs)
        self.auditors = []

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

    def _compute_partial_derivative(self, p, y):
        """
        Compute a smoothed version of the partial derivative function of the cross-entropy
        loss with respect to the predictions.
        To help
        """
        y0 = (1 - y) * ((p < 0.9) / (1 - p + 1e-20) + (p >= 0.9) * (100 * p - 80))
        y1 = y * ((p >= 0.1) / (p + 1e-20) + (p < 0.1) * (20 - 100 * p))

        return y0 + y1

    @requires_columns(
        dp_arg="data_dp", columns=["probs", "target", VariableColumn("self.config.emb")]
    )
    def fit(
        self,
        data_dp: mk.DataPanel,
        model: nn.Module = None,
    ):

        probs = data_dp["probs"].data[:, 1].numpy()
        y = data_dp["target"].data
        latent = data_dp[self.config.emb].data
        logits = np.log(probs / (1 - probs))

        dev_train_idxs, dev_valid_idxs = self._split_data(np.arange(len(data_dp)))
        for t in range(self.config.n_slices):
            # partitioning the input space X based on the initial classifier predictions
            preds = (probs > 0.5).astype(int)
            partitions = [1 - preds, preds, np.ones_like(preds)]

            # compute the partial derivative of the cross-entropy loss with respect to
            # the predictions
            delta = self._compute_partial_derivative(probs, y)
            residual = probs - y

            corrs = []
            candidate_auditors = []
            for partition in partitions:
                # for each partition, train a classifier to predict the partial
                # derivative of the cross entropy loss with respect to predictions
                partition_dev_train = np.where(partition[dev_train_idxs] == 1)[0]
                partition_dev_valid = np.where(partition[dev_valid_idxs] == 1)[0]

                rr = Ridge(alpha=1)
                rr.fit(
                    latent[dev_train_idxs][partition_dev_train],
                    delta[dev_train_idxs][partition_dev_train],
                )
                rr_prediction = rr.predict(latent[dev_valid_idxs][partition_dev_valid])

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
                np.matmul(latent, np.expand_dims(auditor.coef_, -1))[:, 0]
                + auditor.intercept_
            )
            if partition_idx == 0:
                logits += self.config.eta * h * partitions[partition_idx]
            else:
                logits -= self.config.eta * h * partitions[partition_idx]
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            self.auditors.append(auditor)

        return self

    @requires_columns(dp_arg="data_dp", columns=[VariableColumn("self.config.emb")])
    def transform(
        self,
        data_dp: mk.DataPanel,
    ):
        dp = data_dp.view()
        all_weights = []

        for slice_idx in range(self.config.n_slices):
            auditor = self.auditors[slice_idx]
            h = (
                np.matmul(data_dp[self.config.emb], np.expand_dims(auditor.coef_, -1))[
                    :, 0
                ]
                + auditor.intercept_
            )
            all_weights.append(h)
        dp["pred_slices"] = np.stack(all_weights, axis=1)
        return dp

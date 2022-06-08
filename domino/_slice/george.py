from dataclasses import dataclass

import meerkat as mk
import numpy as np
import torch
import torch.nn as nn
import umap
from sklearn.decomposition import PCA
#from stratification.cluster.models.cluster import AutoKMixtureModel
from torch.nn.functional import cross_entropy
from umap import UMAP

from domino.utils import VariableColumn, requires_columns

from .abstract import SliceDiscoveryMethod


class GeorgeSDM(SliceDiscoveryMethod):
    @dataclass
    class Config(SliceDiscoveryMethod.Config):
        n_components: int = 2
        n_clusters: int = 25
        n_classes: int = 2
        reduction_method: str = "umap"
        cluster_method: str = "gmm"
        n_init: int = 3
        concat_loss_component: bool = False

    RESOURCES_REQUIRED = {"cpu": 1, "gpu": 0}

    def __init__(self, config: dict = None, **kwargs):
        super().__init__(config, **kwargs)

        self.class_to_reducer = {
            klass: self._get_reducer() for klass in range(self.config.n_classes)
        }
        self.class_to_clusterer = {
            klass: AutoKMixtureModel(
                cluster_method=self.config.cluster_method,
                max_k=self.config.n_clusters,
                n_init=self.config.n_init,
                search=False,
            )
            for klass in range(self.config.n_classes)
        }

    def _get_reducer(self):
        if self.config.reduction_method == "umap":
            return UMAP(n_components=self.config.n_components)
        elif self.config.reduction_method == "pca":
            return PCA(n_components=self.config.n_components)
        else:
            raise ValueError(
                f"Reduction method {self.config.reduction_method} not supported."
            )

    def _compute_losses(self, data_dp: mk.DataPanel):
        probs = (
            data_dp["probs"].data
            if isinstance(data_dp["probs"], mk.TensorColumn)
            else torch.tensor(data_dp["probs"].data)
        )
        return cross_entropy(
            probs.to(torch.float32),
            torch.tensor(data_dp["target"]).to(torch.long),
            reduction="none",
        )

    @requires_columns(
        dp_arg="data_dp", columns=["probs", "target", VariableColumn("self.config.emb")]
    )
    def fit(
        self,
        data_dp: mk.DataPanel,
        model: nn.Module = None,
    ):
        data_dp["loss"] = self._compute_losses(data_dp).data.numpy()
        self.slice_cluster_indices = {}
        for klass in range(self.config.n_classes):
            # filter `data_dp` to only include rows in the class
            curr_dp = data_dp.lz[data_dp["target"] == klass]

            # (1) reduction phase
            embs = curr_dp[self.config.emb].data

            reduced_embs = self.class_to_reducer[klass].fit_transform(embs)
            if self.config.concat_loss_component:
                reduced_embs = np.concatenate(
                    [
                        reduced_embs,
                        np.expand_dims(curr_dp["loss"].data, axis=1),
                    ],
                    axis=1,
                )

            # (2) clustering phase
            self.class_to_clusterer[klass].fit(reduced_embs)
            clusters = self.class_to_clusterer[klass].predict_proba(reduced_embs)

            cluster_losses = np.dot(curr_dp["loss"].data.T, clusters)

            # need to
            n_slices = self.config.n_slices // self.config.n_classes + (
                self.config.n_slices % self.config.n_classes
            ) * int(klass == 0)

            self.slice_cluster_indices[klass] = (-cluster_losses).argsort()[:n_slices]

        return self

    @requires_columns(
        dp_arg="data_dp", columns=["target", VariableColumn("self.config.emb")]
    )
    def transform(
        self,
        data_dp: mk.DataPanel,
    ):
        slices = np.zeros((len(data_dp), self.config.n_slices))

        start = 0
        for klass in range(self.config.n_classes):
            # filter `data_dp` to only include rows in the class
            curr_dp = data_dp.lz[data_dp["target"] == klass]

            # (1) reduction phase
            acts = curr_dp[self.config.emb].data
            reduced_embs = self.class_to_reducer[klass].transform(acts)

            if self.config.concat_loss_component:
                losses = self._compute_losses(curr_dp).data.numpy()
                reduced_embs = np.concatenate(
                    [reduced_embs, np.expand_dims(losses, axis=1)], axis=1
                )

            # (2) cluster phase
            if self.config.cluster_method == "kmeans":
                raise NotImplementedError
            else:
                # ensure that the slice atrix
                class_clusters = self.class_to_clusterer[klass].predict_proba(
                    reduced_embs
                )
                class_slices = class_clusters[:, self.slice_cluster_indices[klass]]
                slices[
                    data_dp["target"] == klass, start : start + class_slices.shape[-1]
                ] = class_slices
                start = start + class_slices.shape[-1]
        data_dp["pred_slices"] = slices
        # if self.config.cluster_method == "kmeans":
        #     # since slices in other methods are not necessarily mutually exclusive, it's
        #     # important to return as a matrix of binary columns, one for each slice
        #     dp["pred_slices"] = np.stack(
        #         [
        #             (dp["pred_slices"].data == slice_idx).astype(int)
        #             for slice_idx in range(self.config.n_slices)
        #         ],
        #         axis=-1,
        #     )
        return data_dp

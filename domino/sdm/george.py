from dataclasses import dataclass

import meerkat as mk
import numpy as np
import torch.nn as nn

# from stratification.cluster.models.cluster import AutoKMixtureModel
from umap import UMAP

from domino.utils import VariableColumn, requires_columns

from .abstract import SliceDiscoveryMethod


class GeorgeSDM(SliceDiscoveryMethod):
    @dataclass
    class Config(SliceDiscoveryMethod.Config):
        n_umap_components: int = 2
        num_classes: int = 2
        cluster_method: str = "gmm"
        n_init: int = 3

    RESOURCES_REQUIRED = {"cpu": 1}

    def __init__(self, config: dict = None, **kwargs):
        super().__init__(config, **kwargs)

        self.class_to_umap = {
            klass: UMAP(n_components=self.config.n_umap_components)
            for klass in range(self.config.num_classes)
        }
        self.n_slices_per_class = self.config.n_slices // self.config.num_classes
        self.class_to_kmeans = {
            klass: AutoKMixtureModel(
                cluster_method=self.config.cluster_method,
                max_k=self.n_slices_per_class,
                n_init=self.config.n_init,
            )
            for klass in range(self.config.num_classes)
        }

    @requires_columns(
        dp_arg="data_dp", columns=["target", VariableColumn("self.config.emb")]
    )
    def fit(
        self,
        data_dp: mk.DataPanel,
        model: nn.Module = None,
    ):
        self.classes = np.unique(data_dp["target"])
        for klass in self.classes:
            # filter `data_dp` to only include rows in the class
            curr_dp = data_dp.lz[data_dp["target"] == klass]

            # (1) reduction phase
            acts = curr_dp[self.config.emb].data
            print("fitting umap")
            umap_embs = self.class_to_umap[klass].fit_transform(acts)
            print("fitting umap done")
            # (2) clustering phase
            self.class_to_kmeans[klass].fit(umap_embs)
        return self

    @requires_columns(
        dp_arg="data_dp", columns=["target", VariableColumn("self.config.emb")]
    )
    def transform(
        self,
        data_dp: mk.DataPanel,
    ):

        dp = []
        for class_idx, klass in enumerate(self.classes):
            # filter `data_dp` to only include rows in the class
            curr_dp = data_dp.lz[data_dp["target"] == klass]

            # (1) reduction phase
            acts = curr_dp[self.config.emb].data
            umap_embs = self.class_to_umap[klass].transform(acts)
            for component_idx in range(self.config.n_umap_components):
                curr_dp[f"umap_{component_idx}"] = umap_embs[:, component_idx]

            # (2) cluster phase
            if self.config.cluster_method == "kmeans":
                curr_dp["pred_slices"] = self.class_to_kmeans[klass].predict(
                    umap_embs
                ) + (klass * self.config.num_classes)
            else:
                # ensure that the slice atrix
                class_slices = self.class_to_kmeans[klass].predict_proba(umap_embs)
                slices = np.zeros((class_slices.shape[0], self.config.n_slices))
                start = self.n_slices_per_class * class_idx
                slices[:, start : start + class_slices.shape[-1]] = class_slices
                curr_dp["pred_slices"] = slices

            dp.append(curr_dp)
        dp = mk.concat(dp)

        if self.config.cluster_method == "kmeans":
            # since slices in other methods are not necessarily mutually exclusive, it's
            # important to return as a matrix of binary columns, one for each slice
            dp["pred_slices"] = np.stack(
                [
                    (dp["pred_slices"].data == slice_idx).astype(int)
                    for slice_idx in range(self.config.n_slices)
                ],
                axis=-1,
            )
        return dp

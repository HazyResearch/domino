from dataclasses import dataclass

import meerkat as mk
import numpy as np
import torch.nn as nn
from stratification.cluster.models.cluster import AutoKMixtureModel
from umap import UMAP

from domino.utils import requires_columns

from .abstract import SliceDiscoveryMethod


class GeorgeSDM(SliceDiscoveryMethod):
    @dataclass
    class Config(SliceDiscoveryMethod.Config):
        layer: str = "model.layer4"
        n_umap_components: int = 2
        num_classes: int = 2
        cluster_method: str = "kmeans"
        n_init: int = 3

    def __init__(self, config: dict = None):
        super(GeorgeSDM, self).__init__(config)

        self.class_to_umap = {
            klass: UMAP(n_components=self.config.n_umap_components)
            for klass in range(self.config.num_classes)
        }
        self.class_to_kmeans = {
            klass: AutoKMixtureModel(
                cluster_method=self.config.cluster_method,
                max_k=self.config.n_slices // self.config.num_classes,
                n_init=self.config.n_init,
            )
            for klass in range(self.config.num_classes)
        }

    @requires_columns(dp_arg="data_dp", columns=["target", "act"])
    def fit(
        self,
        data_dp: mk.DataPanel,
        model: nn.Module = None,
    ):
        classes = np.unique(data_dp["target"])
        for klass in classes:
            # filter `data_dp` to only include rows in the class
            curr_dp = data_dp.lz[data_dp["target"] == klass]

            # (1) reduction phase
            acts = curr_dp["act"].mean(axis=[-1, -2]).data
            umap_embs = self.class_to_umap[klass].fit_transform(acts)
            # (2) clustering phase
            self.class_to_kmeans[klass].fit(umap_embs)
        return self

    @requires_columns(dp_arg="data_dp", columns=["target", "act"])
    def transform(
        self,
        data_dp: mk.DataPanel,
    ):

        classes = np.unique(data_dp["target"])
        dp = []
        for klass in classes:
            # filter `data_dp` to only include rows in the class
            curr_dp = data_dp.lz[data_dp["target"] == klass]

            # (1) reduction phase
            acts = curr_dp["act"].mean(axis=[-1, -2]).data
            umap_embs = self.class_to_umap[klass].transform(acts)
            for component_idx in range(self.config.n_umap_components):
                curr_dp[f"umap_{component_idx}"] = umap_embs[:, component_idx]

            # (2) cluster phase
            curr_dp["slices"] = self.class_to_kmeans[klass].predict(umap_embs) + (
                klass * self.config.num_classes
            )

            dp.append(curr_dp)
        dp = mk.concat(dp)
        for slice_idx in range(self.config.n_slices):
            dp[f"slice_{slice_idx}"] = (dp["slices"] == slice_idx).astype(int)

        return dp

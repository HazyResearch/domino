from dataclasses import dataclass

import meerkat as mk
import torch.nn as nn
from sklearn.cluster import KMeans
from umap import UMAP

from .abstract import SliceDiscoveryMethod


class GeorgeSDM(SliceDiscoveryMethod):
    @dataclass
    class Config(SliceDiscoveryMethod.Config):
        layer: str = "model.layer4"
        n_umap_components: int = 2

    def __init__(self, config: dict = None):
        super(GeorgeSDM, self).__init__(config)
        self.umap = UMAP(n_components=self.config.n_umap_components)
        self.kmeans = KMeans(n_clusters=self.config.n_slices)

    def fit(self, data_dp: mk.DataPanel, model: nn.Module = None):
        column_name = f"activation_{self.config.layer}"
        if column_name not in data_dp:
            raise ValueError

        acts = data_dp[column_name].data.mean(axis=[-1, -2])
        umap_embs = self.umap.fit_transform(acts)
        self.kmeans.fit(umap_embs)
        return self

    def transform(self, data_dp: mk.DataPanel):
        column_name = f"activation_{self.config.layer}"
        if column_name not in data_dp:
            raise ValueError

        acts = data_dp[column_name].data.mean(axis=[-1, -2])
        umap_embs = self.umap.transform(acts)
        clusters = self.kmeans.predict(umap_embs)

        dp = data_dp.view()
        for component_idx in range(self.config.n_umap_components):
            dp[f"george_umap_{component_idx}"] = umap_embs[:, component_idx]

        dp["george_slices"] = clusters
        for slice_idx in range(self.config.n_slices):
            dp[f"george_slice_{slice_idx}"] = (clusters == slice_idx).astype(int)

        return dp

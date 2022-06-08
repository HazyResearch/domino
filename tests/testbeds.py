import os

import meerkat as mk
import numpy as np
from PIL import Image
from sklearn.datasets import make_blobs
import torch


class ImageColumnTestBed:
    def __init__(
        self,
        tmpdir: str,
        length: int = 16,
    ):
        self.image_paths = []
        self.image_arrays = []
        self.ims = []
        self.data = []

        for i in range(0, length):
            self.image_arrays.append((i * np.ones((4, 4, 3))).astype(np.uint8))
            im = Image.fromarray(self.image_arrays[-1])
            self.ims.append(im)
            self.data.append(im)
            filename = "{}.png".format(i)
            im.save(os.path.join(tmpdir, filename))
            self.image_paths.append(os.path.join(tmpdir, filename))

        self.col = mk.ImageColumn.from_filepaths(
            self.image_paths,
            loader=Image.open,
        )


class TextColumnTestBed:
    def __init__(self, length: int = 16):
        self.data = ["Row " * idx for idx in range(length)]
        self.col = mk.PandasSeriesColumn(self.data)


class SliceTestBed:
    def __init__(self, length: int = 16, type: str = "numpy"):

        if type == "numpy":
            conversion = lambda x: x
        elif type == "torch":
            conversion = torch.tensor
        else:
            raise ValueError("Unknown type: {}".format(type))

        gaussian_means = np.array(
            [
                [0.0, 5.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 4.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 5.0, 1.0],
                [5.0, 2.0, 0.0, 1.0, 10.0],
                [1.0, 10.0, 2.0, 2.0, 0.0],
            ]
        )

        emb, clusters = make_blobs(
            n_samples=length, centers=gaussian_means, cluster_std=1.0, random_state=42
        )

        targets = clusters == 1
        preds = (clusters == 1) | (clusters == 2)

        self.clusters = clusters
        self.dp = mk.DataPanel(
            {
                "embedding": conversion(emb),
                "target": conversion(targets),
                "pred_probs": conversion(
                    np.stack([1 - preds, preds], axis=1).astype(float)
                ),
                "losses": conversion(np.abs(targets ^ preds).astype(float)),
            }
        )

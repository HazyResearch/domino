import os

import meerkat as mk
import numpy as np
from PIL import Image


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
        )


class TextColumnTestBed:
    def __init__(self, length: int = 16):
        self.data = ["Row " * idx for idx in range(length)]
        self.col = mk.PandasSeriesColumn(self.data)

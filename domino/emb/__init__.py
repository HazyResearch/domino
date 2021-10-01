from typing import Iterable

import meerkat as mk
import terra

from domino.utils import merge_in_split


@terra.Task
def embed_images(
    emb_type: str,
    dp: mk.DataPanel,
    img_column: str,
    split_dp: mk.DataPanel = None,
    splits: Iterable[str] = None,
    file_path: str = None,
    **kwargs,
):
    if splits is not None:
        dp = merge_in_split(dp, split_dp)
        dp = dp.lz[dp["split"].isin(splits)]

    if emb_type == "imagenet":
        from .imagenet import embed_images as _embed_images

        return _embed_images(dp=dp, img_column=img_column, **kwargs)
    elif emb_type == "clip":
        from .clip import embed_images as _embed_images

        return _embed_images(dp=dp, img_column=img_column, **kwargs)
    elif emb_type == "bit":
        from .bit import embed_images as _embed_images

        return _embed_images(dp=dp, img_column=img_column, **kwargs)
    elif emb_type == "mimic_multimodal" or emb_type=="mimic_imageonly":
        from .mimic_multimodal import embed_images as _embed_images

        return _embed_images(dp=dp, img_column=img_column, file_path=file_path, **kwargs)
    else:
        raise ValueError(f"Embedding type '{emb_type}' not supported.")

import os
from typing import Callable, Union

import meerkat as mk
import torch

from ..registry import Registry
from .bit import bit
from .clip import clip

encoders = Registry(name="encoders")

encoders.register(clip, aliases=[])
encoders.register(bit, aliases=[])


def infer_modality(col: mk.AbstractColumn):

    if isinstance(col, mk.ImageColumn):
        return "image"
    elif isinstance(col, mk.PandasSeriesColumn):
        return "text"
    else:
        raise ValueError(f"Cannot infer modality from colummn of type {type(col)}.")


def embed(
    data: mk.DataPanel,
    input_col: str,
    encoder: str = "clip",
    modality: str = None,
    out_col: str = None,
    device: Union[int, str] = "cpu",
    mmap_dir: str = None,
    num_workers: int = 4,
    batch_size: int = 128,
    **kwargs,
) -> mk.DataPanel:
    """Embed a column of data with an encoder from the registry.

    .. note::

        You can see the encoders available in the registry with ``domino.encoders``.

    Examples
    --------
    Suppose you have an Image dataset (e.g. CIFAR-10) loaded into a
    `Meerkat DataPanel <https://github.com/robustness-gym/meerkat>`_. After loading the
    DataPanel, you can embed the images with clip using a code snippet like:

    .. code-block:: python

        import meerkat as mk
        from domino import embed

        dp = mk.datasets.get("cifar10")

        dp = embed(
            data=dp,
            input_col="image",
            encoder="clip"
        )


    Args:
        data (mk.DataPanel): DataPanel
        input_col (str): Name of column to embed.
        encoder (str, optional): Name of . Defaults to "clip".
        modality (str, optional): _description_. Defaults to None.
        out_col (str, optional): _description_. Defaults to None.
        device (Union[int, str], optional): _description_. Defaults to "cpu".
        mmap_dir (str, optional): _description_. Defaults to None.
        num_workers (int, optional): _description_. Defaults to 4.
        batch_size (int, optional): _description_. Defaults to 128.
        **kwargs: Additional keyword arguments to passed to the encoder.


    Raises:
        ValueError: _description_

    Returns:
        mk.DataPanel: _description_
    """

    if modality is None:

        modality = infer_modality(col=data[input_col])

    if out_col is None:
        out_col = f"{encoder}({input_col})"

    encoder = encoders.get(encoder, **kwargs)

    if modality not in encoder:
        raise ValueError(f'Encoder "{encoder}" does not support modality "{modality}".')

    encoder = encoder[modality]

    return _embed(
        data=data,
        input_col=input_col,
        out_col=out_col,
        encode=encoder.encode,
        preprocess=encoder.preprocess,
        collate=encoder.collate,
        device=device,
        mmap_dir=mmap_dir,
        num_workers=num_workers,
        batch_size=batch_size,
    )


def _embed(
    data: mk.DataPanel,
    input_col: str,
    out_col: str,
    encode: Callable,
    preprocess: Callable,
    collate: Callable,
    device: int = None,
    mmap_dir: str = None,
    num_workers: int = 4,
    batch_size: int = 128,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if preprocess is not None:
        embed_input = data[input_col].to_lambda(preprocess)
    else:
        embed_input = data[input_col]

    if collate is not None:
        embed_input.collate_fn = collate

    with torch.no_grad():
        data[out_col] = embed_input.map(
            lambda x: encode(x.data.to(device)).cpu().detach().numpy(),
            pbar=True,
            is_batched_fn=True,
            batch_size=batch_size,
            num_workers=num_workers,
            mmap=mmap_dir is not None,
            mmap_path=None
            if mmap_dir is None
            else os.path.join(mmap_dir, "emb_mmap.npy"),
            flush_size=128,
        )
    return data

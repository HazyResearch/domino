import meerkat as mk
import PIL
import pytest
import torch
import hashlib
import numpy as np

import domino
from domino import embed, encoders
from domino._embed.encoder import Encoder
from domino.registry import Registry

from ..testbeds import ImageColumnTestBed, TextColumnTestBed

EMB_SIZE = 4


def simple_encode(batch: torch.Tensor):
    value = batch.to(torch.float32).mean(dim=-1, keepdim=True)
    return torch.ones(batch.shape[0], EMB_SIZE) * value


def simple_image_transform(image: PIL.Image):
    return torch.tensor(np.asarray(image)).to(torch.float32)


def simple_text_transform(text: str):
    return torch.tensor(
        [
            int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest(), "big") % 100
            for token in text.split(" ")
        ]
    )[:1]


def _simple_encoder(variant: str = "ViT-B/32", device: str = "cpu"):
    return {
        "image": Encoder(encode=simple_encode, preprocess=simple_image_transform),
        "text": Encoder(encode=simple_encode, preprocess=simple_text_transform),
    }


@pytest.fixture()
def simple_encoder(monkeypatch):
    if "_simple_encoder" not in encoders.names:
        encoders.register(_simple_encoder)
    return _simple_encoder


def test_embed_images(tmpdir: str, simple_encoder):
    image_testbed = ImageColumnTestBed(tmpdir=tmpdir)

    dp = mk.DataPanel({"image": image_testbed.col})
    dp = embed(
        data=dp,
        input_col="image",
        encoder="_simple_encoder",
        batch_size=4,
        num_workers=0,
    )

    assert isinstance(dp, mk.DataPanel)
    assert "_simple_encoder(image)" in dp
    assert (
        simple_image_transform(dp["image"][0]).mean()
        == dp["_simple_encoder(image)"][0].mean()
    )


def test_embed_text(simple_encoder):
    testbed = TextColumnTestBed()

    dp = mk.DataPanel({"text": testbed.col})
    dp = embed(
        data=dp,
        input_col="text",
        encoder="_simple_encoder",
        batch_size=4,
        num_workers=0,
    )

    assert isinstance(dp, mk.DataPanel)
    assert "_simple_encoder(text)" in dp
    assert (
        simple_text_transform(dp["text"][0]).to(torch.float32).mean()
        == dp["_simple_encoder(text)"][0].mean()
    )


def test_encoders_repr():
    assert isinstance(domino.encoders, Registry)
    assert isinstance(domino.encoders.__repr__(), str)

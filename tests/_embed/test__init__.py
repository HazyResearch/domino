import meerkat as mk
import PIL
import pytest
import torch
import torchvision as tv
from clip import tokenize

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
    return tv.transforms.ToTensor()(image)


def _simple_encoder(variant: str = "ViT-B/32"):
    return {
        "image": Encoder(encode=simple_encode, preprocess=simple_image_transform),
        "text": Encoder(encode=simple_encode, preprocess=tokenize),
    }


@pytest.fixture()
def simple_encoder(monkeypatch):
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
        tv.transforms.ToTensor()(dp["image"][0]).mean()
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
        tokenize(dp["text"][0]).to(torch.float32).mean()
        == dp["_simple_encoder(text)"][0].mean()
    )


def test_encoders_repr():
    assert isinstance(domino.encoders, Registry)
    assert isinstance(domino.encoders.__repr__(), str)

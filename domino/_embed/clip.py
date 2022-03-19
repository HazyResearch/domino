from typing import Union

from clip import load, tokenize

from .encoder import Encoder


def clip(variant: str = "ViT-B/32", device: Union[int, str] = "cpu"):
    model, preprocess = load(variant, device=device)
    return {
        "image": Encoder(encode=model.encode_image, preprocess=preprocess),
        "text": Encoder(encode=model.encode_text, preprocesser=tokenize),
    }

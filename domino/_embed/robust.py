from asyncio import subprocess
from typing import Dict, Union

from .encoder import Encoder


def robust(
    variant: str = "ViT-B/32", device: Union[int, str] = "cpu"
) -> Dict[str, Encoder]:
    """Contrastive Language-Image Pre-training (CLIP) encoders [radford_2021]_. Includes
    encoders for the following modalities:

    - "text"
    - "image"

    Encoders will map these different modalities to the same embedding space.

    Args:
        variant (str, optional): A model name listed by `clip.available_models()`, or
            the path to a model checkpoint containing the state_dict. Defaults to
            "ViT-B/32".
        device (Union[int, str], optional): The device on which the encoders will be
            loaded. Defaults to "cpu".


    .. [radford_2021]

        Radford, A. et al. Learning Transferable Visual Models From Natural Language
        Supervision. arXiv [cs.CV] (2021)
    """


    # download model weights 
    subprocess.run(
        [
            "wget", 
            "-O", 
            os.path.expanduser("~/.cache/robust/robust_resnet50.pth"),
            "https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=0"
        ]
    )


    model, preprocess = load(variant, device=device)
    return {
        "image": Encoder(encode=model.encode_image, preprocess=preprocess),
        "text": Encoder(
            # need to squeeze out the batch dimension for compatibility with collate
            encode=model.encode_text, preprocess=lambda x: tokenize(x, truncate=True).squeeze(0)
        ),
    }

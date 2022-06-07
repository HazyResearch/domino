from ast import Import
import subprocess
from typing import Dict, Union
import os 

from .encoder import Encoder


def robust(
    variant: str = "resnet50", device: Union[int, str] = "cpu", model_path: str=None
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

    model_path = os.path.expanduser("~/.cache/domino/robust/robust_resnet50.pth") if model_path is None else model_path
    model = _load_robust_model(model_path=model_path, variant=variant).to(device)

    return {
        "image": Encoder(encode=lambda x: model(x, with_latent=True)[0][1] , preprocess=_transform_image),
    }

def _load_robust_model(model_path: str, variant: str):
    try:
        from robustness import model_utils
        from robustness import datasets as dataset_utils
    except ImportError:
        raise ImportError(
            "To embed with robust run `pip install robustness`"
        )

    # ensure model_path directories exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    subprocess.run(
        [
            "wget", 
            "-O", 
            model_path,
            "https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=0"
        ]
    )

    dataset_function = getattr(dataset_utils, 'ImageNet')
    dataset = dataset_function('')

    model_kwargs = {
        'arch': variant,
        'dataset': dataset,
        'resume_path': model_path,
        'parallel': False
    }
    model, _ = model_utils.make_and_restore_model(**model_kwargs)
    model.eval()
    return model


def _transform_image(img):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])(img)
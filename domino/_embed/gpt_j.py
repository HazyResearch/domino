from typing import Dict, Union

from .encoder import Encoder


def transformers(
    variant: str = "bert-large-cased", device: Union[int, str] = "cpu"
) -> Dict[str, Encoder]:
    """Contrastive Language-Image Pre-training (CLIP) encoders [radford_2021]_. Includes
    encoders for the following modalities:

    - "text"

    Encoders will map these different modalities to the same embedding space.

    Args:
        variant (str, optional): A model name listed by `clip.available_models()`, or
            the path to a model checkpoint containing the state_dict. Defaults to
            "ViT-B/32".
        device (Union[int, str], optional): The device on which the encoders will be
            loaded. Defaults to "cpu".


    @misc{gpt-j,
        author = {Wang, Ben and Komatsuzaki, Aran},
        title = {{GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model}},
        howpublished = {\url{https://github.com/kingoflolz/mesh-transformer-jax}},
        year = 2021,
        month = May
    }
    @misc{mesh-transformer-jax,
        author = {Wang, Ben},
        title = {{Mesh-Transformer-JAX: Model-Parallel Implementation of Transformer Language Model with JAX}},
        howpublished = {\url{https://github.com/kingoflolz/mesh-transformer-jax}},
        year = 2021,
        month = May
    }
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        raise ImportError(
            "To embed with CLIP run pip install git+https://github.com/openai/CLIP.git"
            "and install domino with the `clip` submodule. For example, "
            "`pip install domino[clip]`"
        )

    tokenizer = AutoTokenizer.from_pretrained(variant)
    model = AutoModelForSequenceClassification.from_pretrained(variant)

    model.to(device)

    def _encode(x: List[str]) -> torch.Tensor:
        return model(**tokenizer(x, return_tensors="pt", padding=True).to(device=device)).last_hidden_state[:, 0]

    model, preprocess = load(variant, device=device)
    return {
        "text": Encoder(
            # need to squeeze out the batch dimension for compatibility with collate
            encode=_encode, preprocess=lambda x: x
        ),
    }

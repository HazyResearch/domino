from typing import Union
import meerkat as mk
import numpy as np
from scipy.stats import mode

from domino.utils import unpack_args


def describe(
    data: mk.DataPanel = None,
    embeddings: Union[str, np.ndarray] = "embedding",
    targets: Union[str, np.ndarray] = "target",
    slices: Union[str, np.ndarray] = "slices",
    text: mk.DataPanel = None,
    text_embeddings: Union[str, np.ndarray] = "embedding",
    phrases: Union[str, np.ndarray] = "output_phrase",
    slice_idx: int = 0,
    slice_threshold: float = 0.5,
) -> mk.DataPanel:
    """Generate descriptions of a discovered slice. 

    Args:
        data (mk.DataPanel, optional): A `Meerkat DataPanel` with columns for
            embeddings, targets, and prediction probabilities. The names of the
            columns can be specified with the ``embeddings``, ``targets``, and
            ``pred_probs`` arguments. Defaults to None.
        embeddings (Union[str, np.ndarray], optional): The name of a column in
            ``data`` holding embeddings. If ``data`` is ``None``, then an np.ndarray
            of shape (n_samples, dimension of embedding). Defaults to
            "embedding".
        targets (Union[str, np.ndarray], optional): The name of a column in
            ``data`` holding class labels. If ``data`` is ``None``, then an
            np.ndarray of shape (n_samples,). Defaults to "target".
        pred_probs (Union[str, np.ndarray], optional): The name of
            a column in ``data`` holding model predictions (can either be "soft"
            probability scores or "hard" 1-hot encoded predictions). If
            ``data`` is ``None``, then an np.ndarray of shape (n_samples, n_classes)
            or (n_samples,) in the binary case. Defaults to "pred_probs".
        slices (str, optional): The name of The name of a column in ``data``
            holding discovered slices. If ``data`` is ``None``, then an
            np.ndarray of shape (num_examples, num_slices). Defaults to "slices".
        text (str, optional): A `Meerkat DataPanel` with columns for text phrases and
            their embeddings. The names of the columns can be specified with the 
            ``text_embeddings`` and ``phrase`` arguments. Defaults to None.
        text_embeddings (Union[str, np.ndarray], optional): The name of a colum in
            ``text`` holding embeddings. If ``text`` is ``None``, then an np.ndarray
            of shape (n_phrases, dimension of embedding). Defaults to "embedding".
        phrase (Union[str, np.ndarray], optional): The name of a column in ``text``
            holding text phrases. If ``text`` is ``None``, then an np.ndarray of
            shape (n_phrases,). Defaults to "output_phrase".
        slice_idx (int, optional): The index of the slice to describe. Defaults to 0.
        slice_threshold (float, optional): The probability threshold for inclusion in 
            the slice. Defaults to 0.5.

    Returns:
        mk.DataPanel: A `Meerkat DataPanel` with columns for the slice description.

    
    Examples
    --------
     .. code-block:: python
        :name: Example:

        from domino import describe, generate_candidate_descriptions

        templates = [
            "a photo of [MASK].",
            "a photo of {} [MASK].",
            "a photo of [MASK] {}.",
            "a photo of [MASK] {} [MASK].",
        ]

        text_dp = generate_candidate_descriptions(templates=templates)

        text_dp = embed(
            text_dp, 
            input_col="output_phrase", 
            encoder="clip",
            device=0
        )

        describe(
            data=dp,
            embeddings="clip(image)",
            pred_probs="prob",
            targets="target",
            slices="domino_slices",
            text=text_dp,
            text_embeddings="clip(output_phrase)",
        )

    
    """

    embeddings, targets, slices = unpack_args(data, embeddings, targets, slices)
    text_embeddings, phrases = unpack_args(text, text_embeddings, phrases)

    slice_mask = slices[:, slice_idx] > slice_threshold
    slice_proto = embeddings[slice_mask].mean(axis=0)
    mode_target = mode(targets[slice_mask].data).mode[0]
    ref_proto = embeddings[targets == mode_target].mean(axis=0)

    scores = np.dot(text_embeddings, (slice_proto - ref_proto))
    return mk.DataPanel({"score": scores, "phrase": phrases})

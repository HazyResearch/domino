from typing import Dict, Union, Tuple, List
from domino._describe.abstract import Describer
from domino.utils import unpack_args

import meerkat as mk
import numpy as np

from ._slice.abstract import Slicer
from ._slice.mixture import MixtureSlicer
from ._embed import embed
from ._describe.abstract import Describer
from ._describe.mean import MeanDescriber


def discover(
    data: Union[dict, mk.DataPanel] = None,
    embeddings: Union[str, np.ndarray] = "embedding",
    targets: Union[str, np.ndarray] = "target",
    pred_probs: Union[str, np.ndarray] = "pred_probs",
    losses: Union[str, np.ndarray] = "loss",
    split: Union[str, np.ndarray] = "split",
    slicer: Slicer = None,
    describer: Describer = None,
) -> Tuple[np.ndarray, List[Dict]]:

    embeddings, targets, pred_probs, losses, split = unpack_args(
        data, embeddings, targets, pred_probs, losses, split
    )

    if embeddings is None:
        raise NotImplementedError

    if slicer is None:
        # provide a simple default slicer if none is provided
        slicer = MixtureSlicer(
            n_slices=5,
            y_log_likelihood_weight=10,
            y_hat_log_likelihood_weight=10,
        )

    train_mask = (split != "test")
    slicer.fit(
        targets=targets[train_mask] if targets is not None else None,
        pred_probs=pred_probs[train_mask] if pred_probs is not None else None,
        losses=losses[train_mask] if losses is not None else None,
        embeddings=embeddings[train_mask] if embeddings is not None else None,
    )

    test_mask = ~train_mask
    slices = slicer.predict_proba(
        targets=None,
        pred_probs=None,
        #losses=losses[test_mask], 
        embeddings=embeddings[test_mask] if embeddings is not None else None,
    )

    if describer is None:
        raise NotImplementedError

    descriptions = describer.describe(
        embeddings=embeddings[test_mask], targets=None, slices=slices
    )

    return slices, descriptions

from typing import Union

import meerkat as mk 
import numpy as np 
from scipy.stats import mode, pearsonr


from .abstract import Describer
from ..utils import unpack_args


class MeanDescriber(Describer):
    """

    Args:
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

    """

    def __init__(
        self,
        data: mk.DataPanel = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        candidates: Union[str, np.ndarray] = "candidates",
        slice_threshold: float = 0.5,
        n_descriptions: int = 10,
    ):
        super().__init__()
        embeddings, candidates = unpack_args(data, embeddings, candidates)

        self.candidates = candidates
        self.candidate_embeddings = embeddings
        self.config.slice_threshold = slice_threshold
        self.config.n_descriptions = n_descriptions
    
    def describe(
        self,
        data: mk.DataPanel = None, 
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        slices: Union[str, np.ndarray] = "slices",
    ):

        embeddings, targets, slices = unpack_args(data, embeddings, targets, slices)

        result = []
        for slice_idx in range(slices.shape[-1]):

            slice_mask = slices[:, slice_idx] > self.config.slice_threshold
            slice_proto = embeddings[slice_mask].mean(axis=0)
            ref_proto = embeddings.mean(axis=0)

            scores = np.dot(self.candidate_embeddings, (slice_proto - ref_proto))
            idxs = np.argsort(-scores)[:self.config.n_descriptions]
            
            selected_embeddings = self.candidate_embeddings[idxs]
            selected_scores = np.dot(selected_embeddings, embeddings.T)
            slice_scores = slices[:, slice_idx]

            result.append(
                [
                    {
                        "text": self.candidates[idx],
                        "score": scores[idx],
                        "corr": pearsonr(slice_scores, selected_scores[i])[0],
                    }
                    for i, idx in enumerate(idxs)
                ]
            )

        return result



class ClassifierMeanDescriber(Describer):
    """

    Args:
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

    """

    def __init__(
        self,
        data: mk.DataPanel = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        candidates: Union[str, np.ndarray] = "candidates",
        slice_threshold: float = 0.5,
        n_descriptions: int = 10,
    ):
        super().__init__()
        embeddings, candidates = unpack_args(data, embeddings, candidates)

        self.candidates = candidates
        self.candidate_embeddings = embeddings
        self.config.slice_threshold = slice_threshold
        self.config.n_descriptions = n_descriptions
    
    def describe(
        self,
        data: mk.DataPanel = None, 
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        slices: Union[str, np.ndarray] = "slices",
    ):

        embeddings, targets, slices = unpack_args(data, embeddings, targets, slices)

        result = []
        for slice_idx in range(slices.shape[-1]):

            slice_mask = slices[:, slice_idx] > self.config.slice_threshold
            slice_proto = embeddings[slice_mask].mean(axis=0)
            mode_target = mode(targets[slice_mask]).mode[0]
            ref_proto = embeddings[targets == mode_target].mean(axis=0)

            scores = np.dot(self.candidate_embeddings, (slice_proto - ref_proto))
            idxs = np.argsort(-scores)[:self.config.n_descriptions]
        
            result.append(
                [
                    {
                        "text": self.candidates[idx],
                        "score": scores[idx],
                    }
                    for idx in idxs
                ]
            )

        return result

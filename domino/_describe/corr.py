from typing import Union

import meerkat as mk
import numpy as np
import torch
from scipy.stats import mode, pearsonr


from .abstract import Describer
from ..utils import convert_to_torch, unpack_args


class CorrDescriber(Describer):
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
        img_embs, slices, text_embs = convert_to_torch(
            embeddings, slices, self.candidate_embeddings
        )

        with torch.no_grad():
            text_scores = torch.matmul(img_embs.to(0), text_embs.to(0).T)

        r = batched_pearsonr(
            x=text_scores.to(torch.float).T, y=slices.to(torch.float).to(0).T
        )

        result = []
        for pred_slice_idx in range(r.shape[-1]):
            slice_scores = r[:, pred_slice_idx].cpu().detach().numpy()
            idxs = np.argsort(-slice_scores)[:10]
            result.append(
                [
                    {
                        "pred_slice_idx": pred_slice_idx,
                        "scores": slice_scores[idx],
                        "corr": slice_scores[idx],
                        "text": self.candidates[idx],
                    }
                    for idx in idxs
                ]
            )
        return result


@torch.no_grad()
def batched_pearsonr(x, y, batch_first=True):

    if len(x.shape) - len(y.shape) == 1:
        y = y.unsqueeze(-1)

    centered_x = x - x.mean(dim=1, keepdim=True)
    centered_y = y - y.mean(dim=1, keepdim=True)
    covariance = centered_x @ centered_y.T  # x_batch x y_batch

    bessel_corrected_covariance = covariance / (x.shape[1] - 1)

    x_std = x.std(dim=1, keepdim=True)
    y_std = y.std(dim=1, keepdim=True)
    std = x_std @ y_std.T
    corr = bessel_corrected_covariance / std

    return corr

from typing import Dict, List, Mapping, Sequence

import meerkat as mk
import numpy as np
import terra
from torchvision import transforms
from tqdm import tqdm

from . import CorrelationImpossibleError, induce_correlation, synthesize_preds


@terra.Task
def build_slice(
    slice_category: str,
    dp: mk.DataPanel,
    synthetic_preds: bool = False,
    synthetic_kwargs: Mapping[str, object] = None,
    **kwargs,
) -> mk.DataPanel:
    if slice_category == "correlation":
        dp = build_correlation_slice(dp, **kwargs)
    elif slice_category == "rare":
        raise NotImplementedError

    if synthetic_preds:
        synthetic_kwargs = {} if synthetic_kwargs is None else synthetic_kwargs
        dp["pred"] = synthesize_preds(dp, **synthetic_kwargs)

    return dp


@terra.Task
def build_correlation_slice(
    dp: mk.DataPanel,
    correlate: str,
    corr: float,
    n: int,
    correlate_threshold: float = None,
    **kwargs,
) -> mk.DataPanel:

    if correlate_threshold:
        dp[f"binarized_{correlate}"] = (
            dp[correlate].data > correlate_threshold
        ).astype(int)
        correlate = f"binarized_{correlate}"

    indices = induce_correlation(
        dp=dp,
        corr=corr,
        attr_a="binary_sz",
        attr_b=correlate,
        match_mu=True,
        n=n,
    )

    dp = dp.lz[indices]

    return dp

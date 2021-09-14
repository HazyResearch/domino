from typing import Dict, List, Mapping, Sequence

import meerkat as mk
import numpy as np
import terra
from meerkat.contrib.eeg import build_stanford_eeg_dp
from torchvision import transforms
from tqdm import tqdm

from . import CorrelationImpossibleError, induce_correlation, synthesize_preds


@terra.Task
def build_slice(
    slice_category: str,
    dp_run_id: int,
    split_dp: mk.DataPanel,
    synthetic_preds: bool = False,
    synthetic_kwargs: Mapping[str, object] = None,
    **kwargs,
) -> mk.DataPanel:
    if slice_category == "correlation":
        dp = build_correlation_slice(dp_run_id, **kwargs)
    elif slice_category == "rare":
        raise NotImplementedError

    if synthetic_preds:
        synthetic_kwargs = {} if synthetic_kwargs is None else synthetic_kwargs
        dp["pred"] = synthesize_preds(dp_run_id, **synthetic_kwargs)

    return dp.merge(split_dp, on=["id", "patient_id"])


def build_correlation_slice(
    dp_run_id: int,
    correlate: str,
    corr: float,
    n: int,
    correlate_threshold: float = None,
    **kwargs,
) -> mk.DataPanel:

    dp = build_stanford_eeg_dp.out(run_id=dp_run_id, load=True)

    if correlate_threshold:
        dp[f"binarized_{correlate}"] = (
            dp[correlate].data > correlate_threshold
        ).astype(int)
        correlate = f"binarized_{correlate}"

    indices = induce_correlation(
        dp=dp,
        corr=corr,
        attr_a="target",
        attr_b=correlate,
        match_mu=True,
        n=n,
    )

    dp = dp.lz[indices]

    return dp


@terra.Task
def collect_correlation_slices(
    dp_run_id: int,
    correlate_list: List[str],
    corr_list: List[float],
    correlate_thresholds: List[float] = None,
    n: int = 2500,
) -> mk.DataPanel:

    settings = []
    for ndx, correlate in enumerate(correlate_list):
        for corr in corr_list:
            settings.append(
                {
                    "slice_category": "correlation",
                    "dp_run_id": dp_run_id,
                    "correlate": correlate,
                    "corr": corr,
                    "correlate_threshold": correlate_thresholds[ndx],
                    "n": n,
                }
            )

    return mk.DataPanel(settings)

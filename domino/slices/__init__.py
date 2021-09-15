from typing import Iterable, Tuple, Union

import meerkat as mk
import numpy as np
import pandas as pd


class CorrelationImpossibleError(ValueError):
    def __init__(
        self,
        corr: float,
        n: int,
        attr_a: str,
        attr_b: str,
        mu_a: float,
        mu_b: float,
        msg: str,
    ):
        super().__init__(
            f"Cannot achieve correlation of {corr} while creating sample with {int(n)} "
            f"examples and means of {mu_a:0.3f} and {mu_b:0.3f} for attributes {attr_a} and "
            f"{attr_b} respectively. " + msg
        )


def induce_correlation(
    dp: Union[pd.DataFrame, mk.DataPanel],
    corr: float,
    n: int,
    attr_a: str,
    attr_b: str,
    mu_a: float = None,
    mu_b: float = None,
    match_mu: bool = False,
    replace: bool = False,
):
    """
    Induce a correlation `corr` between two boolean columns `attr_a` and `attr_b` by
    subsampling `df`, while maintaining mean and variance. If `match_mu` is `True` then
    take the minimum mean among the two attributes and use it for both.
    Details: https://www.notion.so/Slice-Discovery-Evaluation-Framework-63b625318ef4411698c5e369d914db88#8bd2da454826451c80b524149e1c87cc
    """
    if mu_a is None:
        mu_a = dp[attr_a].mean()

    if mu_b is None:
        mu_b = dp[attr_b].mean()

    if match_mu:
        mu = min(mu_a, mu_b)
        mu_a, mu_b = mu, mu

    var_a = (mu_a) * (1 - mu_a)
    var_b = (mu_b) * (1 - mu_b)
    n_a1 = mu_a * n
    n_b1 = mu_b * n

    n_1 = (n_a1 * n_b1 / n) + corr * np.sqrt(var_a * var_b) * (n - 1)
    n_0 = n - (n_a1 + n_b1 - n_1)

    n_a1_b0 = n_a1 - n_1
    n_a0_b1 = n_b1 - n_1

    both_1 = (dp[attr_a] == 1) & (dp[attr_b] == 1)
    both_0 = (dp[attr_a] == 0) & (dp[attr_b] == 0)

    # check if requested correlation is possible
    msg = None
    if int(n_a1) > dp[attr_a].sum():
        msg = "Not enough samples where a=1. Try a lower mu_a."
    elif int(n_b1) > dp[attr_b].sum():
        msg = "Not enough samples where b=1. Try a lower mu_b."
    elif int(n_1) > both_1.sum():
        msg = "Not enough samples where a=1 and b=1. Try a lower corr or smaller n."
    elif int(n_0) > both_0.sum():
        msg = "Not enough samples where a=0 and b=0. Try a lower corr or smaller n."
    elif int(n_a1_b0) > (dp[attr_a] & (1 - both_1)).sum():
        msg = "Not enough samples where a=1 and b=0. Try a higher corr or smaller n."
    elif int(n_a0_b1) > (dp[attr_b] & (1 - both_1)).sum():
        msg = "Not enough samples where a=0 and b=1. Try a higher corr or smaller n."
    elif n_1 < 0:
        msg = "Insufficient variance for desired corr. Try mu_a or mu_b closer to 0.5 "
    elif n_0 < 0:
        msg = "ahh"
    elif (n_1 > n_a1) or (n_1 > n_b1) or n_1 < 0 or n_0 < 0:
        msg = "Not enough samples where a=0 and b=0. Try a lower corr or smaller n."
    if msg is not None:
        raise CorrelationImpossibleError(corr, n, attr_a, attr_b, mu_a, mu_b, msg)

    indices = []
    indices.extend(
        np.random.choice(np.where(both_1)[0], size=int(n_1), replace=replace)
    )
    indices.extend(
        np.random.choice(
            np.where(dp[attr_a] & (1 - both_1))[0], size=int(n_a1_b0), replace=replace
        )
    )
    indices.extend(
        np.random.choice(
            np.where(dp[attr_b] & (1 - both_1))[0], size=int(n_a0_b1), replace=replace
        )
    )
    indices.extend(
        np.random.choice(
            np.where(both_0)[0],
            size=int(n_0),
            replace=replace,
        )
    )
    np.random.shuffle(indices)
    return indices


def synthesize_preds(
    dp: mk.DataPanel,
    sensitivity: float = 0.8,
    specificity: float = 0.8,
    slice_sensitivities: Union[float, Iterable[float]] = None,
    slice_specificities: Union[float, Iterable[float]] = None,
):
    n_slices = dp["slices"].shape[-1]

    if slice_sensitivities is None:
        slice_sensitivities = [0.5] * n_slices
    elif isinstance(slice_sensitivities, float):
        slice_sensitivities = [slice_sensitivities] * n_slices

    if slice_specificities is None:
        slice_specificities = [0.5] * n_slices
    elif isinstance(slice_specificities, float):
        slice_specificities = [slice_specificities] * n_slices

    if len(slice_sensitivities) != n_slices or len(slice_specificities) != n_slices:
        raise ValueError(
            "Both `slice_sensitivies` and `slice_specifities` should be "
            "iterables with length equal to the number of slices."
        )

    preds = np.zeros(len(dp))
    preds[dp["target"] == 1] = np.random.beta(
        sensitivity, 1 - sensitivity, dp["target"].sum()
    )
    preds[dp["target"] == 0] = np.random.beta(
        1 - specificity, specificity, (1 - dp["target"]).sum()
    )
    for slice_idx in range(n_slices):
        mask = (dp["target"] == 1) & (dp["slices"][:, slice_idx] == 1)
        preds[mask] = np.random.beta(
            slice_sensitivities[slice_idx],
            1 - slice_sensitivities[slice_idx],
            mask.sum(),
        )
        mask = (dp["target"] == 0) & (dp["slices"][:, slice_idx] == 1)
        preds[mask] = np.random.beta(
            1 - slice_specificities[slice_idx],
            slice_specificities[slice_idx],
            mask.sum(),
        )

    return preds

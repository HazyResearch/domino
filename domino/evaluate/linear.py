from typing import Union

import mosaic as ms
import numpy as np
import pandas as pd


def induce_correlation(
    df: Union[pd.DataFrame, ms.DataPanel],
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
        mu_a = df[attr_a].mean()

    if mu_b is None:
        mu_b = df[attr_b].mean()

    if match_mu:
        mu = min(mu_a, mu_b)
        mu_a, mu_b = mu, mu

    var_a = (mu_a) * (1 - mu_a)
    var_b = (mu_b) * (1 - mu_b)
    n_a1 = mu_a * n
    n_b1 = mu_b * n

    n_1 = (n_a1 * n_b1 / n) + corr * np.sqrt(var_a * var_b * n ** 2)

    if (n_1 > n_a1) or (n_1 > n_b1) or n_1 < 0:
        raise ValueError(
            f"Cannot achieve correlation of {corr} while maintaining means for "
            f"attributes {attr_a=} and {attr_b=}."
        )

    both1 = (df[attr_a] == 1) & (df[attr_b] == 1)
    indices = []
    indices.extend(np.random.choice(np.where(both1)[0], size=int(n_1), replace=replace))
    indices.extend(
        np.random.choice(
            np.where(df[attr_a] & (1 - both1))[0], size=int(n_a1 - n_1), replace=replace
        )
    )

    indices.extend(
        np.random.choice(
            np.where(df[attr_b] & (1 - both1))[0], size=int(n_b1 - n_1), replace=replace
        )
    )

    indices.extend(
        np.random.choice(
            np.where((df[attr_a] == 0) & (df[attr_b] == 0))[0],
            size=n - len(indices),
            replace=replace,
        )
    )

    return indices

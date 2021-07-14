import hashlib
from functools import reduce, wraps
from inspect import getcallargs
from typing import Collection, Dict, Mapping, Optional, Sequence, Union

import meerkat as mk
import numpy as np
import pandas as pd
import torch
from cytoolz import concat
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only
from sklearn.metrics import roc_auc_score
from terra import Task
from torchmetrics import Metric


def requires_columns(dp_arg: str, columns: Collection[str]):
    def _requires(fn: callable):
        @wraps(fn)
        def _wrapper(*args, aliases: Mapping[str, str] = None, **kwargs):
            args_dict = getcallargs(fn, *args, **kwargs)
            if "kwargs" in args_dict:
                args_dict.update(args_dict.pop("kwargs"))

            dp = args_dict[dp_arg]
            if aliases is not None:
                dp = dp.view()
                for column, alias in aliases.items():
                    dp[column] = dp[alias]

            missing_cols = [column for column in columns if column not in dp]
            if len(missing_cols) > 0:
                raise ValueError(
                    f"DataPanel passed to `{fn.__qualname__}` at argument `{dp_arg}` "
                    f"is missing required columns `{missing_cols}`."
                )
            return fn(*args, **kwargs)

        return _wrapper

    return _requires


def nested_getattr(obj, attr, *args):
    """Get a nested property from an object.
    Source: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    """
    return reduce(lambda o, a: getattr(o, a, *args), [obj] + attr.split("."))


def hash_for_split(example_id: str, salt=""):
    GRANULARITY = 100000
    hashed = hashlib.sha256((str(example_id) + salt).encode())
    hashed = int(hashed.hexdigest().encode(), 16) % GRANULARITY + 1
    return hashed / float(GRANULARITY)


def place_on_gpu(data, device=0):
    """
    Recursively places all 'torch.Tensor's in data on gpu and detaches.
    If elements are lists or tuples, recurses on the elements. Otherwise it
    ignores it.
    source: inspired by place_on_gpu from Snorkel Metal
    https://github.com/HazyResearch/metal/blob/master/metal/utils.py
    """
    data_type = type(data)
    if data_type in (list, tuple):
        data = [place_on_gpu(data[i], device) for i in range(len(data))]
        data = data_type(data)
        return data
    elif data_type is dict:
        data = {key: place_on_gpu(val, device) for key, val in data.items()}
        return data
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


class PredLogger(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state(
            "sample_ids",
            default=[],
            dist_reduce_fx=None,
        )

    def update(
        self, pred: torch.Tensor, target: torch.Tensor, sample_id: Union[str, int]
    ):
        self.preds.append(pred.detach())
        self.targets.append(target.detach())
        self.sample_ids.extend(sample_id)

    def compute(self):
        """TODO: this sometimes returns duplicates."""
        if torch.is_tensor(self.sample_ids[0]):
            sample_ids = torch.cat(self.sample_ids).cpu()
        else:
            # support for string ids
            sample_ids = self.sample_ids
        preds = torch.cat(self.preds).cpu()
        targets = torch.cat(self.targets).cpu()

        return {"preds": preds, "targets": targets, "sample_ids": sample_ids}

    def _apply(self, fn):
        """
        https://github.com/PyTorchLightning/metrics/blob/fb0ee3ff0509fdb13bd07b6aac3e20c642bb5683/torchmetrics/metric.py#L280
        """
        this = super(Metric, self)._apply(fn)
        # Also apply fn to metric states
        for key in this._defaults.keys():
            current_val = getattr(this, key)
            if isinstance(current_val, torch.Tensor):
                setattr(this, key, fn(current_val))
            elif isinstance(current_val, Sequence):
                if (
                    len(current_val) > 0 and isinstance(current_val[0], tuple)
                ) or key == "sample_ids":
                    # avoid calling `.to`, `.cpu`, `.cuda` on string metric states
                    continue

                setattr(this, key, [fn(cur_v) for cur_v in current_val])
            else:
                raise TypeError(
                    "Expected metric state to be either a Tensor"
                    f"or a list of Tensor, but encountered {current_val}"
                )
        return this


class TerraCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_validation_end(self, trainer, pl_module):
        """
        checkpoints can be saved at the end of the val loop
        """
        self.save_checkpoint(trainer, pl_module)
        pl_module.valid_preds.compute()  # needed for automatic reset

    @rank_zero_only
    def _save_model(self, trainer, filepath: str):
        # only dump on rank 0,  see comment on `save_checkpoint` which instructs the
        # save_function to only save on rank 0, like https://github.com/PyTorchLightning/pytorch-lightning/blob/af621f8590b2f2ba046b508da2619cfd4995d876/pytorch_lightning/trainer/training_io.py#L256-L267
        # we use rank_zero_only as recommended by https://github.com/PyTorchLightning/pytorch-lightning/issues/2267#issuecomment-646602749
        lit_module = trainer.lightning_module
        Task.dump(
            {
                "current_epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "model": lit_module,
                "valid": lit_module.valid_preds.compute(),
            },
            run_dir=self.dirpath,
            group_name="best_chkpt",
            overwrite=True,
        )


def auroc_bootstrap_ci(
    targets: np.ndarray, scores: np.ndarray, num_iter: int = 10000, alpha: float = 0.05
):
    pos_indices = np.where(targets == 1)[0]
    neg_indices = np.where(targets == 0)[0]
    pos_samples = np.random.choice(pos_indices, (len(pos_indices), num_iter))
    neg_samples = np.random.choice(neg_indices, (len(neg_indices), num_iter))
    bs_samples = np.concatenate([pos_samples, neg_samples], axis=0)

    def estimator(sample):
        return roc_auc_score(targets[sample], scores[sample])

    bs_sample_estimates = np.apply_along_axis(estimator, axis=0, arr=bs_samples)
    sample_estimate = roc_auc_score(targets, scores)

    return {
        "auroc": sample_estimate,
        "auroc_lower": np.percentile(bs_sample_estimates, alpha * 100),
        "auroc_upper": np.percentile(bs_sample_estimates, 100 * (1 - alpha)),
    }


def compute_bootstrap_ci(
    sample: np.ndarray,
    num_iter: int = 10000,
    alpha: float = 0.05,
    estimator: Union[callable, str] = "mean",
):
    """Compute an empirical confidence using bootstrap resampling."""
    bs_samples = np.random.choice(sample, (sample.shape[0], num_iter))
    if estimator == "mean":
        bs_sample_estimates = bs_samples.mean(axis=0)
        sample_estimate = sample.mean(axis=0)
    else:
        bs_sample_estimates = np.apply_along_axis(estimator, axis=0, arr=bs_samples)
        sample_estimate = estimator(sample)

    return {
        "sample_estimate": sample_estimate,
        "lower": np.percentile(bs_sample_estimates, alpha * 100),
        "upper": np.percentile(bs_sample_estimates, 100 * (1 - alpha)),
    }


def format_ci(df: pd.DataFrame):
    return df.apply(lambda x: f"{x.sample_mean:0.} ({x.lower}, {x.upper})", axis=1)


def batched_pearsonr(x, y, batch_first=True):
    r"""Computes Pearson Correlation Coefficient across rows.
    Pearson Correlation Coefficient (also known as Linear Correlation
    Coefficient or Pearson's :math:`\rho`) is computed as:
    .. math::
        \rho = \frac {E[(X-\mu_X)(Y-\mu_Y)]} {\sigma_X\sigma_Y}
    If inputs are matrices, then then we assume that we are given a
    mini-batch of sequences, and the correlation coefficient is
    computed for each sequence independently and returned as a vector. If
    `batch_fist` is `True`, then we assume that every row represents a
    sequence in the mini-batch, otherwise we assume that batch information
    is in the columns.
    Warning:
        We do not account for the multi-dimensional case. This function has
        been tested only for the 2D case, either in `batch_first==True` or in
        `batch_first==False` mode. In the multi-dimensional case,
        it is possible that the values returned will be meaningless.
    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): target tensor
        batch_first (bool, optional): controls if batch dimension is first.
            Default: `True`
    Returns:
        torch.Tensor: correlation coefficient between `x` and `y`
    Note:
        :math:`\sigma_X` is computed using **PyTorch** builtin
        **Tensor.std()**, which by default uses Bessel correction:
        .. math::
            \sigma_X=\displaystyle\frac{1}{N-1}\sum_{i=1}^N({x_i}-\bar{x})^2
        We therefore account for this correction in the computation of the
        covariance by multiplying it with :math:`\frac{1}{N-1}`.
    Shape:
        - Input: :math:`(N, M)` for correlation between matrices,
          or :math:`(M)` for correlation between vectors
        - Target: :math:`(N, M)` or :math:`(M)`. Must be identical to input
        - Output: :math:`(N, 1)` for correlation between matrices,
          or :math:`(1)` for correlation between vectors
    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> input = torch.rand(3, 5)
        >>> target = torch.rand(3, 5)
        >>> output = pearsonr(input, target)
        >>> print('Pearson Correlation between input and target is {0}'.format(output[:, 0]))
        Pearson Correlation between input and target is tensor([ 0.2991, -0.8471,  0.9138])
    """  # noqa: E501

    if len(x.shape) - len(y.shape) == 1:
        y = y.unsqueeze(-1)

    if batch_first:
        dim = -1
    else:
        dim = 0

    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)

    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr

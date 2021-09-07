from typing import Union

import meerkat as mk
import numpy as np
from sklearn.metrics import roc_auc_score

from domino.utils import flatten_dict, requires_columns


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
        "mean": sample_estimate,
        "lower": np.percentile(bs_sample_estimates, alpha * 100),
        "upper": np.percentile(bs_sample_estimates, 100 * (1 - alpha)),
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
        estimator: sample_estimate,
        "lower": np.percentile(bs_sample_estimates, alpha * 100),
        "upper": np.percentile(bs_sample_estimates, 100 * (1 - alpha)),
    }


def recall_bootstrap_ci(
    targets: np.ndarray, preds: np.ndarray, num_iter: int = 10000, alpha: float = 0.05
):
    return compute_bootstrap_ci(preds[targets == 1], num_iter=num_iter, alpha=alpha)


def precision_bootstrap_ci(
    targets: np.ndarray, preds: np.ndarray, num_iter: int = 10000, alpha: float = 0.05
):
    return compute_bootstrap_ci(targets[preds == 1], num_iter=num_iter, alpha=alpha)


@requires_columns(dp_arg="dp", columns=["output", "target", "slice"])
def compute_slice_metrics(
    dp: mk.DataPanel, num_iter: int = 1000, threshold: float = 0.5, flat: bool = False
):
    probs = dp["output"].probabilities().data[:, -1]
    preds = (probs > threshold).numpy()

    metrics = {
        name: {
            "auroc": auroc_bootstrap_ci(
                dp["target"][mask], probs[mask], num_iter=num_iter
            ),
            "recall": recall_bootstrap_ci(
                dp["target"][mask], preds[mask], num_iter=num_iter
            ),
            "precision": precision_bootstrap_ci(
                dp["target"][mask], preds[mask], num_iter=num_iter
            ),
        }
        for name, mask in [
            ("overall", np.ones_like(probs, dtype=bool)),
            ("in_slice", (dp["slice"] == 1) | (dp["target"] == 0)),
            ("out_slice", dp["slice"] != 1),
        ]
    }

    return flatten_dict(metrics) if flat else metrics

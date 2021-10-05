from typing import List, Set, Union

import meerkat as mk
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from scipy.stats import rankdata
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch._C import Value

from domino.utils import flatten_dict, requires_columns

nltk.download("punkt")


def precision_at_k(slice: np.ndarray, pred_slice: np.ndarray, k: int = 25):
    return precision_score(
        slice, rankdata(-pred_slice, method="ordinal") <= k, zero_division=0
    )


def recall_at_k(slice: np.ndarray, pred_slice: np.ndarray, k: int = 25):
    return recall_score(slice, rankdata(-pred_slice) <= k, zero_division=0)


PRECISION_K = [10, 25, 100]
RECALL_K = [50, 100, 200]


@requires_columns(dp_arg="dp", columns=["pred_slices", "slices"])
def compute_sdm_metrics(dp: mk.DataPanel) -> pd.DataFrame:
    pred_slice = dp["pred_slices"].argmax(axis=-1)
    no_nan_preds = not np.isnan(dp["pred_slices"]).any()

    return pd.DataFrame(
        [
            {
                "pred_slice_idx": pred_slice_idx,
                "slice_idx": slice_idx,
                "auroc": roc_auc_score(
                    dp["slices"][:, slice_idx], dp["pred_slices"][:, pred_slice_idx]
                )
                if len(np.unique(dp["slices"][:, slice_idx])) > 1 and no_nan_preds
                else np.nan,
                **{
                    f"precision_at_{k}": precision_at_k(
                        dp["slices"][:, slice_idx],
                        dp["pred_slices"][:, pred_slice_idx],
                        k=k,
                    )
                    if len(np.unique(dp["slices"][:, slice_idx])) > 1 and no_nan_preds
                    else np.nan
                    for k in PRECISION_K
                },
                **{
                    f"recall_at_{k}": recall_at_k(
                        dp["slices"][:, slice_idx],
                        dp["pred_slices"][:, pred_slice_idx],
                        k=k,
                    )
                    if len(np.unique(dp["slices"][:, slice_idx])) > 1 and no_nan_preds
                    else np.nan
                    for k in RECALL_K
                },
                "recall": recall_score(
                    dp["slices"][:, slice_idx],
                    (pred_slice == pred_slice_idx).astype(int),
                )
                if no_nan_preds
                else np.nan,
                "precision": precision_score(
                    dp["slices"][:, slice_idx],
                    (pred_slice == pred_slice_idx).astype(int),
                )
                if no_nan_preds
                else np.nan,
            }
            for slice_idx in range(dp["slices"].shape[1])
            for pred_slice_idx in range(dp["pred_slices"].shape[1])
        ]
    )


@requires_columns(dp_arg="word_dp", columns=["pred_slices", "word"])
def compute_expl_metrics(
    word_dp: mk.DataPanel, slice_names: List[str], dataset: int = "imagenet"
) -> pd.DataFrame:
    from nltk.corpus import wordnet as wn

    def _check_phrase(phrase: str, slice_synsets: Set[str]):
        words = word_tokenize(phrase)
        for word in words:
            if not set(wn.synsets(word)).isdisjoint(slice_synsets):
                return True
        return False

    rows = []
    ranks = rankdata(-word_dp["pred_slices"], axis=0, method="ordinal")
    for slice_idx, slice_name in enumerate(slice_names):
        if dataset == "imagenet":
            slice_synsets = [wn.synset(slice_name)]
        elif dataset == "celeba":
            from domino.slices.celeba import ATTRIBUTE_SYNSETS

            attr, attr_value = slice_name.split("_", 1)[-1].split("=")
            slice_synsets = [wn.synset(s) for s in ATTRIBUTE_SYNSETS[attr]]
        else:
            raise ValueError("Dataset not supported.")

        matches = word_dp["word"].apply(
            lambda x: _check_phrase(x, slice_synsets=slice_synsets)
        )
        match_ranks = ranks[matches]
        metrics = {
            "mean_reciprocal_rank": (1 / match_ranks).mean(axis=0),
            "max_reciprocal_rank": (1 / match_ranks).max(axis=0),
            "mean_rank": match_ranks.mean(axis=0),
            "min_rank": match_ranks.min(axis=0),
        }
        rows.extend(
            [
                {
                    "pred_slice_idx": pred_slice_idx,
                    "slice_idx": slice_idx,
                    "slice_name": slice_name,
                    "slice_synsets": [s.name() for s in slice_synsets],
                    **{k: v[pred_slice_idx] for k, v in metrics.items()},
                }
                for pred_slice_idx in range(ranks.shape[1])
            ]
        )
    return pd.DataFrame(rows)


@requires_columns(dp_arg="dp", columns=["target", "slices"])
def compute_model_metrics(
    dp: mk.DataPanel,
    num_iter: int = 1000,
    threshold: float = 0.5,
    flat: bool = False,
):
    if "output" in dp:
        probs = dp["output"].softmax(1)[:, 1].numpy()
    else:
        if isinstance(dp["probs"], mk.TensorColumn):
            probs = dp["probs"][:, 1].numpy()
        else:
            probs = dp["probs"][:, 1]

    preds = (probs > threshold).astype(float)

    # # KS: Hacky way to get around having one slice for now
    if len(dp["slices"].shape) == 1:
        dp["slices"] = dp["slices"].reshape(-1, 1)

    metrics = {
        name: {
            "auroc": auroc_bootstrap_ci(
                dp["target"][mask], probs[mask], num_iter=num_iter
            )
            if len(np.unique(dp["target"][mask])) == 2
            else np.nan,
            "recall": recall_bootstrap_ci(
                dp["target"][mask], preds[mask], num_iter=num_iter
            ),
            "precision": precision_bootstrap_ci(
                dp["target"][mask], preds[mask], num_iter=num_iter
            ),
            "accuracy": accuracy_bootstrap_ci(
                dp["target"][mask], preds[mask], num_iter=num_iter
            ),
        }
        for name, mask in [
            ("overall", np.ones_like(probs, dtype=bool)),
            *(
                (f"in_slice_{slice_idx}", (dp["slices"][:, slice_idx] == 1))
                for slice_idx in range(dp["slices"].shape[-1])
            ),
            ("out_slice", dp["slices"].sum(axis=1) == 0),
        ]
    }
    # metrics = {
    #     name: {
    #         "auroc": roc_auc_score(dp["target"][mask], probs[mask])
    #         if len(np.unique(dp["target"][mask])) == 2
    #         else np.nan,
    #         "recall": recall_score(dp["target"][mask], preds[mask]),
    #         "precision": precision_score(dp["target"][mask], preds[mask]),
    #         "f1_score": f1_score(dp["target"][mask], preds[mask]),
    #     }
    #     for name, mask in [
    #         ("overall", np.ones_like(probs, dtype=bool)),
    #         *(
    #             (f"in_slice_{slice_idx}", (dp["slices"][:, slice_idx] == 1))
    #             for slice_idx in range(dp["slices"].shape[-1])
    #         ),
    #         ("out_slice", dp["slices"].sum(axis=1) == 0),
    #     ]
    # }

    return flatten_dict(metrics) if flat else metrics


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


def accuracy_bootstrap_ci(
    targets: np.ndarray, preds: np.ndarray, num_iter: int = 10000, alpha: float = 0.05
):
    return compute_bootstrap_ci(targets == preds, num_iter=num_iter, alpha=alpha)

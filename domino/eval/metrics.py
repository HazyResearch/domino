from typing import List, Tuple
from dcbench import SliceDiscoveryProblem, SliceDiscoverySolution
import meerkat as mk
import numpy as np
import sklearn.metrics as skmetrics
from domino.utils import unpack_args
from scipy.stats import rankdata
import pandas as pd
from tqdm import tqdm


def compute_metrics(
    solutions: List[SliceDiscoverySolution], run_id: int = None
) -> Tuple[mk.DataPanel]:
    global_metrics = []
    slice_metrics = []
    for solution in tqdm(solutions):
        g, s = compute_solution_metrics(solution)
        global_metrics.append(g)
        slice_metrics.extend(s)
    return mk.DataPanel(global_metrics), mk.DataPanel(slice_metrics)


def compute_solution_metrics(
    solution: SliceDiscoverySolution,
):
    metrics = _compute_metrics(
        data=solution.merge(),
        slice_target_column="slices",
        slice_pred_column="slice_preds",
        slice_prob_column="slice_probs",
        slice_names=solution.problem.slice_names,
    )
    for row in metrics:
        row["solution_id"] = solution.id
        row["problem_id"] = solution.problem_id
    return metrics


def _compute_metrics(
    data: mk.DataPanel,
    slice_target_column: str,
    slice_pred_column: str,
    slice_prob_column: str,
    slice_names: List[str],
):
    slice_targets, slice_preds, slice_probs = unpack_args(
        data, slice_target_column, slice_pred_column, slice_prob_column
    )

    # consider complements of slices
    slice_preds = np.concatenate([slice_preds, 1 - slice_preds], axis=1)
    slice_probs = np.concatenate([slice_probs, 1 - slice_probs], axis=1)

    def precision_at_k(slc: np.ndarray, pred_slice: np.ndarray, k: int = 25):
        # don't need to check for zero division because we're taking the top_k
        return skmetrics.precision_score(
            slc, rankdata(-pred_slice, method="ordinal") <= k
        )

    # compute mean response conditional on the slice and predicted slice_targets
    def zero_fill_nan_and_infs(x: np.ndarray):
        return np.nan_to_num(x, nan=0, posinf=0, neginf=0, copy=False)

    metrics = []
    for slice_idx in range(slice_targets.shape[1]):
        slc = slice_targets[:, slice_idx]
        slice_name = slice_names[slice_idx]
        for pred_slice_idx in range(slice_preds.shape[1]):
            slice_pred = slice_preds[:, pred_slice_idx]
            slice_prob = slice_probs[:, pred_slice_idx]

            metrics.append(
                {
                    "target_slice_idx": slice_idx,
                    "target_slice_name": slice_name,
                    "pred_slice_idx": pred_slice_idx,
                    "average_precision": skmetrics.average_precision_score(
                        y_true=slc, y_score=slice_prob
                    ),
                    "precision-at-10": precision_at_k(slc, slice_prob, k=10),
                    "precision-at-25": precision_at_k(slc, slice_prob, k=25),
                    **dict(
                        zip(
                            ["precision", "recall", "f1_score", "support"],
                            skmetrics.precision_recall_fscore_support(
                                y_true=slc,
                                y_pred=slice_pred,
                                average="binary",
                                # note: if slc is empty, recall will be 0 and if pred
                                # is empty precision will be 0
                                zero_division=0,
                            ),
                        )
                    ),
                }
            )

    df = pd.DataFrame(metrics)
    primary_metric = "average_precision"
    slice_metrics = df.iloc[
        df.groupby("target_slice_name")[primary_metric].idxmax().astype(int)
    ]
    return slice_metrics.to_dict("records")

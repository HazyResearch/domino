from __future__ import annotations
from contextlib import redirect_stdout
import dataclasses
from gettext import dpgettext
import io
import itertools

from random import choice, sample
from typing import Collection, Dict, Iterable, List, Tuple, Union
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression

import sklearn.metrics as skmetrics
import pandas as pd
from scipy.stats import rankdata
import terra
import numpy as np
from domino.eval.metrics import compute_solution_metrics
import meerkat as mk
from tqdm.auto import tqdm
import os

from domino import embed, generate_candidate_descriptions
from domino.utils import unpack_args
from dcbench import SliceDiscoveryProblem, SliceDiscoverySolution


def _run_sdms(problems: List[SliceDiscoveryProblem], **kwargs):
    result = []
    for problem in problems:
        # f = io.StringIO()
        # with redirect_stdout(f):
        result.append(run_sdm(problem, **kwargs))
    return result


def run_sdms(
    problems: List[SliceDiscoveryProblem],
    slicer_class: type,
    slicer_config: dict,
    emb_dp: mk.DataPanel,
    embedding_col: str = "emb",
    batch_size: int = 1,
    num_workers: int = 0,
):
    if num_workers > 0:
        import ray

        ray.init()
        run_fn = ray.remote(_run_sdms).remote
    else:
        run_fn = _run_sdms

    total_batches = len(problems)
    results = []
    t = tqdm(total=total_batches)

    for start_idx in range(0, len(problems), batch_size):
        batch = problems[start_idx : start_idx + batch_size]

        result = run_fn(
            problems=batch,
            emb_dp=emb_dp,
            embedding_col=embedding_col,
            # candidate_descriptions=candidate_descriptions,
            slicer_class=slicer_class,
            slicer_config=slicer_config,
        )

        if num_workers == 0:
            t.update(n=len(result))
            results.extend(result)
        else:
            # in the parallel case, this is a single object reference
            # moreover, the remote returns immediately so we don't update tqdm
            results.append(result)

    if num_workers > 0:
        # if we're working in parallel, we need to wait for the results to come back
        # and update the tqdm accordingly
        result_refs = results
        results = []
        while result_refs:
            done, result_refs = ray.wait(result_refs)
            for result in done:
                result = ray.get(result)
                results.extend(result)
                t.update(n=len(result))
        ray.shutdown()
    solutions, metrics = zip(*results)
    # flatten the list of lists
    metrics = [row for slices in metrics for row in slices]

    return solutions, pd.DataFrame(metrics)


def run_sdm(
    problem: SliceDiscoveryProblem,
    # candidate_descriptions: Descriptions,
    slicer_class: type,
    slicer_config: dict,
    emb_dp: mk.DataPanel,
    embedding_col: str = "emb",
) -> SliceDiscoverySolution:
    val_dp = problem.merge(split="val")
    val_dp = val_dp.merge(emb_dp["id", embedding_col], on="id", how="left")

    slicer = slicer_class(pbar=False, **slicer_config)
    slicer.fit(val_dp, embeddings=embedding_col, targets="target", pred_probs="probs")

    test_dp = problem.merge(split="test")
    test_dp = test_dp.merge(emb_dp["id", embedding_col], on="id", how="left")
    result = mk.DataPanel({"id": test_dp["id"]})
    result["slice_preds"] = slicer.predict(
        test_dp, embeddings=embedding_col, targets="target", pred_probs="probs"
    )
    result["slice_probs"] = slicer.predict_proba(
        test_dp, embeddings=embedding_col, targets="target", pred_probs="probs"
    )

    # descriptions = slicer.describe(
    #     text_data=candidate_descriptions.dp,
    #     text_embeddings=candidate_descriptions.embedding_column,
    #     text_descriptions=candidate_descriptions.description_column,
    #     num_descriptions=5,
    # )

    solution = SliceDiscoverySolution(
        artifacts={
            "pred_slices": result,
        },
        attributes={
            "problem_id": problem.id,
            "slicer_class": slicer_class,
            "slicer_config": slicer_config,
            "embedding_column": embedding_col,
        },
    )
    metrics = compute_solution_metrics(
        solution,
    )
    return solution, metrics

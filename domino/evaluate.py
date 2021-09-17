from typing import Dict, Mapping, Sequence, Tuple, Union

import meerkat as mk
import pandas as pd
import terra
import torch.nn as nn
from meerkat.datapanel import DataPanel
from ray import tune
from ray.tune.utils.placement_groups import PlacementGroupFactory
from tqdm import tqdm

from domino.metrics import compute_expl_metrics, compute_sdm_metrics
from domino.sdm.abstract import SliceDiscoveryMethod
from domino.slices.abstract import build_setting
from domino.train import score_model


@terra.Task
def run_sdm(
    model: nn.Module,
    data_dp: mk.DataPanel,
    emb_dp: Union[mk.DataPanel, Mapping[str, mk.DataPanel]],
    sdm_class: type,
    sdm_config: SliceDiscoveryMethod.Config,
    id_column: str,
    word_dp: mk.DataPanel = None,
    **kwargs,
):
    print("Creating slice discovery method...")
    sdm: SliceDiscoveryMethod = sdm_class(sdm_config)

    print("Loading embeddings...")
    data_dp = data_dp.lz[data_dp["split"].isin(["valid", "test"])].merge(
        emb_dp[[id_column, sdm.config.emb]], on=id_column
    )

    print("Fitting slice discovery method...")
    sdm.fit(
        data_dp=data_dp.lz[data_dp["split"] == "valid"],
        model=model,
    )
    print("Transforming slice discovery method...")
    slice_dp = sdm.transform(data_dp=data_dp.lz[data_dp["split"] == "test"])

    if word_dp is not None:
        print("Explaining slices...")
        expl_dp = sdm.explain(word_dp, data_dp=slice_dp)
        return slice_dp, expl_dp

    return slice_dp, None


@terra.Task.make(no_load_args={"emb_dp", "word_dp"})
def run_sdms(
    sdm_config: dict,
    setting_dp: mk.DataPanel,
    emb_dp: Union[mk.DataPanel, Dict[str, mk.DataPanel]] = None,
    word_dp: mk.DataPanel = None,
    id_column: str = "image_id",
    run_dir: str = None,
):
    def _evaluate(config):
        score_run_id = config["slice"]["score_model_run_id"]
        if config["slice"]["synthetic_preds"]:
            dp = build_setting.out(score_run_id)
            model = None
        else:
            dp = score_model.out(score_run_id)
            model = score_model.inp(score_run_id)["model"]

        if isinstance(emb_dp, Mapping):
            emb_tuple = config["sdm"]["sdm_config"]["emb"]
            if not isinstance(emb_tuple, Tuple):
                raise ValueError(
                    "Must 'emb' in the sdm config must be a tuple when "
                    "providing multiple `emb_dp`."
                )
            _emb_dp = emb_dp[emb_tuple[0]]
            config["sdm"]["sdm_config"]["emb"] = emb_tuple[1]
        else:
            _emb_dp = emb_dp

        run_id, _ = run_sdm(
            data_dp=dp,
            emb_dp=_emb_dp,
            model=model,
            id_column=id_column,
            word_dp=word_dp,
            **config["slice"],
            **config["sdm"],
            return_run_id=True,
        )

        # need to return metadata to tune so we get it in the analysis dp
        return {
            "run_sdm_run_id": run_id,
            "score_model_run_id": score_run_id,
            **config["sdm"],
        }

    analysis = tune.run(
        _evaluate,
        config={
            "slice": tune.grid_search(list(setting_dp)),
            "sdm": sdm_config,
        },
        resources_per_trial=tune.sample_from(
            lambda spec: spec.config.sdm["sdm_class"].RESOURCES_REQUIRED
        ),
        raise_on_failed_trial=False,  # still want to return dataframe even if some trials fails
        local_dir=run_dir,
    )
    result_dp = mk.merge(
        setting_dp,
        mk.DataPanel.from_pandas(analysis.dataframe()),
        on="score_model_run_id",
    )

    result_dp["sdm_class"] = (
        result_dp["sdm_class"].str.extract(r"'(.*)'", expand=False).data
    )
    return result_dp


@terra.Task
def score_sdms(evaluate_dp: mk.DataPanel, spec_columns: Sequence[str] = None):
    cols = ["target_synset", "run_sdm_run_id"]
    if spec_columns is not None:
        cols += spec_columns
    dfs = []
    for row in tqdm(evaluate_dp):
        dp, _ = run_sdm.out(run_id=row["run_sdm_run_id"])
        metrics_df = compute_sdm_metrics(dp.load())

        for col in cols:
            metrics_df[col] = row[col]

        metrics_df["slice"] = metrics_df["slice_idx"].apply(
            lambda x: row["slice_synsets"][x]
        )
        dfs.append(metrics_df)

    return pd.concat(dfs, axis=0)


@terra.Task
def score_sdm_explanations(
    evaluate_dp: mk.DataPanel, spec_columns: Sequence[str] = None
):
    cols = ["target_synset", "run_sdm_run_id"]
    if spec_columns is not None:
        cols += spec_columns
    dfs = []
    for row in tqdm(evaluate_dp):
        _, words_dp = run_sdm.out(run_id=row["run_sdm_run_id"])
        metrics_df = compute_expl_metrics(
            words_dp.load(), slice_synsets=row["slice_synsets"]
        )

        for col in cols:
            metrics_df[col] = row[col]

        metrics_df["slice"] = metrics_df["slice_idx"].apply(
            lambda x: row["slice_synsets"][x]
        )
        dfs.append(metrics_df)

    return pd.concat(dfs, axis=0)

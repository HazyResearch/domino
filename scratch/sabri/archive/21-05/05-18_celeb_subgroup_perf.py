import numpy as np
import pandas as pd
import terra
from meerkat import DataPanel
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from domino.data.celeb import ATTRIBUTES, build_celeb_df, get_celeb_dp
from domino.utils import auroc_bootstrap_ci


@terra.Task
def get_model_predictions(
    model_df: pd.DataFrame,
    celeb_df: pd.DataFrame,
    get_artifacts_run_id: int = 327,
    bs_num_iter: int = 100,
    run_dir=True,
):
    # Load Celeb DataPanel
    celeb_dp = get_celeb_dp(celeb_df[celeb_df["split"] == "valid"])

    print("Loading model predictions...")
    for idx, (_, row) in tqdm(enumerate(model_df.iterrows())):
        row = row.to_dict()
        act_target = row["target_column"]
        dp = terra.get_artifacts(
            run_id=get_artifacts_run_id, group_name=f"{act_target}_activations"
        )["dp"].load()

        assert celeb_dp["file"] == dp["file"]
        celeb_dp.add_column(f"pred_{act_target}", dp["pred"], overwrite=True)
        celeb_dp.add_column(f"prob_{act_target}", dp["probs"], overwrite=True)
    return celeb_dp


@terra.Task
def compute_celeb_subgroup_perf(
    preds_dp: DataPanel, bs_num_iter: int = 100, run_dir=True
):
    print("compute celeb subgroup performance")
    celeb_dp = preds_dp
    results = []
    for target_attr in tqdm(ATTRIBUTES):
        ci = auroc_bootstrap_ci(
            celeb_dp[target_attr].data,
            celeb_dp[f"prob_{target_attr}"].data[:, -1],
            num_iter=bs_num_iter,
        )
        results.append(
            {
                **ci,
                "target_attr": target_attr,
                "slice_attr": None,
                "slice_value": None,
                "num_examples": len(celeb_dp),
            }
        )

        for slice_attr in ATTRIBUTES:
            if target_attr == slice_attr:
                continue
            for slice_value in [0, 1]:
                curr_dp = celeb_dp[[f"prob_{target_attr}", slice_attr, target_attr]]
                curr_dp = curr_dp.lz[np.where(curr_dp[slice_attr] == slice_value)[0]]
                if len(np.unique(curr_dp[target_attr])) != 2:
                    continue
                ci = auroc_bootstrap_ci(
                    curr_dp[target_attr].data,
                    curr_dp[f"prob_{target_attr}"].data[:, -1],
                    num_iter=bs_num_iter,
                )
                results.append(
                    {
                        **ci,
                        "target_attr": target_attr,
                        "slice_attr": slice_attr,
                        "slice_value": slice_value,
                        "num_examples": len(curr_dp),
                    }
                )
    return pd.DataFrame(results)


if __name__ == "__main__":
    # preds_dp = get_model_predictions(
    #     model_df=terra.out(290),
    #     celeb_df=build_celeb_df.out(141)
    # )
    preds_dp = get_model_predictions.out(run_id=403)
    compute_celeb_subgroup_perf(preds_dp=preds_dp)

import os

import pandas as pd
from terra import Task
from terra.io import json_load


@Task.make_task
def collect_models(root_dir: str = "/home/common/models/celeba/_runs", run_dir=None):
    models = []
    for run_id in os.listdir(root_dir):
        run_dir = os.path.join(root_dir, run_id)
        if int(run_id) < 229:
            continue
        try:
            inp = json_load(os.path.join(run_dir, "inputs.json"))
        except:
            continue

        try:
            meta = json_load(os.path.join(run_dir, "meta.json"))
        except:
            continue

        start_time = pd.to_datetime(meta["start_time"].split("_")[0], yearfirst=True)
        if pd.to_datetime("04-09-12") > start_time:
            continue

        try:
            out = json_load(os.path.join(run_dir, "best_chkpt.json"))
        except:
            continue

        tgt = inp["target_key"] if "target_key" in inp else inp["target_column"]
        if tgt == "y":
            continue
        models.append(
            {
                "run_id": int(run_id),
                "build_df_run_id": int(inp["data_df"].run_id),
                "run_dir": run_dir,
                "model_path": os.path.join(
                    root_dir, run_id, "artifacts", out["model"].key
                ),
                "target_column": tgt,
            }
        )
    return pd.DataFrame(models)


if __name__ == "__main__":
    collect_models()

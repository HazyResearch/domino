
from domino.evaluate import run_sdms, run_sdm, score_sdm_explanations, score_sdms
import pandas as pd
import meerkat as mk

#PALETTE = ["#9CBDE8", "#53B7AE", "#EFAB79", "#E27E51", "#19416E", "#1B6C7B"]
PALETTE = ["#9CBDE8", "#316FAE", "#29B2A1", "#007C6E", "#FFA17A", "#A4588F"]

def coherence_metric(grouped_df):
    return (grouped_df['auroc']>0.85) & (grouped_df['precision_at_10']>0.5)


EMB_PALETTE = {
    #"random": "#19416E",
    "bit": PALETTE[0],
    "imagenet": PALETTE[1],
    "clip": PALETTE[2],
    "mimic_multimodal": PALETTE[3],
    "convirt": PALETTE[4],
    "activations": PALETTE[5],
}

def generate_group_df(run_sdms_id, score_sdms_id, slice_type):
    setting_dp = run_sdms.out(run_sdms_id).load()
    slice_df = score_sdms.out(score_sdms_id).load() 
    slice_df = pd.DataFrame(slice_df)
    score_dp = mk.DataPanel.from_pandas(slice_df)
    results_dp = mk.merge(
        score_dp,
        setting_dp["config/sdm", "alpha","run_sdm_run_id", "sdm_class"],
        on="run_sdm_run_id"
    )
    emb_col = results_dp["config/sdm"].map(lambda x: x["sdm_config"]["emb"][0] if x["sdm_config"]["emb"][0] != None else "activations")
    results_dp["emb_type"] = emb_col

    results_df = results_dp.to_pandas()
    grouped_df = results_df.iloc[results_df.reset_index().groupby(["slice_name", "slice_idx", "sdm_class", "alpha", "emb_type"])['precision_at_10'].idxmax().astype(int)]
    grouped_df["alpha"] = grouped_df["alpha"].round(3)

    if(slice_type=="correlation"): grouped_df = grouped_df[grouped_df['alpha'] != 0.0]
    grouped_df['success'] = coherence_metric(grouped_df)
    grouped_df['slice_type'] = [slice_type]*len(grouped_df)
    return(grouped_df)
import os

os.environ["TERRA_CONFIG_PATH"] = "/home/sabri/code/domino/terra-config.json"
from domino.data.cxr import build_cxr_df, get_cxr_activations, get_dp

if False:
    df = build_cxr_df.out(load=True)
    dp = get_dp(df)

    act_dp = get_cxr_activations(
        dp=dp,
        model_path="/home/common/datasets/cxr-tube/models/cxr_pmx_mimic_pretrained.pth",
    )
else:
    from domino.data.celeb import build_celeb_df, get_celeb_dp

    df = build_celeb_df.out(load=True)
    dp = get_celeb_dp(df)

    act_dp = get_cxr_activations(
        dp=dp,
        model_path="/home/common/datasets/cxr-tube/models/cxr_pmx_mimic_pretrained.pth",
    )

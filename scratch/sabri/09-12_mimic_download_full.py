from meerkat.contrib.mimic import build_mimic_dp, download_mimic_dp

dp = build_mimic_dp(
    dataset_dir="/home/common/datasets/mimic",
    gcp_project="hai-gcp-fine-grained",
    write=True,
    download_jpg=True,
    download_resize=1024,
)

from domino.data.gqa import read_gqa_dps, split_gqa
from domino.emb.imagenet import embed_images

dps = read_gqa_dps("/home/common/datasets/gqa")
dp = dps["objects"]

split_dp = split_gqa.out(4681, load=True)
dp = dp.merge(split_dp[["image_id", "split"]], on="image_id")

dp = embed_images(
    dp=dp.lz[dp["split"].isin(["test", "valid"])],
    layers={"emb": "layer4"},
    img_column="object_image",
    num_workers=7,
    mmap=True,
)

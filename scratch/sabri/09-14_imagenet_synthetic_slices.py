import numpy as np

from domino import evaluate
from domino.data.imagenet import get_imagenet_dp
from domino.emb.clip import embed_images
from domino.slices.imagenet import collect_rare_slices
from domino.train import synthetic_score_slices
from domino.utils import split_dp

data_dp = get_imagenet_dp.out(6617)
split = split_dp.out(6478)


if False:

    if False:
        slices_dp = collect_rare_slices(data_dp=data_dp)
    else:
        slices_dp = collect_rare_slices.out(6654)

    slices_dp = synthetic_score_slices(
        slices_dp=slices_dp,
        data_dp=data_dp,
        split_dp=split,
        synthetic_kwargs={"slice_sensitivities": 0.5},
    )
else:
    slices_dp = synthetic_score_slices.out(6655)

dp = embed_images(
    dp=data_dp,
    split_dp=split,
    splits=["valid", "test"],
    img_column="image",
    num_workers=7,
    mmap=True,
)

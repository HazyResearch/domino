from domino.data.imagenet import get_imagenet_dp
from domino.slices.imagenet import collect_rare_slices
from domino.train import synthetic_score_slices
from domino.utils import split_dp

data_dp = get_imagenet_dp.out(6129)
split = split_dp.out(6135)
slices_dp = collect_rare_slices(data_dp=data_dp)

synthetic_score_slices(
    slices_dp=slices_dp,
    data_dp=data_dp,
    split_dp=split,
    synthetic_kwargs={"slice_sensitivities": 0.5},
)

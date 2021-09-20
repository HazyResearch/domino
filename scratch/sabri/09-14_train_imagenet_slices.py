from domino.data.imagenet import get_imagenet_dp
from domino.slices.imagenet import collect_rare_slices
from domino.train import train_slices
from domino.utils import split_dp

data_dp = get_imagenet_dp()
split = split_dp(data_dp, split_on="image_id")
slices_dp = collect_rare_slices(data_dp)

train_slices(slices_dp=slices_dp, data_dp=data_dp, split_dp=split)

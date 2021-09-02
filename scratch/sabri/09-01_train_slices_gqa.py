import ray

from domino.slices.gqa import collect_correlation_slices
from domino.slices.train import train_slices

ray.init("ray://ray-t4-1-cluster-ray-head:10001")

slices_dp = collect_correlation_slices.out(4690, load=True)
train_slices(slices_dp=slices_dp, split_run_id=4681)

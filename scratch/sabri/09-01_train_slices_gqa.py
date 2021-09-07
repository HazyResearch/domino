import ray

from domino.slices.gqa import collect_correlation_slices, collect_rare_slices
from domino.slices.train import train_slices

slices_dp = collect_rare_slices.out(4917, load=True).lz[:4]
train_slices(slices_dp=slices_dp, split_run_id=4681)

from domino.slices.train import train_slices
from domino.slices.visual_genome import collect_correlation_slices

slices_dp = collect_correlation_slices.out(4534, load=True)
slices_dp = slices_dp[slices_dp["group"] != "size"][26:]
train_slices(slices_dp=slices_dp, split_run_id=4532)

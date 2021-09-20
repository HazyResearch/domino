import ray

from domino.slices.gqa import collect_correlation_slices, collect_rare_slices
from domino.slices.train import score_slices, train_slices
from domino.vision import score

slices_dp = collect_rare_slices.out(5092).load()
train_slices(
    slices_dp=slices_dp.lz[slices_dp["target_name"] != "person"], split_run_id=4681
)

# score_slices(
#     model_df=train_slices.out(4930),
#     layers={"layer4": "model.layer4"},
#     batch_size=128,
#     reduction_fns=["mean"]
# )

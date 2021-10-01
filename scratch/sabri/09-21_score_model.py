from domino.train import score_settings, train_settings

model_dp = train_settings.out(13480)
score_settings(
    model_dp=model_dp,
    layers={"layer4": "model.layer4"},
    batch_size=512,
    reduction_fns=["mean"],
    num_gpus=4,
    num_cpus=32,
)

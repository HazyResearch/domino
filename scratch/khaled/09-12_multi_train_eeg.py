import ray

from domino.slices.eeg import collect_correlation_slices
from domino.slices.train import score_slices, train_slices
from domino.vision import score

if __name__ == "__main__":

    config = {
        "model_name": "dense_inception",
        "train_transform": None,
        "transform": None,
    }

    kwargs = {
        "batch_size": 10,
        "ckpt_monitor": "valid_auroc",
        "config": config,
        "max_epochs": 10,
    }

    slices_dp = collect_correlation_slices.out(414)
    train_slices(slices_dp=slices_dp, split_run_id=412, **kwargs)

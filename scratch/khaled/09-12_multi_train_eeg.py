from domino.slices.eeg import build_stanford_eeg_dp, collect_correlation_slices
from domino.train import score_slices, train_slices
from domino.utils import split_dp

if __name__ == "__main__":

    dp_art = build_stanford_eeg_dp.out(run_id=409)

    config = {
        "model_name": "dense_inception",
        "train_transform": None,
        "transform": None,
    }

    kwargs = {
        "batch_size": 10,
        "ckpt_monitor": "valid_auroc",
        "train_config": config,
        "max_epochs": 10,
        "drop_last": True,
    }

    slices_dp = collect_correlation_slices.out(516)
    split_dp_ = split_dp.out(515)

    train_slices(slices_dp, data_dp=dp_art, split_dp=split_dp_, **kwargs)

    # score_slices(
    #     model_df=train_slices.out(494),
    #     slice_col="binarized_age",
    #     batch_size=10,
    #     reduction_fns=["mean"],
    # )

#        layers={"fc1": "model.fc1"},

import meerkat as mk
import terra
from meerkat.contrib.eeg import build_stanford_eeg_dp

from domino.slices.eeg import EegSliceBuilder
from domino.train import score_model

# from domino.train import score_slices, train_model, train_slices
from domino.utils import balance_dp, merge_in_split, split_dp
from domino.vision import train


@terra.Task
def train_eeg(
    dp: mk.DataPanel, target_column: str, input_column: str, run_dir: str = None
):
    train(
        config={
            "model_name": "dense_inception",
            "data_shape": (2400, 19),
            "train_transform": None,
            "transform": None,
            "lr": 1e-06,
        },
        dp=dp,
        input_column=input_column,
        id_column="id",
        target_column=target_column,
        ckpt_monitor="valid_auroc",
        batch_size=32,
        run_dir=run_dir,
        val_check_interval=10,
        num_workers=6,
        valid_split="valid",
        use_terra=True,
        max_epochs=75,
        drop_last=True,
    )


if __name__ == "__main__":

    dp = build_stanford_eeg_dp.out(
        812, load=True
    )  # for multimodal with 12 sec: 812 # for multimodal dp with 60 sec: 696 # was balance_dp 623 # build_stanford_eeg_dp.out(run_id=409, load=True)
    split_dp_ = split_dp.out(697, load=True)  # was 625

    # dp = EegSliceBuilder().build_correlation_setting(
    #     dp, correlate="age", corr=0.9, n=8000, correlate_threshold=1
    # )

    # dp["binarized_age"] = dp["age"] > 1

    dp = merge_in_split(dp, split_dp_)

    train_eeg(
        dp=dp,  # train_model.inp(578)["dp"],
        target_column="target",
        input_column="input",
    )

    # score_dp = score_model(
    #     dp=dp,
    #     model=train_eeg.get(664, "best_chkpt")["model"],
    #     split="valid",
    #     layers={"fc1": "model.fc1"},
    # )

# run_dir="/home/ksaab/Documents/domino/scratch/khaled/results",

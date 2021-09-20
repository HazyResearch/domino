import meerkat as mk
import terra
from meerkat.contrib.eeg import build_stanford_eeg_dp

from domino.slices.eeg import EegSliceBuilder
from domino.train import score_slices, train_model, train_slices
from domino.utils import balance_dp, merge_in_split, split_dp
from domino.vision import train


@terra.Task
def train_eeg(
    dp: mk.DataPanel, target_column: str, input_column: str, run_dir: str = None
):
    train(
        config={
            "model_name": "dense_inception",
            "train_transform": None,
            "transform": None,
        },
        dp=dp,
        input_column=input_column,
        id_column="id",
        target_column=target_column,
        ckpt_monitor="valid_auroc",
        batch_size=10,
        run_dir=run_dir,
        val_check_interval=10,
        num_workers=6,
        valid_split="valid",
        use_terra=True,
        max_epochs=10,
        drop_last=True,
    )


if __name__ == "__main__":

    # dp = balance_dp.out(
    #     623, load=True
    # )  # build_stanford_eeg_dp.out(run_id=409, load=True)
    # split_dp_ = split_dp.out(625, load=True)

    # dp["binarized_age"] = dp["age"] > 1

    # dp = merge_in_split(dp, split_dp_)

    train_eeg(
        dp=train_model.inp(578)["dp"],
        target_column="target",
        input_column="input",
    )

# run_dir="/home/ksaab/Documents/domino/scratch/khaled/results",

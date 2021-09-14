import meerkat as mk
import terra

from domino.slices.eeg import build_slice
from domino.utils import split_dp
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
    )


if __name__ == "__main__":

    dp_splits_art = split_dp.out(412)
    dp_age_art = build_slice(
        "correlation",
        409,
        dp_splits_art,
        correlate="age",
        corr=0.9,
        n=2500,
        correlate_threshold=10,
    )

    train_eeg(
        dp=dp_age_art,
        target_column="target",
        input_column="input",
    )

# run_dir="/home/ksaab/Documents/domino/scratch/khaled/results",

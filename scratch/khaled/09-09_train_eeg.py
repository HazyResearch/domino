import meerkat as mk
import terra
from meerkat.contrib.eeg import build_stanford_eeg_dp

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
        id_column="file_id",
        target_column=target_column,
        ckpt_monitor="valid_auroc",
        batch_size=10,
        run_dir=run_dir,
        val_check_interval=10,
        num_workers=6,
        valid_split="dev",
        use_terra=True,
    )


if __name__ == "__main__":

    train_eeg(
        dp=build_stanford_eeg_dp.out(run_id=211),
        target_column="binary_sz",
        input_column="eeg_input",
    )

# run_dir="/home/ksaab/Documents/domino/scratch/khaled/results",

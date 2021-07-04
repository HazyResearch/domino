import meerkat as mk
import terra

from domino.data.mimic import build_dp
from domino.vision import train


@terra.Task.make_task
def train_mimic(dp: mk.DataPanel, target: str, run_dir: str = None):

    train(
        dp=dp,
        input_column="input",
        id_column="dicom_id",
        target_column=target,
        batch_size=64,
        num_workers=6,
        run_dir=run_dir,
        valid_split="validate",
        val_check_interval=20,
        weighted_sampling=True,
        config={"lr": 1e-4, "arch": "resnet50"},
    )


if __name__ == "__main__":
    train_mimic(dp=build_dp.out(run_id=963), target="Pneumothorax")

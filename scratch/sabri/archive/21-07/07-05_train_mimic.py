import meerkat as mk
import terra

from domino.data.mimic import build_dp
from domino.vision import train


@terra.Task
def train_mimic(
    dp: mk.DataPanel, target_column: str, input_column: str, run_dir: str = None
):
    # dp = dp.lz[(dp["Pneumothorax"] == 1) | (dp["Pneumothorax"] == 0)]
    # dp["patient_orientation_lf"] = dp["PatientOrientation"].data == "['L', 'F']"
    # dp["young"] = (dp["anchor_age"] < 40)
    # dp["ethnicity_black"] = (dp["ethnicity"].data == "BLACK/AFRICAN AMERICAN")
    dp["gender_m"] = dp["gender"].data == "M"
    train(
        dp=dp,
        input_column=input_column,
        id_column="dicom_id",
        target_column=target_column,
        gpus=1,
        batch_size=24,
        num_workers=12,
        run_dir=run_dir,
        valid_split="validate",
        val_check_interval=20,
        weighted_sampling=True,
        samples_per_epoch=50000,
        config={"lr": 1e-4, "model_name": "resnet", "arch": "resnet50"},
    )


if __name__ == "__main__":
    train_mimic(
        dp=build_dp.out(run_id=1250),
        target_column="gender_m",
        input_column="input_224",
    )

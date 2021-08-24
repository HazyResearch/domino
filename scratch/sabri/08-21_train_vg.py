import meerkat as mk
import terra

from domino.data.mimic import build_dp
from domino.vision import train


@terra.Task.make_task
def train_mimic(
    dp: mk.DataPanel, target_column: str, input_column: str, run_dir: str = None
):
    train(
        config={"pretrained": True},
        dp=dp,
        input_column=input_column,
        id_column="object_id",
        target_column=target_column,
        ckpt_monitor="valid_auroc",
        batch_size=128,
        run_dir=run_dir,
        val_check_interval=10,
        num_workers=6,
    )


if __name__ == "__main__":
    train_mimic(
        dp=terra.inp(4513)["dp"],
        target_column="target",
        input_column="input",
    )

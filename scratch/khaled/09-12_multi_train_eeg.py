import meerkat as mk
import terra
from meerkat.contrib.eeg import build_stanford_eeg_dp
from sklearn.metrics import roc_auc_score

from domino.slices.eeg import build_slice
from domino.vision import score, train


@terra.Task
def train_eeg(
    dp: mk.DataPanel, target_column: str, input_column: str, run_dir: str = None
):
    model = train(
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
        max_epochs=10,
    )

    return model


if __name__ == "__main__":

    # Load original eeg datapanel
    dp_art = build_stanford_eeg_dp.out(run_id=211)
    dp = dp_art.load()
    dp_train = dp.lz[dp["split"] == "train"]
    dp_val = dp.lz[dp["split"] == "dev"]

    val_targets = dp_val["binary_sz"]
    val_ages = dp_val["age"] > 10
    dp_val["binarized_age"] = val_ages.astype(int)

    # loop over different correlation strengths
    corr_list = [0.3, 0.5, 0.9]
    for corr in corr_list:

        # create train dataset with synthetic correlation
        dp_age = build_slice(
            "correlation",
            dp,
            correlate="age",
            corr=corr,
            n=1200,
            correlate_threshold=10,
        ).load()

        # combine dp_val and dp_train_age
        # dp_age.append(dp_val)

        # train model on spurious dataset
        model = train_eeg(
            dp=dp_age,
            target_column="binary_sz",
            input_column="eeg_input",
        ).load()

        # score model on val set
        val_dp_out = score(model, dp_val, input_column="eeg_input")
        val_probs = val_dp_out["output"].softmax(1)[:, 1]

        print("=" * 20)
        print(f"Correlation {corr}:")
        print(f"Overall AUROC: {roc_auc_score(val_targets,val_probs):.3f}")
        print(
            f"Age > 10, AUROC: {roc_auc_score(val_targets[val_ages],val_probs[val_ages]):.3f}"
        )
        print(
            f"Age < 10, AUROC: {roc_auc_score(val_targets[~val_ages],val_probs[~val_ages]):.3f}"
        )


# run_dir="/home/ksaab/Documents/domino/scratch/khaled/results",
#         dp_age[dp_age["split"] == "dev"]["binary_sz"][0] = 1

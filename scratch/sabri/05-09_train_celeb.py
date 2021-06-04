from terra import Task

from domino.vision import train
from domino.data.celeb import build_celeb_df, celeb_task_config


@Task.make_task
def train_celeb_models(run_dir: str = None):

    ATTRIBUTES = [
        "5_o_clock_shadow",
        "arched_eyebrows",
        "attractive",
        "bags_under_eyes",
        "bald",
        "bangs",
        "big_lips",
        "big_nose",
        "black_hair",
        "blond_hair",
        "blurry",
        "brown_hair",
        "bushy_eyebrows",
        "chubby",
        "double_chin",
        "eyeglasses",
        "goatee",
        "gray_hair",
        "heavy_makeup",
        "high_cheekbones",
        "male",
        "mouth_slightly_open",
        "mustache",
        "narrow_eyes",
        "no_beard",
        "oval_face",
        "pale_skin",
        "pointy_nose",
        "receding_hairline",
        "rosy_cheeks",
        "sideburns",
        "smiling",
        "straight_hair",
        "wavy_hair",
        "wearing_earrings",
        "wearing_hat",
        "wearing_lipstick",
        "wearing_necklace",
        "wearing_necktie",
        "young",
    ]

    df = build_celeb_df.out(141)
    runs = []
    for attribute in ATTRIBUTES:
        run_id, _ = train(
            data_df=df,
            target_column=attribute,
            max_epochs=3,
            batch_size=128,
            ckpt_monitor="valid_auroc",
            val_check_interval=0.05,
            num_workers=4,
            **celeb_task_config,
            return_run_id=True
        )
        runs.append({"run_id": run_id, "target_column": attribute})
    return runs


train_celeb_models()

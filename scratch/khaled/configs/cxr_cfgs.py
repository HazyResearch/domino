from configs.generate import flag, prod


def gdro_tube_sweep():

    sweep = prod(
        [
            flag("train.wd", [0, 1e-5, 1e-3, 1e-1]),
            flag("train.loss.gdro", [True]),
            flag("dataset.subgroup_columns", [["chest_tube"]]),
            flag("train.gaze_split", [True]),
        ]
    )

    return sweep


def scribble_sweep():

    sweep = prod(
        [
            flag("train.wd", [1e-5, 1e-3, 1e-1]),
            flag("train.loss.gdro", [True, False]),
            flag(
                "dataset.datapanel_pth",
                ["/media/4tb_hdd/siim/tubescribble_dp_07-24-21.dp"],
            ),
            flag("dataset.subgroup_columns", [["tube"]]),
        ]
    )

    return sweep


def gazeslicer_time_sweep():

    sweep = prod(
        [
            flag("train.wd", [1e-5, 1e-3, 1e-1]),
            flag("train.loss.gdro", [True, False]),
            flag(
                "dataset.datapanel_pth",
                ["/media/4tb_hdd/siim/gazeslicer_dp_07-23-21.dp"],
            ),
            flag("dataset.subgroup_columns", [["gazeslicer_time"]]),
        ]
    )

    return sweep


def multiclass_sweep():

    sweep = prod(
        [
            flag("train.epochs", [100]),
            flag("train.lr", [1e-4]),  # [1e-5, 1e-4, 1e-3]
            flag("train.wd", [1e-3]),  # [1e-6, 1e-5, 1e-3, 1e-1]
            flag("train.multiclass", [True]),
            flag("train.loss.reweight_class", [True]),
            flag("train.loss.reweight_class_alpha", [1, 2, 5, 10, 20]),
            flag("train.gaze_split", [True]),
            flag(
                "dataset.datapanel_pth",
                ["/media/4tb_hdd/siim/tubescribble_dp_07-24-21.dp"],
            ),
            flag("dataset.subgroup_columns", [["chest_tube"]]),
        ]
    )

    return sweep


def tubeclassification_sweep():

    sweep = prod(
        [
            flag("train.epochs", [150]),
            flag("train.lr", [5e-4, 1e-3]),
            flag("train.wd", [1e-6, 1e-5]),
            flag("train.gaze_split", [True]),
            flag(
                "dataset.datapanel_pth",
                ["/media/4tb_hdd/siim/tubescribble_dp_07-24-21.dp"],
            ),
            flag("dataset.target_column", ["chest_tube"]),
        ]
    )

    return sweep

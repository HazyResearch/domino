from configs.generate import flag, prod


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

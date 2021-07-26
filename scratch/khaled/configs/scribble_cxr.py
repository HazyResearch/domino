from configs.generate import flag, prod


def gdro_sweep():

    sweep = prod(
        [
            flag("train.wd", [1e-5, 1e-3, 1e-1]),
            flag("train.loss.gdro", [True, False]),
        ]
    )

    return sweep

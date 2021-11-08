from configs.generate import flag, prod


def erm_runs():

    sweep = prod(
        [
            flag("train.wd", [0]),
            flag("train.loss.gdro", [False]),
            flag("dataset.subgroup_columns", [["chest_tube"]]),
            flag("train.gaze_split", [True]),
            flag("train.seed", [101, 102, 103, 104, 105]),
        ]
    )

    return sweep


def gaze_cnc_runs():

    sweep = prod(
        [
            flag("train.method", ["cnc_gaze"]),
            flag("train.batch_size", [32]),
            flag("train.contrastive_config.contrastive_weight", [0.1]),
            flag("train.seed", [101, 102, 103, 104, 105]),
        ]
    )

    return sweep


def randcon_runs():

    sweep = prod(
        [
            flag("train.wd", [0]),
            flag("train.loss.gdro", [False]),
            flag("dataset.subgroup_columns", [["chest_tube"]]),
            flag("train.gaze_split", [True]),
            flag("train.method", ["randcon"]),
            flag("train.batch_size", [12]),
            flag("train.contrastive_config.contrastive_weight", [0.1]),
            flag("train.seed", [101, 102, 103, 104, 105]),
        ]
    )

    return sweep


def gaze_erm_sweep_lstm():

    sweep = prod(
        [
            flag("train.epochs", [100]),
            flag("train.lr", [1e-3, 1e-5]),
            flag("train.wd", [0, 1e-5]),
            flag("train.gaze_encoder_config.num_layers", [2, 4, 8]),
            flag("train.gaze_encoder_config.bidirectional", [False]),
            flag("train.gaze_encoder_config.hidden_size", [64, 128, 256]),
            flag("train.gaze_encoder_config.encoder", ["lstm"]),
            flag("train.method", ["gaze_erm"]),
        ]
    )

    return sweep


def gaze_erm_sweep_1dconv():

    sweep = prod(
        [
            flag("train.epochs", [100]),
            flag("train.lr", [1e-3, 1e-5]),
            flag("train.wd", [0, 1e-5]),
            flag("train.gaze_encoder_config.num_layers", [2, 3]),
            # flag("train.gaze_encoder_config.nheads", [4, 8, 16]),
            flag("train.gaze_encoder_config.hidden_size", [64, 128, 256]),
            flag("train.gaze_encoder_config.T", [25, 50, 75]),
            flag("train.gaze_encoder_config.encoder", ["1dconv"]),
            flag("train.method", ["gaze_erm"]),
        ]
    )

    return sweep


def gaze_erm_sweep_transformer():

    sweep = prod(
        [
            flag("train.epochs", [100]),
            flag("train.lr", [1e-3]),
            flag("train.wd", [0, 1e-5]),
            flag("train.gaze_encoder_config.num_layers", [2, 4, 8]),
            flag("train.gaze_encoder_config.nheads", [8, 16]),
            flag("train.gaze_encoder_config.hidden_size", [64, 128, 256]),
            flag("train.gaze_encoder_config.T", [25, 50, 75]),
            flag("train.gaze_encoder_config.encoder", ["transformer"]),
            flag("train.method", ["gaze_erm"]),
        ]
    )

    return sweep


def gaze_clip_sweep():

    sweep = prod(
        [
            flag("train.epochs", [100]),
            flag("train.lr", [1e-4]),
            flag("train.wd", [0]),
            flag(
                "train.contrastive_config.contrastive_weight",
                [0, 0.1, 0.3, 0.5, 0.75, 0.9, 1.0],
            ),
            flag("train.gaze_encoder_config.num_layers", [2]),
            flag("train.gaze_encoder_config.nheads", [16]),
            flag("train.gaze_encoder_config.hidden_size", [128]),
            flag("train.gaze_encoder_config.T", [50]),
            flag("train.gaze_encoder_config.encoder", ["transformer"]),
            flag("train.method", ["gaze_clip"]),
        ]
    )

    return sweep


def erm_sweep():

    sweep = prod(
        [
            flag("train.wd", [0, 1e-5, 1e-3, 1e-1]),
            flag("train.loss.gdro", [False]),
            flag("dataset.subgroup_columns", [["chest_tube"]]),
            flag("train.gaze_split", [True]),
        ]
    )

    return sweep


def gdro_sweep():

    sweep = prod(
        [
            flag("train.wd", [0, 1e-5, 1e-3, 1e-1]),
            flag("train.loss.gdro", [True]),
            flag("dataset.subgroup_columns", [["chest_tube"]]),
            flag("train.gaze_split", [True]),
        ]
    )

    return sweep


def cnc_sweep():

    sweep = prod(
        [
            flag("train.wd", [1e-1]),
            flag("train.cnc", [True]),
            flag("train.cnc_config.contrastive_weight", [0.75, 0.9]),
            flag(
                "train.cnc_config.contrastive_dp_pth",
                ["/media/4tb_hdd/siim/contrastive_dp_na_-1_np_32_nn_32.dp"],
            ),
            flag("train.gaze_split", [True]),
            flag("dataset.subgroup_columns", [["chest_tube"]]),
            flag("train.epochs", [1]),
        ]
    )

    return sweep

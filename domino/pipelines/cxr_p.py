import hydra
from omegaconf import DictConfig, OmegaConf

from domino.data.cxr import build_cxr_dp
from domino.evaluate import run_sdms, score_sdms
from domino.pipelines.utils import parse_pipeline_args
from domino.slices import collect_settings
from domino.slices.abstract import concat_settings, random_filter_settings
from domino.train import (
    filter_settings,
    score_settings,
    synthetic_score_settings,
    train_settings,
)


def run_pipeline(cfg: DictConfig):
    data_dp = build_cxr_dp(
        root_dir="/media/4tb_hdd/siim", tube_mask=True, skip_terra_cache=False
    )

    skip_terra_cache = cfg.skip_terra_cache
    setting_dp = concat_settings(
        [
            collect_settings(
                dataset="cxr",
                slice_category="correlation",
                data_dp=data_dp,
                correlate_list=["chest_tube"],
                correlate_thresholds=[None],
                num_corr=5,
                n=675,
                skip_terra_cache=skip_terra_cache,
            ),
            collect_settings(
                dataset="cxr",
                slice_category="rare",
                data_dp=data_dp,
                attributes=["pmx_area"],
                attribute_thresholds=[0.004],
                min_slice_frac=0.01,
                max_slice_frac=0.5,
                num_frac=5,
                target_frac=0.2,
                n=675,
                skip_terra_cache=skip_terra_cache,
            ),
        ]
    )
    print(f"Number of total settings: {len(setting_dp.load())}")

    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["train"]
    train_settings_kwargs = dict(
        setting_dp=setting_dp,
        data_dp=data_dp,
        split_dp=None,
        batch_size=train_cfg["batch_size"],
        num_workers=dataset_cfg["num_workers"],
        valid_split=train_cfg["valid_split"],
        val_check_interval=train_cfg["val_check_interval"],
        max_epochs=train_cfg["epochs"],
        trainmodel_config=cfg,
    )
    setting_dp, _ = train_settings(
        **train_settings_kwargs, skip_terra_cache=skip_terra_cache
    )

    score_settings_kwargs = dict(
        layers={"layer4": "model.layer4"},
        batch_size=train_cfg.batch_size,
        reduction_fns=["mean"],
        split=["test"],
    )

    setting_dp = score_settings(
        model_dp=setting_dp, **score_settings_kwargs, skip_terra_cache=skip_terra_cache
    )

    breakpoint()


@hydra.main(config_path="cfg", config_name="cxr_p_config.yaml")
def main(cfg: OmegaConf):
    # We want to add fields to cfg so need to call OmegaConf.set_struct
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))

    run_pipeline(cfg)


if __name__ == "__main__":
    main()

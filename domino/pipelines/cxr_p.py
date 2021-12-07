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
                skip_terra_cache=True,
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
                skip_terra_cache=True,
            ),
        ]
    )
    print(f"Number of total settings: {len(setting_dp.load())}")

    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["train"]
    train_settings_kwargs = dict(
        setting_dp=setting_dp,
        data_dp=data_dp,
        split_dp=split,
        model_config={"pretrained": False},
        input_column=dataset_cfg["input_column"],
        id_column=dataset_cfg["id_column"],
        target_column=dataset_cfg["target_column"],
        batch_size=train_cfg["batch_size"],
        num_workers=dataset_cfg["num_workers"],
        valid_split=train_cfg["valid_split"],
        val_check_interval=train_cfg["val_check_interval"],
        max_epochs=train_cfg["epochs"],
    )
    setting_dp, _ = train_settings(**train_settings_kwargs)

    breakpoint()


@hydra.main(config_path="cfg", config_name="cxr_p_config.yaml")
def main(cfg: OmegaConf):
    # We want to add fields to cfg so need to call OmegaConf.set_struct
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))

    run_pipeline(cfg)


if __name__ == "__main__":
    main()

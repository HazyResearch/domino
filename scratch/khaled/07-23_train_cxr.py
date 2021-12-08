import hydra
import meerkat as mk
from omegaconf import DictConfig, OmegaConf

from domino.data.cxr import build_cxr_dp
from domino.vision_ks import train


def train_cxr(
    cfg: DictConfig,
):

    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["train"]

    dp = build_cxr_dp(root_dir="/media/4tb_hdd/siim", tube_mask=True).load()

    train(
        dp=dp,
        input_column=dataset_cfg["input_column"],
        id_column=dataset_cfg["id_column"],
        target_column=dataset_cfg["target_column"],
        batch_size=train_cfg["batch_size"],
        num_workers=dataset_cfg["num_workers"],
        valid_split=train_cfg["valid_split"],
        val_check_interval=train_cfg["val_check_interval"],
        max_epochs=train_cfg["epochs"],
        wandb_config={},
        config=cfg,
    )


@hydra.main(config_path="cfg", config_name="config.yaml")
def main(cfg: OmegaConf):
    # We want to add fields to cfg so need to call OmegaConf.set_struct
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))

    train_cxr(cfg)


if __name__ == "__main__":
    main()

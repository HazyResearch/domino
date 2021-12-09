import os
from typing import Dict, Iterable, List, Mapping, Sequence, Union

import meerkat as mk
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import wandb
from meerkat.nn import ClassificationOutputColumn
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from terra.pytorch import TerraModule
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from torchmetrics.functional import dice_score
from torchvision import transforms as transforms
from torchvision.models.segmentation import fcn_resnet50

from domino.contrastive_loss import ContrastiveLoss
from domino.gdro_loss import LossComputer
from domino.modeling import DenseNet, ResNet, SequenceModel
from domino.utils import PredLogger, TerraCheckpoint

# from domino.data.iwildcam import get_iwildcam_model
# from domino.data.wilds import get_wilds_model

DOMINO_DIR = "/home/ksaab/Documents/domino"


def free_gpu(tensors, delete):
    for tensor in tensors:
        tensor = tensor.detach().cpu()
        if delete:
            del tensor


def get_save_dir(config):
    gaze_split = config["train"]["gaze_split"]
    target = config["dataset"]["target_column"]
    subgroup_columns = config["dataset"]["subgroup_columns"]
    subgroups = ""
    for name in subgroup_columns:
        subgroups += f"_{name}"
    subgroups = subgroups if len(subgroup_columns) > 0 else "none"
    method = config["train"]["method"]

    lr = config["train"]["lr"]
    wd = config["train"]["wd"]
    dropout = config["model"]["dropout"]
    save_dir = f"{DOMINO_DIR}/scratch/khaled/results/method_{method}/gaze_split_{gaze_split}/target_{target}/subgroup_{subgroups}/lr_{lr}/wd_{wd}/dropout_{dropout}"

    if "erm" not in config["train"]["method"]:
        cw = config["train"]["contrastive_weight"]
        save_dir += f"/cw_{cw}"

        # if method == "cnc_gaze":
        #     alpha = config["train"]["cnc_config"]["gaze_alpha"]
        #     save_dir += f"/alpha_{alpha}"

    seed = config["train"]["seed"]
    save_dir += f"/seed_{seed}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir


def dictconfig_to_dict(d):
    """Convert object of type OmegaConf to dict so Wandb can log properly
    Support nested dictionary.
    """
    return {
        k: dictconfig_to_dict(v) if isinstance(v, DictConfig) else v
        for k, v in d.items()
    }


class Classifier(pl.LightningModule, TerraModule):
    def __init__(self, config: dict = None):
        super().__init__()
        self.config = config

        self.gazecon = config["train"]["method"] == "gazecon"

        self._set_model()
        criterion_dict = {"cross_entropy": nn.CrossEntropyLoss, "mse": nn.MSELoss}
        criterion_fnc = criterion_dict[config["train"]["loss"]["criterion"]]
        if config["train"]["loss"]["criterion"] == "cross_entropy":
            criterion = criterion_fnc(
                reduction="none",
            )
        else:
            criterion = criterion_fnc(
                reduction="none",
            )

        loss_cfg = config["train"]["loss"]
        dataset_cfg = config["dataset"]

        self.train_loss_computer = LossComputer(
            criterion,
            is_robust=loss_cfg["gdro"],
            dataset_config=dataset_cfg["train_dataset_config"],
            gdro_config=loss_cfg["gdro_config"],
        )
        self.val_loss_computer = LossComputer(
            criterion,
            is_robust=loss_cfg["gdro"],
            dataset_config=dataset_cfg["val_dataset_config"],
            gdro_config=loss_cfg["gdro_config"],
        )

        metrics = self.config.get("metrics", ["auroc", "accuracy"])
        self.set_metrics(metrics, num_classes=dataset_cfg["num_classes"])
        self.valid_preds = PredLogger()

    def set_metrics(self, metrics: List[str], num_classes: int = None):
        num_classes = (
            self.config["dataset"]["num_classes"]
            if num_classes is None
            else num_classes
        )
        _metrics = {
            "accuracy": torchmetrics.Accuracy(compute_on_step=False),
            "auroc": torchmetrics.AUROC(compute_on_step=False),
            # TODO (Sabri): Use sklearn metrics here, torchmetrics doesn't handle case
            # there are only a subset of classes in a test set
            "macro_f1": torchmetrics.F1(num_classes=num_classes, average="macro"),
            "macro_recall": torchmetrics.Recall(
                num_classes=num_classes, average="macro"
            ),
        }
        self.metrics = nn.ModuleDict(
            {name: metric for name, metric in _metrics.items() if name in metrics}
        )  # metrics need to be child module of the model, https://pytorch-lightning.readthedocs.io/en/stable/metrics.html#metrics-and-devices

    def _set_model(self):
        model_cfg = self.config["model"]
        num_classes = self.config["dataset"]["num_classes"]
        if self.gazecon:
            gaze_enc_cfg = self.config["train"]["gaze_encoder_config"]
            model = SequenceModel(
                input_size=3,
                hidden_size=gaze_enc_cfg["hidden_size"],
                num_layers=gaze_enc_cfg["num_layers"],
                encoder=gaze_enc_cfg["encoder"],
                bidirectional=gaze_enc_cfg["bidirectional"],
                nheads=gaze_enc_cfg["nheads"],
                T=gaze_enc_cfg["T"],
            )
            self.gaze_encoder = model
            self.gaze_encoder.classifier = nn.Identity()
            self.contrastive_loss = ContrastiveLoss(type="sim_clr")

        if model_cfg["model_name"] == "resnet":
            self.model = ResNet(
                num_classes=num_classes,
                arch=model_cfg["arch"],
                dropout=model_cfg["dropout"],
                pretrained=model_cfg["pretrained"],
            )
        elif model_cfg["model_name"] == "densenet":
            self.model = DenseNet(num_classes=num_classes, arch=model_cfg["arch"])
        else:
            raise ValueError(f"Model name {model_cfg['model_name']} not supported.")

        if self.gazecon:
            self.fc = self.model.fc
            self.model.fc = nn.Identity()
            self.image_proj = nn.Linear(
                self.fc[1].in_features, int(gaze_enc_cfg["hidden_size"] / 2)
            )

    def forward(self, x):
        if self.gazecon:
            return self.fc(self.model(x))
        return self.model(x)

    def training_step(self, batch, batch_idx):

        if self.gazecon:
            inputs, gaze_inputs, targets, group_ids = (
                batch["input"],
                batch["gaze_seq"],
                batch["target"],
                batch["group_id"],
            )
            embs = self.image_proj(self.model(inputs))

            seq_len = batch["gaze_seq_len"]
            gaze_embs = self.gaze_encoder(gaze_inputs, seq_len)

            labels = torch.ones_like(gaze_embs)[:, 0].long()
            x = torch.stack([embs, gaze_embs]).transpose(0, 1)
            labels = torch.stack([labels, labels]).transpose(0, 1)
            contrastive_loss = self.contrastive_loss(x, labels)

            outs = self.forward(inputs)
            loss = self.train_loss_computer.loss(outs, targets, group_ids)
            self.train_loss_computer.log_stats(self.log, is_training=True)

            self.log("criterion_loss", loss, on_step=True, logger=True)
            cw = self.config["train"]["contrastive_weight"]
            loss = (1 - cw) * loss + cw * contrastive_loss
            self.log("train_loss", loss, on_step=True, logger=True)
            self.log(
                "contrastive_loss",
                contrastive_loss,
                on_step=True,
                logger=True,
            )

        else:
            inputs, targets, group_ids = (
                batch["input"],
                batch["target"],
                batch["group_id"],
            )

            outs = self.forward(inputs)

            loss = self.train_loss_computer.loss(outs, targets, group_ids)
            self.train_loss_computer.log_stats(self.log, is_training=True)
            self.log("train_loss", loss, on_step=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):

        inputs, targets, group_ids, sample_id = (
            batch["input"],
            batch["target"],
            batch["group_id"],
            batch["id"],
        )

        outs = self.forward(inputs)

        loss = self.val_loss_computer.loss(outs, targets, group_ids)
        self.log("valid_loss", loss)  # , sync_dist=True)

        for metric in self.metrics.values():
            metric(torch.softmax(outs, dim=-1)[:, 1], targets)

        self.valid_preds.update(torch.softmax(outs, dim=-1), targets, sample_id)

    def validation_epoch_end(self, outputs) -> None:
        self.val_loss_computer.log_stats(self.log)
        for metric_name, metric in self.metrics.items():
            self.log(f"valid_{metric_name}", metric.compute())  # , sync_dist=True)

    def test_epoch_end(self, outputs) -> None:
        return self.validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        train_cfg = self.config["train"]
        optimizer = torch.optim.Adam(
            self.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["wd"]
        )
        return optimizer


class MTClassifier(Classifier):
    def training_step(self, batch, batch_idx):
        inputs, targets, _ = batch["input"], batch["target"], batch["id"]
        outs = self.forward(inputs)
        loss = nn.functional.cross_entropy(outs, targets)
        self.log("train_loss", loss, on_step=True, logger=True)  # , sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, sample_id = batch["input"], batch["target"], batch["id"]
        outs = self.forward(inputs)
        loss = nn.functional.cross_entropy(outs, targets)
        self.log("valid_loss", loss)  # , sync_dist=True)

        for metric in self.metrics.values():
            metric(torch.softmax(outs, dim=-1), targets)

        self.valid_preds.update(torch.softmax(outs, dim=-1), targets, sample_id)


def train(
    dp: mk.DataPanel,
    input_column: str,
    target_column: str,
    id_column: str,
    model: Classifier = None,
    config: dict = None,
    wandb_config: dict = None,
    num_classes: int = 2,
    max_epochs: int = 50,
    samples_per_epoch: int = None,
    gpus: int = 1,  # Union[int, Iterable] = [0],
    num_workers: int = 10,
    batch_size: int = 16,
    train_split: str = "train",
    valid_split: str = "valid",
    run_dir: str = None,
    pbar: bool = True,
    **kwargs,
):
    # Note from https://pytorch-lightning.readthedocs.io/en/0.8.3/multi_gpu.html: Make sure to set the random seed so that each model initializes with the same weights.
    pl.utilities.seed.seed_everything(config["train"]["seed"])

    train_mask = dp["split"].data == train_split
    # train_mask = np.logical_and(train_mask, dp["chest_tube"].data.astype(str) != "nan")
    val_mask = dp["split"].data == valid_split

    subgroup_columns = config["dataset"]["subgroup_columns"]
    if len(subgroup_columns) > 0:
        group_ids = dp[target_column].data
        for i in range(len(subgroup_columns)):
            group_ids = group_ids + (2 ** (i + 1)) * dp[subgroup_columns[i]]
    else:
        group_ids = dp[target_column].data
    group_ids[np.isnan(group_ids)] = -1

    dp = mk.DataPanel.from_batch(
        {
            "input": dp[input_column],
            "target": dp[target_column],
            "id": dp[id_column],
            "group_id": group_ids.astype(int),
            "gaze_seq": dp["padded_gaze_seq"],
            "gaze_seq_len": dp["gaze_seq_len"],
        }
    )
    train_dp = dp.lz[train_mask]
    val_dp = dp["input", "target", "id", "group_id"].lz[val_mask]

    # create train_dataset_config and val_dataset_config
    subgroup_columns_ = []
    binary_strs = ["without", "with"]
    for i in range(2 ** (len(subgroup_columns) + 1)):
        subgroup_name = f"{binary_strs[(i%2)!=0]}_{target_column}"
        for ndx, name in enumerate(subgroup_columns):
            subgroup_name += f"_{binary_strs[(int(i/(2**(ndx+1)))%2)!=0]}_{name}"
        subgroup_columns_.append(subgroup_name)

    train_dataset_config = {
        "n_groups": len(subgroup_columns_),
        "group_counts": [
            int((train_dp["group_id"] == group_i).sum())
            for group_i in range(len(subgroup_columns_))
        ],
        "group_str": subgroup_columns_,
    }
    val_dataset_config = {
        "n_groups": len(subgroup_columns_),
        "group_counts": [
            int((val_dp["group_id"] == group_i).sum())
            for group_i in range(len(subgroup_columns_))
        ],
        "group_str": subgroup_columns_,
    }

    print(f"Train config: {train_dataset_config}")

    config["dataset"]["train_dataset_config"] = train_dataset_config
    config["dataset"]["val_dataset_config"] = val_dataset_config

    if (model is not None) and (config is not None):
        raise ValueError("Cannot pass both `model` and `config`.")

    if model is None:
        config = {} if config is None else config
        config["dataset"]["num_classes"] = num_classes
        if config["model"]["resume_ckpt"]:
            model = Classifier.load_from_checkpoint(
                checkpoint_path=config["model"]["resume_ckpt"], config=config
            )
        else:
            model = Classifier(config)

    save_dir = get_save_dir(config)
    config = dict(config)
    config.update(wandb_config)
    logger = WandbLogger(
        config=dictconfig_to_dict(config),
        config_exclude_keys="wandb",
        save_dir=save_dir,
        **config["wandb"],
    )

    model.train()
    ckpt_metric = "valid_accuracy"
    mode = "max"
    # if len(subgroup_columns) > 0:
    #     ckpt_metric = "robust val acc"
    # if "erm" not in config["train"]["method"]:
    #     ckpt_metric = "contrastive_loss"
    #     mode = "min"
    # checkpoint_callback = ModelCheckpoint(
    #     monitor=ckpt_metric, mode=mode, every_n_train_steps=5
    # )
    checkpoint_callbacks = [
        TerraCheckpoint(
            dirpath=run_dir,
            monitor=ckpt_metric,
            save_top_k=1,
            mode=mode,
        )
    ]
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        accumulate_grad_batches=1,
        log_every_n_steps=1,
        logger=logger,
        callbacks=checkpoint_callbacks,
        default_root_dir=save_dir,
        **kwargs,
    )

    sampler = None

    if samples_per_epoch is not None:
        sampler = RandomSampler(train_dp, num_samples=samples_per_epoch)

    train_dl = DataLoader(
        train_dp,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
    )
    valid_dl = DataLoader(
        val_dp,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    trainer.fit(model, train_dl, valid_dl)
    wandb.finish()
    return model


def score(
    model: nn.Module,
    dp: mk.DataPanel,
    layers: Mapping[str, nn.Module] = None,
    reduction_fns: Sequence[callable] = None,
    input_column: str = "input",
    pbar: bool = True,
    device: int = 0,
    run_dir: str = None,
    **kwargs,
):
    model.to(device).eval()

    class ActivationExtractor:
        """Class for extracting activations a targetted intermediate layer"""

        def __init__(self, reduction_fn: callable = None):
            self.activation = None
            self.reduction_fn = reduction_fn

        def add_hook(self, module, input, output):
            if self.reduction_fn is not None:
                output = self.reduction_fn(output)
            self.activation = output

    layer_to_extractor = {}

    if layers is not None:
        for name, layer in layers.items():
            if reduction_fns is not None:
                for reduction_fn in reduction_fns:
                    extractor = ActivationExtractor(reduction_fn=reduction_fn)
                    layer.register_forward_hook(extractor.add_hook)
                    layer_to_extractor[name] = extractor
                    # layer_to_extractor[f"{name}_{reduction_fn.__name__}"] = extractor
            else:
                extractor = ActivationExtractor()
                layer.register_forward_hook(extractor.add_hook)
                layer_to_extractor[name] = extractor

    @torch.no_grad()
    def _score(batch: mk.DataPanel):
        x = batch[input_column].data.to(device)
        out = model(x)  # Run forward pass

        return {
            "output": ClassificationOutputColumn(logits=out.cpu(), multi_label=False),
            **{
                name: extractor.activation.cpu()
                for name, extractor in layer_to_extractor.items()
            },
        }

    dp = dp.update(
        function=_score,
        is_batched_fn=True,
        pbar=pbar,
        input_columns=[input_column],
        **kwargs,
    )
    return dp

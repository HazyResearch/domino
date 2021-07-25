import os
from typing import Dict, Iterable, List, Mapping, Union

import meerkat as mk
import pandas as pd
import pytorch_lightning as pl
import terra
import torch
import torch.nn as nn
import torchmetrics
import wandb
from pytorch_lightning.loggers import WandbLogger
from terra import Task
from terra.torch import TerraModule
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from torchvision import transforms as transforms

from domino.gdro_loss import LossComputer
from domino.modeling import DenseNet, ResNet
from domino.utils import PredLogger, TerraCheckpoint

# from domino.data.iwildcam import get_iwildcam_model
# from domino.data.wilds import get_wilds_model


class Classifier(pl.LightningModule, TerraModule):

    DEFAULT_CONFIG = {
        "lr": 1e-4,
        "wd": 0,
        "model_name": "resnet",
        "arch": "resnet18",
        "num_classes": 2,
        "gdro": False,
    }

    def __init__(self, config: dict = None):
        super().__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        if config is not None:
            self.config.update(config)

        self._set_model()
        criterion = nn.CrossEntropyLoss(reduction="none")
        loss_config = config["loss_config"]
        self.train_loss_computer = LossComputer(
            criterion,
            is_robust=loss_config["gdro"],
            dataset_config=config["train_dataset_config"],
            alpha=loss_config["alpha"],
            gamma=loss_config["gamma"],
            step_size=loss_config["robust_step_size"],
            normalize_loss=loss_config["use_normalized_loss"],
            btl=loss_config["btl"],
            min_var_weight=loss_config["min_var_weight"],
        )
        self.val_loss_computer = LossComputer(
            criterion,
            is_robust=loss_config["gdro"],
            dataset_config=config["val_dataset_config"],
            alpha=loss_config["alpha"],
            gamma=loss_config["gamma"],
            step_size=loss_config["robust_step_size"],
            normalize_loss=loss_config["use_normalized_loss"],
            btl=loss_config["btl"],
            min_var_weight=loss_config["min_var_weight"],
        )

        metrics = self.config.get("metrics", ["auroc", "accuracy"])
        self.set_metrics(metrics, num_classes=self.config["num_classes"])
        self.valid_preds = PredLogger()

    def set_metrics(self, metrics: List[str], num_classes: int = None):
        num_classes = self.config["num_classes"] if num_classes is None else num_classes
        _metrics = {
            "accuracy": torchmetrics.Accuracy(compute_on_step=False),
            "auroc": torchmetrics.AUROC(compute_on_step=False, num_classes=num_classes),
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
        if self.config["model_name"] == "resnet":
            self.model = ResNet(
                num_classes=self.config["num_classes"], arch=self.config["arch"]
            )
        elif self.config["model_name"] == "densenet":
            self.model = DenseNet(
                num_classes=self.config["num_classes"], arch=self.config["arch"]
            )
        else:
            raise ValueError(f"Model name {self.config['model_name']} not supported.")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets, group_ids = batch["input"], batch["target"], batch["group_id"]
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
        self.val_loss_computer.log_stats(self.log)
        self.log("valid_loss", loss)

        for metric in self.metrics.values():
            metric(torch.softmax(outs, dim=-1), targets)

        self.valid_preds.update(torch.softmax(outs, dim=-1), targets, sample_id)

    def validation_epoch_end(self, outputs) -> None:
        for metric_name, metric in self.metrics.items():
            self.log(f"valid_{metric_name}", metric.compute())

    def test_epoch_end(self, outputs) -> None:
        return self.validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config["lr"], weight_decay=self.config["wd"]
        )
        return optimizer


class MTClassifier(Classifier):
    def training_step(self, batch, batch_idx):
        inputs, targets, _ = batch["input"], batch["target"], batch["id"]
        outs = self.forward(inputs)
        loss = nn.functional.cross_entropy(outs, targets)
        self.log("train_loss", loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, sample_id = batch["input"], batch["target"], batch["id"]
        outs = self.forward(inputs)
        loss = nn.functional.cross_entropy(outs, targets)
        self.log("valid_loss", loss)

        for metric in self.metrics.values():
            metric(torch.softmax(outs, dim=-1), targets)

        self.valid_preds.update(torch.softmax(outs, dim=-1), targets, sample_id)


def train(
    dp: mk.DataPanel,
    input_column: str,
    target_column: str,
    id_column: str,
    subgroup_columns: List[str] = ["none"],
    model: Classifier = None,
    config: dict = None,
    num_classes: int = 2,
    max_epochs: int = 50,
    samples_per_epoch: int = None,
    gpus: Union[int, Iterable] = 1,
    num_workers: int = 10,
    batch_size: int = 16,
    ckpt_monitor: str = "valid_accuracy",
    train_split: str = "train",
    valid_split: str = "valid",
    wandb_config: dict = None,
    weighted_sampling: bool = False,
    seed: int = 123,
    run_dir: str = None,
    **kwargs,
):
    # Note from https://pytorch-lightning.readthedocs.io/en/0.8.3/multi_gpu.html: Make sure to set the random seed so that each model initializes with the same weights.
    pl.utilities.seed.seed_everything(seed)

    # TODO: make this work for multiple subgroup columns
    group_ids = 2 * dp[subgroup_columns[0]] + dp[target_column]
    dp = mk.DataPanel.from_batch(
        {
            "input": dp[input_column],
            "target": dp[target_column].astype(int),
            "id": dp[id_column],
            "split": dp["split"],
            "group_id": group_ids.astype(int),
        }
    )
    train_dp = dp.lz[dp["split"].data == train_split]
    val_dp = dp.lz[dp["split"].data == valid_split]

    # create train_dataset_config and val_dataset_config
    subgroup_columns_ = []
    for name in subgroup_columns:
        for str_ in ["without", "with"]:
            subgroup_columns_.append(f"without_{name}_{str_}_{target_column}")
            subgroup_columns_.append(f"with_{name}_{str_}_{target_column}")

    train_dataset_config = {
        "n_groups": len(subgroup_columns_),
        "group_counts": torch.Tensor(
            [
                (train_dp["group_id"] == group_i).sum()
                for group_i in range(len(subgroup_columns_))
            ]
        ),
        "group_str": subgroup_columns_,
    }
    val_dataset_config = {
        "n_groups": len(subgroup_columns_),
        "group_counts": torch.Tensor(
            [
                (val_dp["group_id"] == group_i).sum()
                for group_i in range(len(subgroup_columns_))
            ]
        ),
        "group_str": subgroup_columns_,
    }

    print(f"Train config: {train_dataset_config}")

    config["train_dataset_config"] = train_dataset_config
    config["val_dataset_config"] = val_dataset_config

    if (model is not None) and (config is not None):
        raise ValueError("Cannot pass both `model` and `config`.")

    if model is None:
        config = {} if config is None else config
        config["num_classes"] = num_classes
        model = Classifier(config)

    run_id = int(os.path.basename(run_dir))
    metadata = terra.get_meta(run_id)
    logger = WandbLogger(
        project="domino",
        save_dir=run_dir,
        name=f"{metadata['fn']}-run_id={os.path.basename(run_dir)}",
        tags=[f"{metadata['module']}.{metadata['fn']}"],
        config=config,
    )

    checkpoint_callbacks = [
        TerraCheckpoint(
            dirpath=run_dir,
            monitor=ckpt_monitor,
            save_top_k=1,
            mode="max",
        )
    ]
    model.train()
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        log_every_n_steps=1,
        logger=logger,
        callbacks=checkpoint_callbacks,
        default_root_dir=run_dir,
        accelerator=None,
        auto_select_gpus=True,
        **kwargs,
    )

    sampler = None
    if weighted_sampling:
        weights = torch.ones(len(train_dp))
        weights[train_dp["target"] == 1] = (1 - dp["target"]).sum() / (
            dp["target"].sum()
        )
        samples_per_epoch = (
            len(train_dp) if samples_per_epoch is None else samples_per_epoch
        )
        sampler = WeightedRandomSampler(weights=weights, num_samples=samples_per_epoch)
    elif samples_per_epoch is not None:
        sampler = RandomSampler(train_dp, num_samples=samples_per_epoch)

    train_dl = DataLoader(
        train_dp,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=sampler is None,
        sampler=sampler,
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
    layers: Union[nn.Module, Mapping[str, nn.Module]] = None,
    input_column: str = "input",
    device: int = 0,
    run_dir: str = None,
    **kwargs,
):
    model.to(device).eval()

    class ActivationExtractor:
        """Class for extracting activations a targetted intermediate layer"""

        def __init__(self):
            self.activation = None

        def add_hook(self, module, input, output):
            self.activation = output

    layer_to_extractor = {}
    if layers is not None:
        for name, layer in layers.items():
            extractor = ActivationExtractor()
            layer.register_forward_hook(extractor.add_hook)
            layer_to_extractor[name] = extractor

    @torch.no_grad()
    def _score(batch: mk.DataPanel):
        x = batch[input_column].data.to(device)
        out = model(x)  # Run forward pass

        return {
            "output": mk.ClassificationOutputColumn(
                logits=out.cpu(), multi_label=False
            ),
            **{
                f"activation_{name}": extractor.activation.cpu()
                for name, extractor in layer_to_extractor.items()
            },
        }

    dp = dp.update(
        function=_score,
        is_batched_fn=True,
        pbar=True,
        input_columns=[input_column],
        **kwargs,
    )
    return dp

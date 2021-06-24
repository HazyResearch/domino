import os
from typing import Dict, Iterable, List, Union

import meerkat as mk
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import wandb
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from terra import Task
from terra.torch import TerraModule
from torch.utils.data import DataLoader
from torchvision import transforms as transforms

from domino.bss import SourceSeparator
from domino.modeling import ResNet
from domino.utils import PredLogger, TerraCheckpoint

# from domino.data.iwildcam import get_iwildcam_model
# from domino.data.wilds import get_wilds_model


class Classifier(pl.LightningModule, TerraModule):

    DEFAULT_CONFIG = {
        "lr": 1e-4,
        "model_name": "resnet",
        "num_classes": 2,
    }

    def __init__(self, config: dict = None):
        super().__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        if config is not None:
            self.config.update(config)

        self._set_model()

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
            self.model = ResNet(num_classes=self.config["num_classes"])
        elif self.config["model_name"] == "iwildcam":
            self.model = get_iwildcam_model()
        elif self.config["model_name"] == "wilds_model":
            self.model = get_wilds_model(
                model_type=self.config["model_type"],
                d_out=self.config["num_classes"],
                model_path=self.config["model_path"],
            )
        else:
            raise ValueError(f"Model name {self.config['model_name']} not supported.")

    def forward(self, x):
        return self.model(x)

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

    def validation_epoch_end(self, outputs) -> None:
        for metric_name, metric in self.metrics.items():
            self.log(f"valid_{metric_name}", metric.compute())

    def test_epoch_end(self, outputs) -> None:
        return self.validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        return optimizer


def train(
    dp: mk.DataPanel,
    input_column: str,
    target_column: str,
    id_column: str,
    model: Classifier = None,
    config: dict = None,
    num_classes: int = 2,
    max_epochs: int = 50,
    gpus: Union[int, Iterable] = 1,
    num_workers: int = 10,
    batch_size: int = 16,
    ckpt_monitor: str = "valid_accuracy",
    seed: int = 123,
    run_dir: str = None,
    **kwargs,
):
    # Note from https://pytorch-lightning.readthedocs.io/en/0.8.3/multi_gpu.html: Make sure to set the random seed so that each model initializes with the same weights.
    pl.utilities.seed.seed_everything(seed)

    if (model is not None) and (config is not None):
        raise ValueError("Cannot pass both `model` and `config`.")

    if model is None:
        config = {} if config is None else config
        config["num_classes"] = num_classes
        model = Classifier(config)
    logger = WandbLogger(project="domino", save_dir=run_dir)

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
        val_check_interval=10,
        **kwargs,
    )
    dp = mk.DataPanel.from_batch(
        {
            "input": dp[input_column],
            "target": dp[target_column],
            "id": dp[id_column],
            "split": dp["split"],
        }
    )
    train_dl = DataLoader(
        dp.lz[dp["split"].data == "train"],
        batch_size=batch_size,
        num_workers=num_workers,
    )
    valid_dl = DataLoader(
        dp.lz[dp["split"].data == "valid"],
        batch_size=batch_size,
        num_workers=num_workers,
    )
    trainer.fit(model, train_dl, valid_dl)
    wandb.finish()
    return model


@Task.make_task
def score(
    model: Classifier,
    data_df: pd.DataFrame,
    img_column: str,
    target_column: str,
    id_column: str,
    img_transform: callable = None,
    metrics: List[str] = None,
    gpus: Union[int, Iterable] = 1,
    split: str = "valid",
    num_workers: int = 10,
    batch_size: int = 16,
    seed: int = 123,
    run_dir: str = None,
    **kwargs,
):
    # Note from https://pytorch-lightning.readthedocs.io/en/0.8.3/multi_gpu.html: Make sure to set the random seed so that each model initializes with the same weights.
    pl.utilities.seed.seed_everything(seed)

    if img_transform is None:
        img_transform = transforms.Lambda(lambda x: x)
    data_df = data_df[data_df.split == split]
    model.set_metrics(metrics=metrics)
    dataset = Dataset.load_image_dataset(
        data_df.to_dict("records"),
        img_columns=img_column,
        transform=img_transform,
    )
    model.eval()
    trainer = pl.Trainer(
        gpus=gpus,
        default_root_dir=run_dir,
        accelerator=None,
    )
    metrics = trainer.test(
        model=model,
        test_dataloaders=dataset.to_dataloader(
            columns=[img_column, target_column, id_column],
            num_workers=num_workers,
            batch_size=batch_size,
        ),
    )

    preds = model.valid_preds.compute()

    return metrics, preds


@Task.make_task
def fit_bss(
    model: Classifier,
    data_df: pd.DataFrame,
    img_column: str,
    target_column: str,
    id_column: str,
    img_transform: callable,
    config: Dict = None,
    batch_size: int = 16,
    num_workers: int = 4,
    num_epochs: int = 10,
    memmap: bool = False,
    split="valid",
    seed: int = 123,
    run_dir: str = None,
    **kwargs,
):
    pl.utilities.seed.seed_everything(seed)

    if img_transform is None:
        img_transform = transforms.Lambda(lambda x: x)

    if model is None:
        model = Classifier()

    if config is None:
        config = {}

    dataset = Dataset.load_image_dataset(
        data_df[data_df.split == split].to_dict("records"),
        img_columns=img_column,
        transform=img_transform,
    )
    dl = dataset.to_dataloader(
        columns=[img_column, target_column, id_column],
        num_workers=num_workers,
        batch_size=batch_size,
    )

    separator = SourceSeparator(model, config=config)

    separator.fit(dl, log_dir=run_dir, num_epochs=num_epochs, memmap=memmap)

    return separator


@Task.make_task
def compute_bss(
    separator: SourceSeparator,
    data_df: pd.DataFrame,
    img_column: str,
    target_column: str,
    id_column: str,
    img_transform: callable,
    batch_size: int = 16,
    num_workers: int = 4,
    split="valid",
    seed: int = 123,
    run_dir: str = None,
    **kwargs,
):
    pl.utilities.seed.seed_everything(seed)

    if img_transform is None:
        img_transform = transforms.Lambda(lambda x: x)

    dataset = Dataset.load_image_dataset(
        data_df[data_df.split == split].to_dict("records"),
        img_columns=img_column,
        transform=img_transform,
    )
    dl = dataset.to_dataloader(
        columns=[img_column, target_column, id_column],
        num_workers=num_workers,
        batch_size=batch_size,
    )

    return separator.compute(dl)

import json
import os
from typing import Dict, Iterable, List, Mapping, Sequence, Union

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

from domino.bss import SourceSeparator
from domino.modeling import DenseNet, ResNet
from domino.utils import PredLogger, TerraCheckpoint

# from domino.data.iwildcam import get_iwildcam_model
# from domino.data.wilds import get_wilds_model


class Classifier(pl.LightningModule, TerraModule):

    DEFAULT_CONFIG = {
        "lr": 1e-4,
        "model_name": "resnet",
        "arch": "resnet18",
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
    pbar: bool = True,
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

    run_id = int(os.path.basename(run_dir))
    metadata = terra.get_meta(run_id)
    logger = WandbLogger(
        project="domino",
        save_dir=run_dir,
        name=f"{metadata['fn']}-run_id={os.path.basename(run_dir)}",
        tags=[f"{metadata['module']}.{metadata['fn']}"],
        config={} if wandb_config is None else wandb_config,
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
        progress_bar_refresh_rate=None if pbar else 0,
        **kwargs,
    )
    dp = mk.DataPanel.from_batch(
        {
            "input": dp[input_column],
            "target": dp[target_column].astype(int),
            "id": dp[id_column],
            "split": dp["split"],
        }
    )

    train_dp = dp.lz[dp["split"].data == train_split]
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
        dp.lz[dp["split"].data == valid_split],
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
                    layer_to_extractor[f"{name}_{reduction_fn.__name__}"] = extractor
            else:
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

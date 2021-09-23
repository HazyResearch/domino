import json
import os
from typing import Iterable, List, Mapping, Sequence, Union

import meerkat as mk
import pandas as pd
import pytorch_lightning as pl
import terra
import torch
import torch.nn as nn
import torchmetrics
import wandb
from meerkat.columns.lambda_column import PIL
from meerkat.nn import ClassificationOutputColumn
from pytorch_lightning.loggers import WandbLogger
from terra import Task
from terra.pytorch import TerraModule
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from torchvision import transforms as transforms

from domino.eeg_modeling.dense_inception import DenseInception
from domino.modeling import DenseNet, ResNet
from domino.utils import PredLogger, TerraCheckpoint


def default_transform(img: PIL.Image.Image):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(img)


def default_train_transform(img: PIL.Image.Image):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(img)


class Classifier(pl.LightningModule, TerraModule):

    DEFAULT_CONFIG = {
        "lr": 1e-4,
        "model_name": "resnet",
        "arch": "resnet18",
        "pretrained": True,
        "num_classes": 2,
        "transform": default_transform,
        "train_transform": default_train_transform,
    }

    def __init__(self, config: dict = None):
        super().__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        if config is not None:
            self.config.update(config)

        self._set_model()

        metrics = self.config.get("metrics", ["auroc", "accuracy"])
        self.metrics = self._get_metrics(
            metrics, num_classes=self.config["num_classes"]
        )
        self.valid_preds = PredLogger()

    def _get_metrics(self, metrics: List[str], num_classes: int = None):
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
        return nn.ModuleDict(
            {name: metric for name, metric in _metrics.items() if name in metrics}
        )  # metrics need to be child module of the model, https://pytorch-lightning.readthedocs.io/en/stable/metrics.html#metrics-and-devices

    def _set_model(self):
        if self.config["model_name"] == "resnet":
            self.model = ResNet(
                num_classes=self.config["num_classes"],
                arch=self.config["arch"],
                pretrained=self.config["pretrained"],
            )
        elif self.config["model_name"] == "densenet":
            self.model = DenseNet(
                num_classes=self.config["num_classes"], arch=self.config["arch"]
            )
        elif self.config["model_name"] == "dense_inception":
            self.model = DenseInception()
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
            if metric_name == "auroc":
                print("len auroc", len(metric.preds))
            self.log(f"valid_{metric_name}", metric.compute())
            metric.reset()

    def test_epoch_end(self, outputs) -> None:
        return self.validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        return optimizer


class BinaryMTClassifier(Classifier):
    def __init__(self, config: dict = None):
        super().__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        if config is not None:
            self.config.update(config)
        targets = self.config["targets"]
        self.config["num_classes"] = len(targets)
        self._set_model()

        metrics = self.config.get("metrics", ["auroc", "accuracy"])
        self.metrics = nn.ModuleDict(
            {
                target: self._get_metrics(
                    metrics, num_classes=self.config["num_classes"]
                )
                for target in targets
            }
        )
        self.targets = targets
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.valid_preds = PredLogger()

    def training_step(self, batch, batch_idx):
        inputs = batch["input"]
        targets = torch.stack([batch[target] for target in self.targets], axis=-1)
        outs = self.forward(inputs)
        loss = self.loss(outs, targets.to(float))
        self.log("train_loss", loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, sample_id = batch["input"], batch["id"]
        targets = torch.stack([batch[target] for target in self.targets], axis=-1)
        outs = self.forward(inputs)
        loss = self.loss(outs, targets.to(float))
        self.log("valid_loss", loss)

        for idx, target in enumerate(self.targets):
            for metric in self.metrics[target].values():
                metric(torch.sigmoid(outs[:, idx]), targets[:, idx])
            metric.reset()

        self.valid_preds.update(torch.sigmoid(outs), targets, sample_id)

    def validation_epoch_end(self, outputs) -> None:
        for target in self.targets:
            for metric_name, metric in self.metrics[target].items():
                self.log(f"valid/{metric_name}/{target}", metric.compute())


def train(
    dp: mk.DataPanel,
    input_column: str,
    target_column: Union[Sequence[str], str],
    id_column: str,
    model: Classifier = None,
    config: dict = None,
    num_classes: int = 2,
    max_epochs: int = 50,
    samples_per_epoch: int = None,
    gpus: Union[int, Iterable] = 1,
    num_workers: int = 6,
    batch_size: int = 16,
    ckpt_monitor: str = "valid_accuracy",
    train_split: str = "train",
    valid_split: str = "valid",
    wandb_config: dict = None,
    use_terra: bool = True,
    weighted_sampling: bool = False,
    pbar: bool = True,
    seed: int = 123,
    drop_last: bool = False,
    run_dir: str = None,
    **kwargs,
):
    # see here for preprocessing https://github.com/pytorch/vision/issues/39#issuecomment-403701432
    # Note from https://pytorch-lightning.readthedocs.io/en/0.8.3/multi_gpu.html: Make sure to set the random seed so that each model initializes with the same weights.
    pl.utilities.seed.seed_everything(seed)

    if (model is not None) and (config is not None):
        raise ValueError("Cannot pass both `model` and `config`.")

    if model is None:
        config = {} if config is None else config
        config["num_classes"] = num_classes
        if isinstance(target_column, List):
            model = BinaryMTClassifier(config=config)
        else:
            model = Classifier(config)
    if use_terra:
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
    else:
        logger = WandbLogger(
            project="domino",
            save_dir=run_dir,
            name=run_dir,
            config={} if wandb_config is None else wandb_config,
        )
        checkpoint_callbacks = []

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

    if isinstance(target_column, str):
        dp = mk.DataPanel.from_batch(
            {
                "input": dp[input_column],
                "target": dp[target_column].astype(int),
                "id": dp[id_column],
                "split": dp["split"],
            }
        )
    else:
        dp = mk.DataPanel.from_batch(
            {
                "input": dp[input_column],
                "id": dp[id_column],
                "split": dp["split"],
                **{target: dp[target].astype(int) for target in target_column},
            }
        )

    train_dp = dp.lz[dp["split"] == train_split]
    if model.config.get("train_transform", None) is not None:
        train_dp["input"] = train_dp["input"].to_lambda(model.config["train_transform"])
    sampler = None
    if weighted_sampling:
        if isinstance(target_column, Sequence):
            raise ValueError(
                "Weighted sampling with multiple targets is not supported."
            )
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
        drop_last=drop_last,
    )

    valid_dp = dp.lz[dp["split"] == valid_split]
    if model.config.get("transform", None) is not None:
        valid_dp["input"] = valid_dp["input"].to_lambda(model.config["transform"])
    valid_dl = DataLoader(
        valid_dp, batch_size=batch_size, num_workers=num_workers, shuffle=True
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

    dp = dp.view()

    if hasattr(model, "config"):
        if model.config.get("transform", None) is not None:
            dp[input_column] = dp[input_column].to_lambda(model.config["transform"])

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
    dp["probs"] = dp["output"].probabilities()
    return dp

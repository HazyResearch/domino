from __future__ import annotations

import meerkat as mk
import torch
import PIL
from torchvision.models import ResNet as _ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls as resnet_model_urls
from torch.hub import load_state_dict_from_url
from torchvision import transforms
from torch.utils.data import DataLoader
import torchmetrics
from torch import nn
import pytorch_lightning as pl
from typing import Union, Sequence, Iterable, List


def train(
    dp: mk.DataPanel,
    input_column: str,
    target_column: Union[Sequence[str], str],
    id_column: str,
    model: Classifier = None,
    config: dict = None,
    num_classes: int = 2,
    max_epochs: int = 50,
    gpus: Union[int, Iterable] = 1,
    num_workers: int = 6,
    batch_size: int = 16,
    train_split: str = "train",
    valid_split: str = "valid",
    pbar: bool = True,
    seed: int = 123,
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
        model = Classifier(config)

    model.train()
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        log_every_n_steps=1,
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

    train_dp = dp.lz[dp["split"] == train_split]
    if model.config.get("train_transform", None) is not None:
        train_dp["input"] = train_dp["input"].to_lambda(model.config["train_transform"])

    train_dl = DataLoader(
        train_dp,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    valid_dp = dp.lz[dp["split"] == valid_split]
    if model.config.get("transform", None) is not None:
        valid_dp["input"] = valid_dp["input"].to_lambda(model.config["transform"])
    valid_dl = DataLoader(
        valid_dp, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    trainer.fit(model, train_dl, valid_dl)
    return model


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


class ResNet(_ResNet):

    ACTIVATION_DIMS = [64, 128, 256, 512]
    ACTIVATION_WIDTH_HEIGHT = [64, 32, 16, 8]
    RESNET_TO_ARCH = {"resnet18": [2, 2, 2, 2], "resnet50": [3, 4, 6, 3]}

    def __init__(
        self,
        num_classes: int,
        arch: str = "resnet18",
        dropout: float = 0.0,
        pretrained: bool = True,
    ):
        if arch not in self.RESNET_TO_ARCH:
            raise ValueError(
                f"config['classifier'] must be one of: {self.RESNET_TO_ARCH.keys()}"
            )

        block = BasicBlock if arch == "resnet18" else Bottleneck
        super().__init__(block, self.RESNET_TO_ARCH[arch])
        if pretrained:
            state_dict = load_state_dict_from_url(
                resnet_model_urls[arch], progress=True
            )
            self.load_state_dict(state_dict)

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(512 * block.expansion, num_classes)
        )


class Classifier(pl.LightningModule):

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

    def validation_epoch_end(self, outputs) -> None:
        for metric_name, metric in self.metrics.items():
            self.log(f"valid_{metric_name}", metric.compute())
            metric.reset()

    def test_epoch_end(self, outputs) -> None:
        return self.validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        return optimizer

import os
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from domino.loss import SoftCrossEntropyLoss
from domino.utils import nested_getattr


class ActivationExtractor:
    """Class for extracting activations a targetted intermediate layer"""

    def __init__(self):
        self.activation = None

    def add_hook(self, module, input, output):
        self.activation = output


@dataclass
class ActivationDataset:
    acts: torch.Tensor
    outs: torch.Tensor
    targets: torch.Tensor
    batch_size: int

    def __getitem__(self, batch_index):
        start = self.batch_size * batch_index
        end = start + self.batch_size
        return (
            self.acts[start:end],
            self.outs[start:end],
            self.targets[start:end],
        )

    def __len__(self):
        return len(self.acts) // self.batch_size + int(
            len(self.acts) % self.batch_size != 0
        )


class SourceSeparator(nn.Module):
    CONFIG = {
        "target_module": "model.layer4",
        "num_classes": 2,
        "class_idx": None,  # only applicable if num_classes > 2
        "activation_dim": 512,
        "num_components": 5,
        "lr": 1e-1,
        "cond_on_target": True,
        "dropout_prob": 0.0,
        "pred_loss_weight": 100,
        "cov_loss_weight": 1e-7,
    }

    def __init__(self, model: nn.Module, config: dict = None):
        super().__init__()
        self.config = self.CONFIG.copy()
        if config is not None:
            self.config.update(config)

        self.model = model
        try:
            target_module = nested_getattr(model, self.config["target_module"])
        except nn.modules.module.ModuleAttributeError:
            raise ValueError(
                f"model does not have a submodule {self.config['target_module']}"
            )
        self.extractor = ActivationExtractor()
        target_module.register_forward_hook(self.extractor.add_hook)

        # initialize parameters
        self.unmixer = nn.Sequential(
            nn.Linear(self.config["activation_dim"], self.config["activation_dim"]),
            nn.ReLU(),
            nn.Linear(self.config["activation_dim"], self.config["num_components"]),
        )
        input_size = self.config["num_components"] + self.config["cond_on_target"]
        self.component_classifier = nn.Linear(input_size, 2)

    def forward(self, x):
        # compute prediction and activation
        pred = self.model(x)
        act = self.extractor.activation
        act = act.permute((0, 2, 3, 1))

        # concatenate and repeat activations across layers into one tensor
        # act = self._cat_acts_across_layers(act)

        # compute components
        components = self.unmixer(act)
        return pred, act, components

    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int = 10,
        cache_batch_size: int = 128,
        log_dir: str = None,
        device: Union[str, int] = 0,
        memmap: bool = False,
        cache_path: str = None,
        pbars: bool = True,
    ):
        # define parameters
        component_dropout = nn.Dropout(p=self.config["dropout_prob"])
        loss_fn = SoftCrossEntropyLoss(reduction="mean")

        self.to(device)
        self.eval()

        opt = torch.optim.Adam(
            [*self.unmixer.parameters(), *self.component_classifier.parameters()],
            lr=self.config["lr"],
        )

        avg_loss = 0
        writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None

        acts = []
        outs = []
        targets = []
        for idx, (inp, target, _) in enumerate(tqdm(dataloader, desc="cache_acts")):
            inp = inp.to(device)
            out, act, _ = self.forward(inp)

            if out.shape[-1] != self.config["num_classes"]:
                raise ValueError(
                    "The model outputs predictions for more classes than specified in "
                    "the SourceSeparator config."
                )
            if memmap:
                if idx == 0:
                    acts = np.memmap(
                        filename=os.path.join("/home/sabri/", "acts.dat"),
                        dtype=np.double,
                        mode="w+",
                        shape=(len(dataloader.dataset), *act.shape[1:]),
                    )
                start = idx * dataloader.batch_size
                acts[start : start + act.shape[0]] = act.detach().cpu()
            else:
                acts.append(act.detach().cpu())
            outs.append(out.detach().cpu())
            targets.append(target)
        if memmap:
            acts = torch.from_numpy(acts)
        else:
            acts = torch.cat(acts)
        outs = torch.cat(outs)
        targets = torch.cat(targets)

        # https://pytorch.org/docs/stable/data.html#disable-automatic-batching
        cache_dataloader = DataLoader(
            ActivationDataset(acts, outs, targets, cache_batch_size),
            batch_size=None,
            batch_sampler=None,
        )
        with tqdm(
            total=num_epochs,
            disable=not pbars,
            desc=f"fit_source_separator",
        ) as batch_t:
            for epoch_idx in range(num_epochs):
                for batch_idx, (act, out, target) in enumerate(cache_dataloader):
                    act, out, target = (
                        act.to(device).to(torch.float),
                        out.to(device),
                        target.to(device),
                    )
                    components = self.unmixer(act)
                    components = components.view(
                        components.shape[0], -1, components.shape[-1]
                    )

                    # compute spatial covariance
                    # center each position's component around the global mean (across
                    # examples) for that component (dim=(1, 2))
                    comps_centered = components - components.mean(
                        dim=(1, 2), keepdim=True
                    )
                    comps_cov = torch.matmul(
                        comps_centered.permute(0, 2, 1),
                        comps_centered.permute(0, 1, 2),
                    )

                    # predict models prediction from components
                    # components_max = components.mean(dim=1, keepdim=False)
                    comps_max = components.max(dim=1, keepdim=False)[0]  # trying max
                    comps_max_centered = comps_max - comps_max.mean(
                        dim=-1, keepdim=True
                    )
                    comps_max_centered = component_dropout(comps_max_centered)

                    if self.config["cond_on_target"]:
                        comps_max_centered = torch.cat(
                            [comps_max_centered, target.unsqueeze(-1)], dim=-1
                        )
                    components_out = self.component_classifier(comps_max_centered)

                    if self.config["num_classes"] > 2:
                        # if multi-class problem, convert into binary prediction between
                        # "class_idx" and the other classes
                        out = out[:, self.config["class_idx"]]
                        out = torch.stack((1 - out, out), dim=-1)

                    # compute loss
                    components_out = torch.softmax(components_out, dim=-1)
                    pred_loss = loss_fn(out, components_out)
                    cov_loss = (torch.triu(comps_cov, diagonal=1) ** 2).mean()
                    loss = (
                        self.config["pred_loss_weight"] * pred_loss
                        + self.config["cov_loss_weight"] * cov_loss
                    )

                    # backward pass
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    # compute which example idx we're on
                    iter_idx = (
                        epoch_idx * len(cache_dataloader) + batch_idx
                    ) * cache_batch_size

                    # compute average loss and update progress bar
                    loss = loss.cpu().detach().numpy()
                    if writer is not None:
                        for name, l in [
                            ("loss", loss),
                            ("cov_loss", cov_loss),
                            ("pred_loss", pred_loss),
                        ]:
                            writer.add_scalar(
                                tag=f"{name}/train",
                                scalar_value=l,
                                global_step=iter_idx,
                            )
                    avg_loss = ((avg_loss * batch_idx) + loss) / (batch_idx + 1)

                batch_t.set_postfix(loss="{:05.3f}".format(float(avg_loss)))
                batch_t.update()

    def compute(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: Union[str, int] = 0,
        pbars: bool = True,
    ) -> torch.Tensor:
        self.to(device)
        self.eval()

        components_list, out_list, act_list, target_list = [], [], [], []
        with tqdm(total=len(dataloader), disable=not pbars, desc="components") as t:
            for inp, target, _ in dataloader:
                inp = inp.to(device)
                out, acts, components = self.forward(inp)
                act_list.append([act.detach().cpu() for act in acts])

                components = components.view(
                    components.shape[0], -1, components.shape[-1]
                )

                if self.config["num_classes"] > 2:
                    # if multi-class problem, convert into binary prediction between
                    # "class_idx" and the other classes
                    out = out[:, self.config["class_idx"]]
                    out = torch.stack((1 - out, out), dim=-1)
                    target = (target == self.config["class_idx"]).to(torch.long)

                # center components
                components -= components.mean(dim=1, keepdim=True)
                out_list.append(out.detach().cpu())
                target_list.append(target)
                components_list.append(components.detach().cpu())
                t.update()

        outs = torch.cat(out_list)
        components = torch.cat(components_list)
        targets = torch.cat(target_list)
        return components, outs, targets

    @staticmethod
    def _cat_acts_across_layers(acts: List[torch.Tensor]):
        """repeat activations in lower layers so they match the height and width of the first layer"""
        if len(acts) == 1:
            return acts[0]

        width, height = acts[0].shape[-2:]
        new_acts = [
            act.repeat_interleave(width // act.shape[-2], dim=2).repeat_interleave(
                height // act.shape[-1], dim=3
            )
            for act in acts
        ]
        return torch.cat(new_acts, dim=1)

    def write(self, path):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config,
                "model_cls": type(self.model),
                "model_config": self.model.config,
            },
            path,
        )

    @classmethod
    def read(cls, path):
        dct = torch.load(path, map_location="cpu")
        model = dct["model_cls"](config=dct["model_config"])
        separator = cls(model=model, config=dct["config"])
        separator.load_state_dict(dct["state_dict"])
        return separator

    __terra_read__ = read

    __terra_write__ = write

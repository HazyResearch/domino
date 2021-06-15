import os

# from dataclasses import dataclass
from typing import List, Mapping, Union

import numpy as np
import torch
import torch.nn as nn
from mosaic import DataPanel, NumpyArrayColumn

# from torch.utils.data import TensorDataset
# from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from domino.loss import SoftCrossEntropyLoss

# from domino.utils import nested_getattr


class ActivationExtractor:
    """Class for extracting activations a targetted intermediate layer"""

    def __init__(self):
        self.activation = None

    def add_hook(self, module, input, output):
        self.activation = output


class SourceSeparator(nn.Module):
    CONFIG = {
        "num_classes": 2,
        "class_idx": None,  # only applicable if num_classes > 2
        "activation_dim": 512,
        "num_components": 5,
        "lr": 1e-1,
        "cond_on_target": True,
        "dropout_prob": 0.0,
        "pred_loss_weight": 100,
        "cov_loss_weight": 1e-7,
        "batch_size": 2048,
    }

    def __init__(self, model: nn.Module, config: dict = None):
        super().__init__()
        self.config = self.CONFIG.copy()
        if config is not None:
            self.config.update(config)

        self.model = model

        # initialize parameters
        self.unmixer = nn.Sequential(
            nn.Linear(self.config["activation_dim"], self.config["activation_dim"]),
            nn.ReLU(),
            nn.Linear(self.config["activation_dim"], self.config["num_components"]),
        )
        input_size = self.config["num_components"] + self.config["cond_on_target"]
        self.component_classifier = nn.Linear(input_size, 2)

    def prepare_dp(
        self,
        dp: DataPanel,
        layers: Union[nn.Module, Mapping[str, nn.Module]] = None,
        device: int = 0,
        input_col: str = "input",
        *args,
        **kwargs,
    ):
        extractor = ActivationExtractor()
        if layers is None:
            layers = {"0": self.model.model.layer4}
        elif isinstance(layers, nn.Module):
            layers = {"0": layers}
        elif not isinstance(layers, Mapping):
            raise ValueError(
                "Layers must be `nn.Module` or mapping from str to `nn.Module`"
            )

        layer_to_extractor = {}
        for name, layer in layers.items():
            extractor = ActivationExtractor()
            layer.register_forward_hook(extractor.add_hook)
            layer_to_extractor[name] = extractor

        self.model.to(device).eval()

        @torch.no_grad()
        def predict(batch: dict):
            out = torch.softmax(self.model(batch[input_col].data.to(device)), axis=-1)
            return {
                "pred": out.cpu().numpy().argmax(axis=-1),
                "probs": out.cpu().numpy(),
                **{
                    f"activation_{name}": extractor.activation.cpu().numpy()
                    for name, extractor in layer_to_extractor.items()
                },
            }

        return dp.update(
            function=predict,
            is_batched_fn=True,
            input_columns=[input_col],
            pbar=True,
            num_workers=6,
            *args,
            **kwargs,
        )

    def forward(self, act: torch.Tensor):
        """[summary]

        Args:
            act (torch.Tensor): tensor of shape (batch_size, channels, height, width)

        Returns:
            [type]: [description]
        """
        # compute prediction and activation
        act = act.permute((0, 2, 3, 1))

        # concatenate and repeat activations across layers into one tensor
        # act = self._cat_acts_across_layers(act)

        # compute components
        components = self.unmixer(act)
        return components

    def fit(
        self,
        dp: DataPanel,
        num_epochs: int = 10,
        log_dir: str = None,
        device: Union[str, int] = 0,
        num_workers: int = 4,
        pbars: bool = True,
        target_col: str = "y",
        activation_col: str = "activation",
    ):
        required_cols = [activation_col, target_col, "probs"]
        if not set(required_cols).issubset(dp.columns):
            raise ValueError(f"DataPanel `dp` must have columns {required_cols}")

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

        with dp.format(columns=required_cols), tqdm(
            total=num_epochs,
            disable=not pbars,
            desc="fit_source_separator",
        ) as batch_t:
            for epoch_idx in range(num_epochs):
                for batch_idx, batch in enumerate(
                    dp.batch(
                        batch_size=self.config["batch_size"], num_workers=num_workers
                    )
                ):
                    act, out, target = (
                        batch[activation_col].to_tensor().to(device).to(torch.float),
                        batch["probs"].to_tensor().to(device),
                        batch[target_col].to_tensor().to(device),
                    )
                    # put channels last
                    act = act.permute((0, 2, 3, 1))

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
                    print("cov_loss:", cov_loss.cpu().detach().numpy())
                    print("pred_loss: ", pred_loss.cpu().detach().numpy())

                    # backward pass
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    # compute which example idx we're on
                    iter_idx = (epoch_idx * len(dp)) + (
                        batch_idx * self.config["batch_size"]
                    )

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
        dp: DataPanel,
        device: Union[str, int] = 0,
        num_workers: int = 4,
        batch_size: int = 1024,
        activation_col: str = "activation",
        *args,
        **kwargs,
    ) -> torch.Tensor:
        self.to(device)
        self.eval()

        required_cols = [activation_col]
        if not set(required_cols).issubset(dp.columns):
            raise ValueError(f"DataPanel `dp` must have columns {required_cols}")

        @torch.no_grad()
        def compute_components(batch: dict):
            components = self.forward(batch[activation_col].to_tensor().to(0))
            # components = components.permute((3, 0, 1, 2))
            return {"components": components.cpu().numpy()}

        return dp.update(
            function=compute_components,
            batch_size=batch_size,
            batched=True,
            num_workers=num_workers,
            overwrite=True,
            input_columns=[activation_col],
            *args,
            **kwargs,
        )

    def solicit_feedback(
        self,
        dp: DataPanel,
        comp_idx: int = 0,
        size=(224, 224),
        overwrite: bool = False,
    ):
        import gradio as gr

        # TODO: find a better solution for this
        os.makedirs("flagged", exist_ok=True)
        dir_path = "flagged"
        component = dp["components"].max(axis=(1, 2))._data[:, comp_idx]

        if overwrite or "feedback_label" not in dp.column_names:
            dp.add_column(
                "feedback_label",
                NumpyArrayColumn(["unlabeled"] * len(dp)),
                overwrite=overwrite,
            )

        if overwrite or "feedback_mask" not in dp.column_names:
            dp.add_column(
                "feedback_mask",
                NumpyArrayColumn(np.zeros((len(dp), *size))),
                overwrite=overwrite,
            )

        # prepare examples
        examples = []
        for rank, example_idx in enumerate((-component).argsort()[0:30]):
            example_idx = int(example_idx)
            image = dp["raw_input"][example_idx]
            label = dp["category"][example_idx]
            image_path = os.path.join(dir_path, f"image_{example_idx}.jpg")
            image.save(image_path)
            examples.append(
                [
                    rank,
                    example_idx,
                    label,
                    image_path,
                    dp["feedback_label"][example_idx],
                ]
            )

        # define feedback function
        feedback = []

        def submit_feedback(rank, example_idx, label, img, feedback_label):
            mask = (img == np.array([0, 169, 255])).all(axis=-1)
            feedback.append({"mask": mask, "example_idx": example_idx, "label": label})
            dp["feedback_label"][example_idx] = feedback_label
            dp["feedback_mask"][example_idx] = mask
            return []

        iface = gr.Interface(
            submit_feedback,
            [
                gr.inputs.Number(),
                gr.inputs.Number(),
                gr.inputs.Textbox(),
                gr.inputs.Image(shape=size),
                gr.inputs.Radio(choices=["positive", "negative", "abstain"]),
            ],
            outputs=[],
            examples=examples,
            layout="vertical",
        )
        return iface.launch(inbrowser=True, inline=False)

    @staticmethod
    def _cat_acts_across_layers(acts: List[torch.Tensor]):
        """
        repeat activations in lower layers so they
        match the height and width of the first layer
        """
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
